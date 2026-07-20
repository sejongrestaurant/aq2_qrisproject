"""End-to-end CLI orchestration for the MUST30 research pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from src.pipeline.stages import (
    PIPELINE_STAGES,
    PipelineContext,
    StageFunction,
    StageResult,
    build_stage_registry,
    default_context,
)
from src.pipeline.state import (
    PipelineState,
    StageExecutionRecord,
    load_pipeline_state,
    save_pipeline_state,
)

LOGGER = logging.getLogger(__name__)


class PipelineExecutionError(RuntimeError):
    """Raised when a pipeline stage fails or breaches a blocking threshold."""


def run_pipeline(
    context: PipelineContext,
    *,
    from_stage: str | None = None,
    to_stage: str | None = None,
    dry_run: bool = False,
    full_refresh: bool | None = None,
    state_path: Path | None = None,
    config_output_path: Path | None = None,
    registry: dict[str, StageFunction] | None = None,
) -> list[StageExecutionRecord]:
    """Run the configured pipeline stages and return stage execution records.

    The orchestrator uses only point-in-time artifacts produced by earlier stages. Month-end
    signals and next-day execution remain inside the backtest and rebalance modules, so this
    wrapper does not introduce same-day execution or future-data joins.
    """
    resolved_context = (
        context
        if full_refresh is None
        else PipelineContext(**{**asdict(context), "full_refresh": bool(full_refresh)})
    )
    stages = _stage_slice(from_stage=from_stage, to_stage=to_stage)
    stage_registry = registry or build_stage_registry()

    state_file = state_path or resolved_context.output_path("pipeline_state.json")
    config_file = config_output_path or resolved_context.output_path("pipeline_config.json")
    config = _context_to_config(resolved_context, from_stage, to_stage, dry_run)
    state = load_pipeline_state(state_file, config)
    _write_json(config_file, config)

    first_failed = state.first_failed_stage(list(PIPELINE_STAGES))
    if from_stage is None and first_failed in stages and not resolved_context.full_refresh:
        stages = stages[stages.index(first_failed) :]
    _validate_registry(stages, stage_registry)

    records: list[StageExecutionRecord] = []
    for stage_name in stages:
        if dry_run:
            record = _skipped_record(stage_name, state.config_hash, "dry-run")
            records.append(record)
            continue

        if not resolved_context.full_refresh and state.is_stage_completed(stage_name):
            record = _skipped_record(
                stage_name,
                state.config_hash,
                "already completed for the same pipeline configuration",
            )
            records.append(record)
            continue

        record = _execute_stage(stage_name, stage_registry[stage_name], resolved_context, state)
        state.set_stage(record)
        save_pipeline_state(state, state_file)
        records.append(record)
        if record.status == "failed":
            raise PipelineExecutionError(record.message)
        _raise_if_collection_threshold_exceeded(
            record, resolved_context.collection_failure_threshold
        )

    if dry_run:
        LOGGER.info("Dry run completed. No stages were executed.")
    _print_summary(records)
    return records


def main(argv: list[str] | None = None) -> None:
    """Run the end-to-end pipeline CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    _configure_logging(output_dir / "logs" / "pipeline.log")
    try:
        context = default_context(
            start_date=_parse_date(args.start_date),
            end_date=_parse_date(args.end_date),
            universe_path=Path(args.universe_path) if args.universe_path else None,
            database_url=args.database_url,
            output_dir=output_dir,
            input_dir=Path(args.input_dir),
            strategy=args.strategy,
            transaction_cost=float(args.transaction_cost),
            target_size=int(args.target_size),
            full_refresh=bool(args.full_refresh),
            collection_failure_threshold=float(args.collection_failure_threshold),
        )
        run_pipeline(
            context,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            dry_run=bool(args.dry_run),
            full_refresh=bool(args.full_refresh),
            state_path=Path(args.state_path) if args.state_path else None,
        )
    except Exception as error:
        LOGGER.exception("Pipeline failed: %s", error)
        print("Pipeline execution failed.", file=sys.stderr)
        print(f"Reason: {error}", file=sys.stderr)
        print("Stack trace:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full MUST30 pipeline.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--from-stage", choices=PIPELINE_STAGES, default=None)
    parser.add_argument("--to-stage", choices=PIPELINE_STAGES, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--universe-path", default=None)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/pipeline")
    parser.add_argument("--state-path", default=None)
    parser.add_argument(
        "--strategy",
        choices=["equal_weight", "score_weight", "rank_weight"],
        default="score_weight",
    )
    parser.add_argument("--transaction-cost", type=float, default=0.002)
    parser.add_argument("--target-size", type=int, default=30)
    parser.add_argument("--collection-failure-threshold", type=float, default=0.10)
    return parser


def _execute_stage(
    stage_name: str,
    stage_function: StageFunction,
    context: PipelineContext,
    state: PipelineState,
) -> StageExecutionRecord:
    started_at = datetime.now().astimezone().replace(microsecond=0).isoformat()
    started = time.perf_counter()
    LOGGER.info("Starting stage: %s", stage_name)
    state.set_stage(
        StageExecutionRecord(
            stage_name=stage_name,
            status="running",
            started_at=started_at,
            message="Stage is running.",
            config_hash=state.config_hash,
        )
    )
    try:
        result = stage_function(context)
        elapsed = time.perf_counter() - started
        record = _completed_record(result, state.config_hash, started_at, elapsed)
        LOGGER.info("Completed stage: %s in %.2fs", stage_name, elapsed)
        return record
    except Exception as error:
        elapsed = time.perf_counter() - started
        stack = traceback.format_exc()
        LOGGER.exception("Stage failed: %s", stage_name)
        return StageExecutionRecord(
            stage_name=stage_name,
            status="failed",
            started_at=started_at,
            finished_at=datetime.now().astimezone().replace(microsecond=0).isoformat(),
            elapsed_seconds=elapsed,
            message=f"Stage '{stage_name}' failed. Check the log file for details.",
            error=str(error),
            traceback=stack,
            config_hash=state.config_hash,
        )


def _completed_record(
    result: StageResult,
    config_hash: str,
    started_at: str,
    elapsed_seconds: float,
) -> StageExecutionRecord:
    return StageExecutionRecord(
        stage_name=result.stage_name,
        status="completed",
        started_at=started_at,
        finished_at=datetime.now().astimezone().replace(microsecond=0).isoformat(),
        elapsed_seconds=elapsed_seconds,
        message=result.message,
        output_paths=[str(path) for path in result.output_paths],
        metrics=result.metrics,
        config_hash=config_hash,
    )


def _skipped_record(stage_name: str, config_hash: str, reason: str) -> StageExecutionRecord:
    return StageExecutionRecord(
        stage_name=stage_name,
        status="skipped",
        started_at=None,
        finished_at=datetime.now().astimezone().replace(microsecond=0).isoformat(),
        elapsed_seconds=0.0,
        message=f"Stage skipped: {reason}.",
        config_hash=config_hash,
    )


def _stage_slice(*, from_stage: str | None, to_stage: str | None) -> list[str]:
    stages = list(PIPELINE_STAGES)
    start_index = stages.index(from_stage) if from_stage else 0
    end_index = stages.index(to_stage) if to_stage else len(stages) - 1
    if start_index > end_index:
        raise ValueError("--from-stage must be earlier than or equal to --to-stage")
    return stages[start_index : end_index + 1]


def _validate_registry(stages: list[str], registry: dict[str, StageFunction]) -> None:
    missing = [stage for stage in stages if stage not in registry]
    if missing:
        raise ValueError(f"Missing stage functions: {', '.join(missing)}")


def _raise_if_collection_threshold_exceeded(
    record: StageExecutionRecord,
    threshold: float,
) -> None:
    if record.stage_name not in {"collect_prices", "collect_fundamentals"}:
        return
    target = float(record.metrics.get("target_count", 0.0))
    failures = float(record.metrics.get("failure_count", 0.0))
    if target <= 0.0:
        return
    failure_ratio = failures / target
    if failure_ratio > threshold:
        raise PipelineExecutionError(
            f"{record.stage_name} failure ratio {failure_ratio:.2%} exceeds threshold {threshold:.2%}"
        )


def _context_to_config(
    context: PipelineContext,
    from_stage: str | None,
    to_stage: str | None,
    dry_run: bool,
) -> dict[str, object]:
    values = asdict(context)
    values["start_date"] = context.start_date.isoformat()
    values["end_date"] = context.end_date.isoformat()
    values["universe_path"] = str(context.universe_path)
    values["output_dir"] = str(context.output_dir)
    values["input_dir"] = str(context.input_dir)
    values["from_stage"] = from_stage
    values["to_stage"] = to_stage
    values["dry_run"] = dry_run
    return values


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


def _print_summary(records: list[StageExecutionRecord]) -> None:
    print("Pipeline summary")
    for record in records:
        print(
            f"- {record.stage_name}: {record.status} "
            f"({record.elapsed_seconds:.2f}s) {record.message}"
        )


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


if __name__ == "__main__":
    main()
