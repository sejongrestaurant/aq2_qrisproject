"""Tests for the end-to-end pipeline orchestrator."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.pipeline.run_pipeline import PipelineExecutionError, run_pipeline
from src.pipeline.stages import PipelineContext, StageResult
from src.pipeline.state import load_pipeline_state


def test_pipeline_runs_requested_stage_range(tmp_path: Path) -> None:
    """The orchestrator should execute only the requested stage slice."""
    calls: list[str] = []
    context = _context(tmp_path)
    registry = {
        name: _stage(name, calls)
        for name in [
            "validate_universe",
            "collect_prices",
            "collect_fundamentals",
            "validate_data",
        ]
    }

    records = run_pipeline(
        context,
        from_stage="collect_prices",
        to_stage="validate_data",
        state_path=tmp_path / "state.json",
        registry=registry,
    )

    assert calls == ["collect_prices", "collect_fundamentals", "validate_data"]
    assert [record.status for record in records] == ["completed", "completed", "completed"]


def test_pipeline_dry_run_does_not_execute_stages(tmp_path: Path) -> None:
    """Dry-run mode should report stages without calling stage functions."""
    calls: list[str] = []
    context = _context(tmp_path)
    registry = {"validate_universe": _stage("validate_universe", calls)}

    records = run_pipeline(
        context,
        from_stage="validate_universe",
        to_stage="validate_universe",
        dry_run=True,
        state_path=tmp_path / "state.json",
        registry=registry,
    )

    assert calls == []
    assert records[0].status == "skipped"
    assert "dry-run" in records[0].message


def test_pipeline_skips_completed_same_config(tmp_path: Path) -> None:
    """Re-running the same date/config should skip already completed stages."""
    calls: list[str] = []
    context = _context(tmp_path)
    registry = {"validate_universe": _stage("validate_universe", calls)}
    state_path = tmp_path / "state.json"

    run_pipeline(
        context,
        from_stage="validate_universe",
        to_stage="validate_universe",
        state_path=state_path,
        registry=registry,
    )
    records = run_pipeline(
        context,
        from_stage="validate_universe",
        to_stage="validate_universe",
        state_path=state_path,
        registry=registry,
    )

    assert calls == ["validate_universe"]
    assert records[0].status == "skipped"
    assert "already completed" in records[0].message


def test_pipeline_restarts_from_failed_stage(tmp_path: Path) -> None:
    """A later run without from-stage should resume from the first failed stage."""
    calls: list[str] = []
    context = _context(tmp_path)
    state_path = tmp_path / "state.json"

    def failing_stage(_: PipelineContext) -> StageResult:
        calls.append("collect_prices")
        raise RuntimeError("collector unavailable")

    with pytest.raises(PipelineExecutionError):
        run_pipeline(
            context,
            from_stage="validate_universe",
            to_stage="collect_fundamentals",
            state_path=state_path,
            registry={
                "validate_universe": _stage("validate_universe", calls),
                "collect_prices": failing_stage,
                "collect_fundamentals": _stage("collect_fundamentals", calls),
            },
        )

    calls.clear()
    run_pipeline(
        context,
        to_stage="collect_fundamentals",
        state_path=state_path,
        registry={
            "collect_prices": _stage("collect_prices", calls),
            "collect_fundamentals": _stage("collect_fundamentals", calls),
        },
    )

    assert calls == ["collect_prices", "collect_fundamentals"]
    state = load_pipeline_state(state_path, _config(context))
    assert state.stages["collect_prices"].status == "completed"


def test_pipeline_blocks_when_collection_failures_exceed_threshold(tmp_path: Path) -> None:
    """Collection stages may continue under threshold but must fail above it."""
    context = PipelineContext(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        output_dir=tmp_path,
        collection_failure_threshold=0.10,
    )

    def bad_collection(_: PipelineContext) -> StageResult:
        return StageResult(
            stage_name="collect_prices",
            message="collection completed with too many failures",
            metrics={"target_count": 10, "failure_count": 2},
        )

    with pytest.raises(PipelineExecutionError, match="failure ratio"):
        run_pipeline(
            context,
            from_stage="collect_prices",
            to_stage="collect_prices",
            state_path=tmp_path / "state.json",
            registry={"collect_prices": bad_collection},
        )


def _context(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        universe_path=tmp_path / "universe.csv",
        output_dir=tmp_path,
    )


def _stage(name: str, calls: list[str]) -> object:
    def inner(_: PipelineContext) -> StageResult:
        calls.append(name)
        return StageResult(stage_name=name, message=f"{name} done")

    return inner


def _config(context: PipelineContext) -> dict[str, object]:
    return {
        "start_date": context.start_date.isoformat(),
        "end_date": context.end_date.isoformat(),
        "universe_path": str(context.universe_path),
        "database_url": context.database_url,
        "output_dir": str(context.output_dir),
        "input_dir": str(context.input_dir),
        "strategy": context.strategy,
        "transaction_cost": context.transaction_cost,
        "target_size": context.target_size,
        "full_refresh": context.full_refresh,
        "collection_failure_threshold": context.collection_failure_threshold,
        "from_stage": None,
        "to_stage": None,
        "dry_run": False,
    }
