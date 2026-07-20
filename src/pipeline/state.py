"""Persistent execution state for the MUST30 pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

PipelineStageStatus = Literal["pending", "running", "completed", "failed", "skipped"]


@dataclass(frozen=True)
class StageExecutionRecord:
    """Serializable state for one pipeline stage execution."""

    stage_name: str
    status: PipelineStageStatus
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    message: str = ""
    error: str | None = None
    traceback: str | None = None
    output_paths: list[str] = field(default_factory=list)
    metrics: dict[str, int | float | str | bool] = field(default_factory=dict)
    config_hash: str = ""


@dataclass
class PipelineState:
    """JSON-backed pipeline state used for restart and duplicate-run checks."""

    run_id: str
    config_hash: str
    config: dict[str, object]
    stages: dict[str, StageExecutionRecord] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())

    def is_stage_completed(self, stage_name: str) -> bool:
        """Return True when a stage completed for the current config hash."""
        record = self.stages.get(stage_name)
        return (
            record is not None
            and record.status == "completed"
            and record.config_hash == self.config_hash
        )

    def first_failed_stage(self, ordered_stages: list[str]) -> str | None:
        """Return the earliest failed stage in the configured stage order."""
        for stage_name in ordered_stages:
            record = self.stages.get(stage_name)
            if record is not None and record.status == "failed":
                return stage_name
        return None

    def set_stage(self, record: StageExecutionRecord) -> None:
        """Update one stage record and refresh the state timestamp."""
        self.stages[record.stage_name] = record
        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the state."""
        values = asdict(self)
        values["stages"] = {
            stage_name: asdict(record) for stage_name, record in self.stages.items()
        }
        return values


def config_hash(config: dict[str, object]) -> str:
    """Return a deterministic hash for a pipeline config dictionary."""
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_id_for_config(config: dict[str, object]) -> str:
    """Return the deterministic run id used to avoid duplicate same-date storage."""
    digest = config_hash(
        {
            "start_date": config.get("start_date"),
            "end_date": config.get("end_date"),
            "strategy": config.get("strategy"),
            "transaction_cost": config.get("transaction_cost"),
            "target_size": config.get("target_size"),
        }
    )
    return f"must30_{config.get('start_date')}_{config.get('end_date')}_{digest}"


def load_pipeline_state(path: str | Path, config: dict[str, object]) -> PipelineState:
    """Load an existing state file or create a fresh state for the given config."""
    state_path = Path(path)
    current_hash = config_hash(config)
    if not state_path.exists():
        return PipelineState(
            run_id=run_id_for_config(config),
            config_hash=current_hash,
            config=config,
        )

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    records = {
        str(stage_name): StageExecutionRecord(**record)
        for stage_name, record in dict(raw.get("stages", {})).items()
    }
    return PipelineState(
        run_id=str(raw.get("run_id") or run_id_for_config(config)),
        config_hash=current_hash,
        config=config,
        stages=records,
        created_at=str(raw.get("created_at") or _utc_now()),
        updated_at=str(raw.get("updated_at") or _utc_now()),
    )


def save_pipeline_state(state: PipelineState, path: str | Path) -> None:
    """Persist pipeline state to JSON."""
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "PipelineStageStatus",
    "PipelineState",
    "StageExecutionRecord",
    "config_hash",
    "load_pipeline_state",
    "run_id_for_config",
    "save_pipeline_state",
]
