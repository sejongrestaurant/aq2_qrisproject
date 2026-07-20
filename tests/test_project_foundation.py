"""Smoke tests for the project foundation."""

from __future__ import annotations

from pathlib import Path

from src.config.settings import Settings


def test_default_settings_point_to_project_paths() -> None:
    """Default settings should expose the expected local project paths."""
    settings = Settings()

    assert settings.app_name == "must30-active-etf"
    assert settings.universe_csv_path == Path("data/universe/korea_active_etf_universe_100.csv")
    assert settings.raw_data_dir == Path("data/raw")
    assert settings.processed_data_dir == Path("data/processed")
