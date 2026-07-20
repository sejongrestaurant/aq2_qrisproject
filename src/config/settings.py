"""Project settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the project foundation."""

    app_name: str = "must30-active-etf"
    app_env: str = "local"
    log_level: str = "INFO"
    database_url: str = "sqlite:///database/must30.db"
    universe_csv_path: Path = Path("data/universe/korea_active_etf_universe_100.csv")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    dart_api_key: str = ""
    api_cors_origins: str = "*"
    api_max_page_size: int = 1000
    api_default_page_size: int = 100

    @property
    def cors_origin_list(self) -> list[str]:
        """Return configured CORS origins as a list."""
        if self.api_cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Return cached project settings."""
    return Settings()


__all__ = ["Settings", "get_settings"]
