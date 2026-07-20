"""Validation helpers for the Korean equity universe."""

from __future__ import annotations

from typing import Final

import pandas as pd

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "rank",
    "ticker",
    "company_name",
    "market",
    "sector",
    "industry",
    "investment_theme",
    "universe_role",
    "selection_reason",
    "data_start_date",
    "is_active",
    "notes",
)
ALLOWED_MARKETS: Final[frozenset[str]] = frozenset({"KOSPI", "KOSDAQ"})
ALLOWED_UNIVERSE_ROLES: Final[frozenset[str]] = frozenset(
    {"Core", "Growth", "Cyclical", "Defensive"}
)
EXPECTED_UNIVERSE_SIZE: Final[int] = 100


class UniverseValidationError(ValueError):
    """Raised when the universe file fails validation."""


def validate_universe(df: pd.DataFrame) -> list[str]:
    """Return validation errors for a normalized universe DataFrame."""
    errors: list[str] = []

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return errors

    if len(df) != EXPECTED_UNIVERSE_SIZE:
        errors.append(
            f"Universe must contain exactly {EXPECTED_UNIVERSE_SIZE} rows; found {len(df)}"
        )

    null_columns = [column for column in REQUIRED_COLUMNS if df[column].isna().any()]
    if null_columns:
        errors.append(f"Missing values found in columns: {', '.join(null_columns)}")

    duplicate_tickers = sorted(df.loc[df["ticker"].duplicated(keep=False), "ticker"].unique())
    if duplicate_tickers:
        errors.append(f"Duplicate ticker values found: {', '.join(duplicate_tickers)}")

    duplicate_companies = sorted(
        df.loc[df["company_name"].duplicated(keep=False), "company_name"].unique()
    )
    if duplicate_companies:
        errors.append(f"Duplicate company_name values found: {', '.join(duplicate_companies)}")

    invalid_markets = sorted(set(df["market"].dropna()) - ALLOWED_MARKETS)
    if invalid_markets:
        errors.append(
            "Invalid market values found: "
            f"{', '.join(invalid_markets)}. Allowed values: {', '.join(sorted(ALLOWED_MARKETS))}"
        )

    invalid_roles = sorted(set(df["universe_role"].dropna()) - ALLOWED_UNIVERSE_ROLES)
    if invalid_roles:
        errors.append(
            "Invalid universe_role values found: "
            f"{', '.join(invalid_roles)}. Allowed values: "
            f"{', '.join(sorted(ALLOWED_UNIVERSE_ROLES))}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df["data_start_date"]):
        errors.append("data_start_date must be converted to a datetime dtype")

    if not pd.api.types.is_bool_dtype(df["is_active"]):
        errors.append("is_active must be converted to a bool dtype")

    return errors


def raise_for_validation_errors(errors: list[str]) -> None:
    """Raise a clear validation exception when errors exist."""
    if errors:
        message = "Universe validation failed:\n- " + "\n- ".join(errors)
        raise UniverseValidationError(message)


__all__ = [
    "ALLOWED_MARKETS",
    "ALLOWED_UNIVERSE_ROLES",
    "EXPECTED_UNIVERSE_SIZE",
    "REQUIRED_COLUMNS",
    "UniverseValidationError",
    "raise_for_validation_errors",
    "validate_universe",
]
