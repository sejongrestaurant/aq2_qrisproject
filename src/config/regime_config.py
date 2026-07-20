"""Configuration for rule-based market regime classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RegimeName = Literal["Risk-On", "Neutral", "Risk-Off"]


@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds and smoothing options for market regime classification."""

    kospi_ma_window: int = 200
    kospi_momentum_window: int = 60
    kospi_volatility_window: int = 20
    breadth_ma_window: int = 120
    annualization_days: int = 252
    risk_on_breadth_threshold: float = 0.55
    risk_off_breadth_threshold: float = 0.45
    risk_on_momentum_threshold: float = 0.0
    risk_off_momentum_threshold: float = 0.0
    confirmation_days: int = 1
    use_hysteresis: bool = False
    hysteresis_breadth_buffer: float = 0.03
    hysteresis_momentum_buffer: float = 0.01
    risk_on_equity_weight: float = 1.0
    neutral_equity_weight: float = 0.8
    risk_off_equity_weight: float = 0.5

    def validate(self) -> None:
        """Validate windows, thresholds, and allocation weights."""
        windows = {
            "kospi_ma_window": self.kospi_ma_window,
            "kospi_momentum_window": self.kospi_momentum_window,
            "kospi_volatility_window": self.kospi_volatility_window,
            "breadth_ma_window": self.breadth_ma_window,
            "annualization_days": self.annualization_days,
            "confirmation_days": self.confirmation_days,
        }
        invalid_windows = [name for name, value in windows.items() if value < 1]
        if invalid_windows:
            raise ValueError(f"Window values must be positive: {', '.join(invalid_windows)}")

        if self.risk_off_breadth_threshold >= self.risk_on_breadth_threshold:
            raise ValueError(
                "risk_off_breadth_threshold must be lower than risk_on_breadth_threshold"
            )

        for name, value in {
            "risk_on_equity_weight": self.risk_on_equity_weight,
            "neutral_equity_weight": self.neutral_equity_weight,
            "risk_off_equity_weight": self.risk_off_equity_weight,
        }.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")


DEFAULT_REGIME_ALLOCATIONS: dict[RegimeName, tuple[float, float]] = {
    "Risk-On": (1.0, 0.0),
    "Neutral": (0.8, 0.2),
    "Risk-Off": (0.5, 0.5),
}


__all__ = ["DEFAULT_REGIME_ALLOCATIONS", "RegimeConfig", "RegimeName"]
