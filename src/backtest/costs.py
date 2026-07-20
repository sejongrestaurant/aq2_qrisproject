"""Transaction cost helpers for backtests and rebalancing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransactionCostConfig:
    """One-way transaction cost assumptions."""

    commission_rate: float = 0.0015
    market_impact_rate: float = 0.0005

    @property
    def total_one_way_rate(self) -> float:
        """Return commission plus market impact."""
        return self.commission_rate + self.market_impact_rate

    def validate(self) -> None:
        """Validate cost rates."""
        if self.commission_rate < 0.0:
            raise ValueError("commission_rate must be non-negative")
        if self.market_impact_rate < 0.0:
            raise ValueError("market_impact_rate must be non-negative")


def calculate_transaction_cost(
    trade_value: float,
    config: TransactionCostConfig | None = None,
) -> float:
    """Return one-way transaction cost for an absolute trade value."""
    resolved = config or TransactionCostConfig()
    resolved.validate()
    return abs(float(trade_value)) * resolved.total_one_way_rate


__all__ = ["TransactionCostConfig", "calculate_transaction_cost"]
