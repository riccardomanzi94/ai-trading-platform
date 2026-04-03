"""Risk management policies and rules.

Defines risk rules for:
- Position sizing based on volatility
- Maximum position size limits
- Portfolio-level risk constraints
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ai_trading.shared.config import config

logger = logging.getLogger(__name__)


@dataclass
class RiskContext:
    """Context information for risk calculations."""

    ticker: str
    signal_type: str  # "BUY" or "SELL"
    signal_strength: float
    current_price: float
    volatility: float  # Annualized volatility
    atr: float  # Average True Range
    portfolio_value: float
    current_position_value: float  # Current position in this ticker
    total_invested: float  # Total value invested across all positions


class RiskRule(ABC):
    """Abstract base class for risk rules."""

    @abstractmethod
    def evaluate(self, context: RiskContext) -> tuple[bool, float, Optional[str]]:
        """Evaluate the risk rule.

        Args:
            context: Risk calculation context

        Returns:
            Tuple of (approved, position_size_pct, rejection_reason)
            - approved: Whether the trade is approved
            - position_size_pct: Recommended position size as % of portfolio
            - rejection_reason: Reason for rejection (if not approved)
        """
        pass


@dataclass
class PositionSizeRule(RiskRule):
    """Rule for maximum position size per ticker.

    Limits exposure to any single ticker to a percentage of portfolio.
    """

    max_position_pct: float = field(
        default_factory=lambda: config.trading.max_position_size
    )

    def evaluate(self, context: RiskContext) -> tuple[bool, float, Optional[str]]:
        """Enforce maximum position size."""
        max_position_value = context.portfolio_value * self.max_position_pct
        current_exposure_pct = context.current_position_value / context.portfolio_value

        if context.signal_type == "BUY":
            # Check if we can add to position
            if current_exposure_pct >= self.max_position_pct:
                return (
                    False,
                    0.0,
                    f"Position limit reached: {current_exposure_pct:.1%} >= {self.max_position_pct:.1%}",
                )

            # Calculate remaining room for this position
            remaining_pct = self.max_position_pct - current_exposure_pct
            return True, min(remaining_pct, self.max_position_pct), None

        elif context.signal_type == "SELL":
            # Sells are generally allowed (reduces risk)
            return True, current_exposure_pct, None

        return True, 0.0, None


@dataclass
class MaxDrawdownRule(RiskRule):
    """Rule to prevent trading during high portfolio drawdown.

    Reduces position sizes when portfolio is in drawdown.
    """

    max_drawdown_pct: float = 0.15  # 15% max drawdown
    reduction_factor: float = 0.5  # Reduce size by 50% in drawdown

    def evaluate(self, context: RiskContext) -> tuple[bool, float, Optional[str]]:
        """Apply drawdown-based position reduction."""
        # Calculate implied drawdown from invested vs portfolio value
        if context.total_invested > 0:
            pnl_pct = (context.portfolio_value - context.total_invested) / context.total_invested
            if pnl_pct < -self.max_drawdown_pct:
                if context.signal_type == "BUY":
                    return (
                        False,
                        0.0,
                        f"Portfolio drawdown {pnl_pct:.1%} exceeds limit {-self.max_drawdown_pct:.1%}",
                    )

        return True, config.trading.max_position_size, None


@dataclass
class VolatilityAdjustmentRule(RiskRule):
    """Rule to adjust position size based on volatility.

    Higher volatility = smaller position size.
    Uses inverse volatility scaling.
    """

    target_volatility: float = 0.20  # 20% annualized target
    min_position_pct: float = 0.02  # Minimum 2% position
    max_position_pct: float = field(
        default_factory=lambda: config.trading.max_position_size
    )

    def evaluate(self, context: RiskContext) -> tuple[bool, float, Optional[str]]:
        """Calculate volatility-adjusted position size."""
        if context.volatility <= 0:
            return True, self.max_position_pct, None

        # Inverse volatility scaling
        vol_ratio = self.target_volatility / context.volatility
        adjusted_size = self.max_position_pct * vol_ratio

        # Clamp to min/max bounds
        adjusted_size = max(self.min_position_pct, min(adjusted_size, self.max_position_pct))

        logger.debug(
            f"{context.ticker} volatility={context.volatility:.2%}, "
            f"adjusted_size={adjusted_size:.2%}"
        )

        return True, adjusted_size, None


@dataclass
class RiskPolicy:
    """Aggregates multiple risk rules into a single policy.

    Rules are evaluated in order. If any rule rejects, the trade is rejected.
    Position size is the minimum of all rule recommendations.
    """

    rules: list[RiskRule] = field(default_factory=list)

    def __post_init__(self):
        if not self.rules:
            # Default rules
            self.rules = [
                PositionSizeRule(),
                MaxDrawdownRule(),
                VolatilityAdjustmentRule(),
            ]

    def evaluate(
        self, context: RiskContext
    ) -> tuple[bool, float, float, Optional[str]]:
        """Evaluate all rules and aggregate results.

        Args:
            context: Risk calculation context

        Returns:
            Tuple of (approved, position_size_pct, risk_score, rejection_reason)
            - approved: Whether the trade is approved by all rules
            - position_size_pct: Final position size (minimum of all rules)
            - risk_score: Calculated risk score (0-1)
            - rejection_reason: First rejection reason (if any)
        """
        approved = True
        min_position_size = config.trading.max_position_size
        rejection_reason = None

        for rule in self.rules:
            rule_approved, rule_size, rule_reason = rule.evaluate(context)

            if not rule_approved:
                approved = False
                rejection_reason = rule_reason
                break

            min_position_size = min(min_position_size, rule_size)

        # Calculate risk score based on volatility and position size
        risk_score = self._calculate_risk_score(context, min_position_size)

        return approved, min_position_size, risk_score, rejection_reason

    def _calculate_risk_score(
        self, context: RiskContext, position_size_pct: float
    ) -> float:
        """Calculate a risk score for the trade.

        Args:
            context: Risk context
            position_size_pct: Final position size

        Returns:
            Risk score between 0 (low risk) and 1 (high risk)
        """
        # Factors contributing to risk:
        # - High volatility increases risk
        # - Large position size increases risk
        # - Weak signal strength increases risk

        vol_risk = min(context.volatility / 0.5, 1.0)  # Normalize to ~50% vol
        size_risk = position_size_pct / config.trading.max_position_size
        signal_risk = 1.0 - context.signal_strength

        # Weighted average
        risk_score = 0.4 * vol_risk + 0.3 * size_risk + 0.3 * signal_risk
        return min(max(risk_score, 0.0), 1.0)


# Default policy instance
default_policy = RiskPolicy()


if __name__ == "__main__":
    # Example usage
    context = RiskContext(
        ticker="AAPL",
        signal_type="BUY",
        signal_strength=0.7,
        current_price=150.0,
        volatility=0.25,
        atr=3.5,
        portfolio_value=100000.0,
        current_position_value=0.0,
        total_invested=50000.0,
    )

    policy = RiskPolicy()
    approved, size, risk, reason = policy.evaluate(context)
    print(f"Approved: {approved}")
    print(f"Position Size: {size:.2%}")
    print(f"Risk Score: {risk:.2f}")
    print(f"Reason: {reason}")
