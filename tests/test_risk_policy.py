"""Tests for risk management policy."""

from __future__ import annotations

import pytest
from datetime import datetime

from ai_trading.risk_engine.policy import (
    RiskContext,
    RiskPolicy,
    PositionSizeRule,
    MaxDrawdownRule,
    VolatilityAdjustmentRule,
)


class TestPositionSizeRule:
    """Tests for PositionSizeRule."""

    def test_buy_approved_when_no_existing_position(self):
        """Should approve BUY when no existing position."""
        rule = PositionSizeRule(max_position_pct=0.1)
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

        approved, size, reason = rule.evaluate(context)

        assert approved is True
        assert size == 0.1
        assert reason is None

    def test_buy_rejected_when_position_limit_reached(self):
        """Should reject BUY when position limit already reached."""
        rule = PositionSizeRule(max_position_pct=0.1)
        context = RiskContext(
            ticker="AAPL",
            signal_type="BUY",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.25,
            atr=3.5,
            portfolio_value=100000.0,
            current_position_value=10000.0,  # Already at 10% limit
            total_invested=50000.0,
        )

        approved, size, reason = rule.evaluate(context)

        assert approved is False
        assert size == 0.0
        assert "Position limit reached" in reason

    def test_buy_partial_when_near_limit(self):
        """Should return remaining room when near position limit."""
        rule = PositionSizeRule(max_position_pct=0.1)
        context = RiskContext(
            ticker="AAPL",
            signal_type="BUY",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.25,
            atr=3.5,
            portfolio_value=100000.0,
            current_position_value=7000.0,  # 7% used, 3% remaining
            total_invested=50000.0,
        )

        approved, size, reason = rule.evaluate(context)

        assert approved is True
        assert size == pytest.approx(0.03, rel=0.01)  # 3% remaining
        assert reason is None

    def test_sell_always_approved(self):
        """Should approve SELL signals (reduces risk)."""
        rule = PositionSizeRule(max_position_pct=0.1)
        context = RiskContext(
            ticker="AAPL",
            signal_type="SELL",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.25,
            atr=3.5,
            portfolio_value=100000.0,
            current_position_value=10000.0,
            total_invested=50000.0,
        )

        approved, size, reason = rule.evaluate(context)

        assert approved is True
        assert reason is None


class TestMaxDrawdownRule:
    """Tests for MaxDrawdownRule."""

    def test_approved_when_no_drawdown(self):
        """Should approve when portfolio is not in drawdown."""
        rule = MaxDrawdownRule(max_drawdown_pct=0.15)
        context = RiskContext(
            ticker="AAPL",
            signal_type="BUY",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.25,
            atr=3.5,
            portfolio_value=110000.0,  # 10% profit
            current_position_value=0.0,
            total_invested=100000.0,
        )

        approved, size, reason = rule.evaluate(context)

        assert approved is True
        assert reason is None

    def test_rejected_when_drawdown_exceeded(self):
        """Should reject BUY when drawdown exceeds limit."""
        rule = MaxDrawdownRule(max_drawdown_pct=0.15)
        context = RiskContext(
            ticker="AAPL",
            signal_type="BUY",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.25,
            atr=3.5,
            portfolio_value=80000.0,  # 20% loss
            current_position_value=0.0,
            total_invested=100000.0,
        )

        approved, size, reason = rule.evaluate(context)

        assert approved is False
        assert "drawdown" in reason.lower()

    def test_sell_approved_in_drawdown(self):
        """Should approve SELL even in drawdown (risk reduction)."""
        rule = MaxDrawdownRule(max_drawdown_pct=0.15)
        context = RiskContext(
            ticker="AAPL",
            signal_type="SELL",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.25,
            atr=3.5,
            portfolio_value=80000.0,  # 20% loss
            current_position_value=10000.0,
            total_invested=100000.0,
        )

        approved, size, reason = rule.evaluate(context)

        assert approved is True


class TestVolatilityAdjustmentRule:
    """Tests for VolatilityAdjustmentRule."""

    def test_reduces_size_for_high_volatility(self):
        """Should reduce position size for high volatility."""
        rule = VolatilityAdjustmentRule(
            target_volatility=0.20,
            max_position_pct=0.10,
        )
        context = RiskContext(
            ticker="AAPL",
            signal_type="BUY",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.40,  # 2x target volatility
            atr=3.5,
            portfolio_value=100000.0,
            current_position_value=0.0,
            total_invested=50000.0,
        )

        approved, size, reason = rule.evaluate(context)

        assert approved is True
        assert size < 0.10  # Should be reduced
        assert size == pytest.approx(0.05, rel=0.1)  # ~half of max

    def test_increases_size_for_low_volatility(self):
        """Should allow full size for low volatility."""
        rule = VolatilityAdjustmentRule(
            target_volatility=0.20,
            max_position_pct=0.10,
        )
        context = RiskContext(
            ticker="AAPL",
            signal_type="BUY",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.10,  # Half of target
            atr=3.5,
            portfolio_value=100000.0,
            current_position_value=0.0,
            total_invested=50000.0,
        )

        approved, size, reason = rule.evaluate(context)

        assert approved is True
        assert size == 0.10  # Should be at max (capped)


class TestRiskPolicy:
    """Tests for RiskPolicy aggregation."""

    def test_combines_multiple_rules(self):
        """Should use minimum position size from all rules."""
        policy = RiskPolicy(
            rules=[
                PositionSizeRule(max_position_pct=0.10),
                VolatilityAdjustmentRule(
                    target_volatility=0.20,
                    max_position_pct=0.10,
                ),
            ]
        )
        context = RiskContext(
            ticker="AAPL",
            signal_type="BUY",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.40,  # High volatility
            atr=3.5,
            portfolio_value=100000.0,
            current_position_value=0.0,
            total_invested=50000.0,
        )

        approved, size, risk_score, reason = policy.evaluate(context)

        assert approved is True
        assert size < 0.10  # Reduced due to volatility

    def test_rejects_if_any_rule_rejects(self):
        """Should reject if any rule rejects."""
        policy = RiskPolicy(
            rules=[
                PositionSizeRule(max_position_pct=0.10),
                MaxDrawdownRule(max_drawdown_pct=0.10),
            ]
        )
        context = RiskContext(
            ticker="AAPL",
            signal_type="BUY",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.25,
            atr=3.5,
            portfolio_value=85000.0,  # 15% drawdown > 10% limit
            current_position_value=0.0,
            total_invested=100000.0,
        )

        approved, size, risk_score, reason = policy.evaluate(context)

        assert approved is False
        assert "drawdown" in reason.lower()

    def test_calculates_risk_score(self):
        """Should calculate a risk score between 0 and 1."""
        policy = RiskPolicy()
        context = RiskContext(
            ticker="AAPL",
            signal_type="BUY",
            signal_strength=0.5,
            current_price=150.0,
            volatility=0.30,
            atr=3.5,
            portfolio_value=100000.0,
            current_position_value=0.0,
            total_invested=50000.0,
        )

        approved, size, risk_score, reason = policy.evaluate(context)

        assert 0.0 <= risk_score <= 1.0


class TestRiskContext:
    """Tests for RiskContext dataclass."""

    def test_creates_context(self):
        """Should create risk context with all fields."""
        context = RiskContext(
            ticker="AAPL",
            signal_type="BUY",
            signal_strength=0.7,
            current_price=150.0,
            volatility=0.25,
            atr=3.5,
            portfolio_value=100000.0,
            current_position_value=5000.0,
            total_invested=50000.0,
        )

        assert context.ticker == "AAPL"
        assert context.signal_type == "BUY"
        assert context.portfolio_value == 100000.0
