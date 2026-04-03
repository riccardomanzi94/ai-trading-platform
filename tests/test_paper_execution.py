"""Tests for paper trading execution."""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from ai_trading.execution.paper_execution import (
    Execution,
    simulate_slippage,
    calculate_commission,
    execute_order,
)
from ai_trading.risk_engine.apply_risk import RiskOrder


class TestSlippage:
    """Tests for slippage simulation."""

    def test_buy_slippage_increases_price(self):
        """BUY orders should have positive slippage (pay more)."""
        price = 100.0
        slippage_rate = 0.001  # 0.1%

        result = simulate_slippage(price, "BUY", slippage_rate)

        assert result > price
        assert result == pytest.approx(100.10, rel=0.01)

    def test_sell_slippage_decreases_price(self):
        """SELL orders should have negative slippage (receive less)."""
        price = 100.0
        slippage_rate = 0.001  # 0.1%

        result = simulate_slippage(price, "SELL", slippage_rate)

        assert result < price
        assert result == pytest.approx(99.90, rel=0.01)

    def test_zero_slippage(self):
        """Zero slippage should return original price."""
        price = 100.0

        buy_result = simulate_slippage(price, "BUY", 0)
        sell_result = simulate_slippage(price, "SELL", 0)

        assert buy_result == price
        assert sell_result == price


class TestCommission:
    """Tests for commission calculation."""

    def test_commission_calculation(self):
        """Should calculate commission as percentage of trade value."""
        quantity = 100
        price = 50.0
        commission_rate = 0.001  # 0.1%

        result = calculate_commission(quantity, price, commission_rate)

        assert result == pytest.approx(5.0)  # 100 * 50 * 0.001 = 5

    def test_zero_quantity_commission(self):
        """Zero quantity should result in zero commission."""
        result = calculate_commission(0, 100.0, 0.001)

        assert result == 0

    def test_commission_is_absolute(self):
        """Commission should be absolute (always positive)."""
        result = calculate_commission(-100, 50.0, 0.001)

        assert result > 0


class TestExecution:
    """Tests for Execution dataclass."""

    def test_execution_to_dict(self):
        """Should convert execution to dictionary."""
        execution = Execution(
            time=datetime(2024, 1, 15, 10, 30),
            ticker="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            commission=15.0,
            slippage=7.5,
            total_value=15015.0,
        )

        result = execution.to_dict()

        assert result["ticker"] == "AAPL"
        assert result["side"] == "BUY"
        assert result["quantity"] == 100
        assert result["price"] == 150.0
        assert result["commission"] == 15.0


class TestExecuteOrder:
    """Tests for order execution."""

    @patch("ai_trading.execution.paper_execution.get_current_price")
    @patch("ai_trading.execution.paper_execution._save_execution_to_db")
    @patch("ai_trading.execution.paper_execution.update_portfolio")
    def test_execute_approved_buy_order(
        self, mock_update, mock_save, mock_price
    ):
        """Should execute approved BUY order."""
        mock_price.return_value = 150.0

        order = RiskOrder(
            time=datetime.now(),
            ticker="AAPL",
            signal_type="BUY",
            original_quantity=100,
            adjusted_quantity=80,
            position_size_pct=0.08,
            risk_score=0.3,
            approved=True,
        )

        execution = execute_order(order, save_to_db=False)

        assert execution is not None
        assert execution.ticker == "AAPL"
        assert execution.side == "BUY"
        assert execution.quantity == 80
        assert execution.price > 150.0  # Slippage added

    @patch("ai_trading.execution.paper_execution.get_current_price")
    def test_reject_unapproved_order(self, mock_price):
        """Should reject unapproved orders."""
        mock_price.return_value = 150.0

        order = RiskOrder(
            time=datetime.now(),
            ticker="AAPL",
            signal_type="BUY",
            original_quantity=100,
            adjusted_quantity=0,
            position_size_pct=0.0,
            risk_score=0.8,
            approved=False,
            rejection_reason="Drawdown exceeded",
        )

        execution = execute_order(order, save_to_db=False)

        assert execution is None

    @patch("ai_trading.execution.paper_execution.get_current_price")
    def test_reject_zero_quantity_order(self, mock_price):
        """Should reject orders with zero quantity."""
        mock_price.return_value = 150.0

        order = RiskOrder(
            time=datetime.now(),
            ticker="AAPL",
            signal_type="BUY",
            original_quantity=100,
            adjusted_quantity=0,
            position_size_pct=0.0,
            risk_score=0.3,
            approved=True,
        )

        execution = execute_order(order, save_to_db=False)

        assert execution is None

    @patch("ai_trading.execution.paper_execution.get_current_price")
    def test_reject_when_no_price_available(self, mock_price):
        """Should reject when price is unavailable."""
        mock_price.return_value = None

        order = RiskOrder(
            time=datetime.now(),
            ticker="UNKNOWN",
            signal_type="BUY",
            original_quantity=100,
            adjusted_quantity=80,
            position_size_pct=0.08,
            risk_score=0.3,
            approved=True,
        )

        execution = execute_order(order, save_to_db=False)

        assert execution is None

    @patch("ai_trading.execution.paper_execution.get_current_price")
    @patch("ai_trading.execution.paper_execution._save_execution_to_db")
    @patch("ai_trading.execution.paper_execution.update_portfolio")
    def test_sell_execution_includes_commission(
        self, mock_update, mock_save, mock_price
    ):
        """SELL execution should subtract commission from proceeds."""
        mock_price.return_value = 150.0

        order = RiskOrder(
            time=datetime.now(),
            ticker="AAPL",
            signal_type="SELL",
            original_quantity=100,
            adjusted_quantity=50,
            position_size_pct=0.05,
            risk_score=0.3,
            approved=True,
        )

        execution = execute_order(order, save_to_db=False)

        assert execution is not None
        assert execution.side == "SELL"
        # Total value should be less than gross proceeds (commission deducted)
        gross = execution.quantity * execution.price
        assert execution.total_value < gross


class TestRiskOrder:
    """Tests for RiskOrder dataclass."""

    def test_risk_order_to_dict(self):
        """Should convert risk order to dictionary."""
        order = RiskOrder(
            time=datetime(2024, 1, 15),
            ticker="AAPL",
            signal_type="BUY",
            original_quantity=100,
            adjusted_quantity=80,
            position_size_pct=0.08,
            risk_score=0.35,
            approved=True,
        )

        result = order.to_dict()

        assert result["ticker"] == "AAPL"
        assert result["approved"] is True
        assert result["adjusted_quantity"] == 80

    def test_risk_order_with_rejection(self):
        """Should include rejection reason."""
        order = RiskOrder(
            time=datetime(2024, 1, 15),
            ticker="AAPL",
            signal_type="BUY",
            original_quantity=100,
            adjusted_quantity=0,
            position_size_pct=0.0,
            risk_score=0.9,
            approved=False,
            rejection_reason="Position limit exceeded",
        )

        result = order.to_dict()

        assert result["approved"] is False
        assert "Position limit" in result["rejection_reason"]
