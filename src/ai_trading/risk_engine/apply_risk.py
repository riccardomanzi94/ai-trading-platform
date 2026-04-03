"""Apply risk management to trading signals.

Takes raw signals and applies risk policy to produce risk-adjusted orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import text

from ai_trading.shared.config import config, get_db_engine
from ai_trading.signals import Signal, SignalType
from ai_trading.risk_engine.policy import RiskContext, RiskPolicy, default_policy

logger = logging.getLogger(__name__)


@dataclass
class RiskOrder:
    """Risk-adjusted order ready for execution."""

    time: datetime
    ticker: str
    signal_type: str
    original_quantity: int
    adjusted_quantity: int
    position_size_pct: float
    risk_score: float
    approved: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "time": self.time,
            "ticker": self.ticker,
            "signal_type": self.signal_type,
            "original_quantity": self.original_quantity,
            "adjusted_quantity": self.adjusted_quantity,
            "position_size_pct": self.position_size_pct,
            "risk_score": self.risk_score,
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
        }


def get_portfolio_state() -> dict:
    """Get current portfolio state from database.

    Returns:
        Dict with portfolio info:
        - cash: Available cash
        - positions: Dict of ticker -> {quantity, avg_cost, current_value}
        - total_value: Total portfolio value
        - total_invested: Total cost basis
    """
    engine = get_db_engine()

    with engine.connect() as conn:
        # Get portfolio positions
        result = conn.execute(
            text("""
                SELECT ticker, quantity, avg_cost, current_value
                FROM portfolio
                WHERE quantity > 0
            """)
        )
        positions = {}
        for row in result:
            positions[row[0]] = {
                "quantity": row[1],
                "avg_cost": row[2],
                "current_value": row[3],
            }

        # Get cash balance (stored as special ticker '_CASH')
        result = conn.execute(
            text("SELECT current_value FROM portfolio WHERE ticker = '_CASH'")
        )
        row = result.fetchone()
        cash = row[0] if row else config.trading.initial_capital

    total_value = cash + sum(p["current_value"] or 0 for p in positions.values())
    total_invested = sum(
        (p["avg_cost"] or 0) * p["quantity"] for p in positions.values()
    )

    return {
        "cash": cash,
        "positions": positions,
        "total_value": total_value,
        "total_invested": total_invested,
    }


def get_current_volatility(ticker: str) -> float:
    """Get the most recent volatility for a ticker.

    Args:
        ticker: Stock/ETF ticker symbol

    Returns:
        Annualized volatility (default: 0.20 if not available)
    """
    engine = get_db_engine()

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT volatility_20
                FROM features
                WHERE ticker = :ticker
                ORDER BY time DESC
                LIMIT 1
            """),
            {"ticker": ticker},
        )
        row = result.fetchone()
        return row[0] if row and row[0] else 0.20


def get_current_atr(ticker: str) -> float:
    """Get the most recent ATR for a ticker.

    Args:
        ticker: Stock/ETF ticker symbol

    Returns:
        ATR value (default: 0 if not available)
    """
    engine = get_db_engine()

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT atr_14
                FROM features
                WHERE ticker = :ticker
                ORDER BY time DESC
                LIMIT 1
            """),
            {"ticker": ticker},
        )
        row = result.fetchone()
        return row[0] if row and row[0] else 0.0


def calculate_quantity(
    position_size_pct: float,
    portfolio_value: float,
    price: float,
) -> int:
    """Calculate number of shares to trade.

    Args:
        position_size_pct: Target position as percentage of portfolio
        portfolio_value: Total portfolio value
        price: Current share price

    Returns:
        Number of shares (integer)
    """
    target_value = portfolio_value * position_size_pct
    quantity = int(target_value / price)
    return max(0, quantity)


def apply_risk_to_signal(
    signal: Signal,
    policy: Optional[RiskPolicy] = None,
    save_to_db: bool = True,
) -> RiskOrder:
    """Apply risk policy to a trading signal.

    Args:
        signal: Raw trading signal
        policy: Risk policy to apply (default: default_policy)
        save_to_db: Whether to save the risk order to database

    Returns:
        Risk-adjusted order
    """
    if policy is None:
        policy = default_policy

    # Get current state
    portfolio = get_portfolio_state()
    volatility = get_current_volatility(signal.ticker)
    atr = get_current_atr(signal.ticker)

    # Current position in this ticker
    position_info = portfolio["positions"].get(signal.ticker, {})
    current_position_value = position_info.get("current_value", 0) or 0

    # Build risk context
    context = RiskContext(
        ticker=signal.ticker,
        signal_type=signal.signal_type.value,
        signal_strength=signal.strength,
        current_price=signal.price_at_signal,
        volatility=volatility,
        atr=atr,
        portfolio_value=portfolio["total_value"],
        current_position_value=current_position_value,
        total_invested=portfolio["total_invested"],
    )

    # Evaluate risk policy
    approved, position_size_pct, risk_score, rejection_reason = policy.evaluate(context)

    # Calculate quantities
    original_qty = calculate_quantity(
        config.trading.max_position_size,
        portfolio["total_value"],
        signal.price_at_signal,
    )
    adjusted_qty = (
        calculate_quantity(
            position_size_pct,
            portfolio["total_value"],
            signal.price_at_signal,
        )
        if approved
        else 0
    )

    # For SELL signals, limit to current position
    if signal.signal_type == SignalType.SELL and approved:
        current_qty = position_info.get("quantity", 0)
        adjusted_qty = min(adjusted_qty, current_qty)
        if adjusted_qty == 0:
            approved = False
            rejection_reason = "No position to sell"

    # Create risk order
    order = RiskOrder(
        time=signal.time,
        ticker=signal.ticker,
        signal_type=signal.signal_type.value,
        original_quantity=original_qty,
        adjusted_quantity=adjusted_qty,
        position_size_pct=position_size_pct,
        risk_score=risk_score,
        approved=approved,
        rejection_reason=rejection_reason,
    )

    if save_to_db:
        _save_risk_order_to_db(order)

    logger.info(
        f"Risk order for {signal.ticker}: "
        f"{'APPROVED' if approved else 'REJECTED'} - "
        f"{adjusted_qty} shares ({position_size_pct:.1%} of portfolio), "
        f"risk score: {risk_score:.2f}"
    )

    return order


def _save_risk_order_to_db(order: RiskOrder) -> None:
    """Save a risk order to the database.

    Args:
        order: Risk order to save
    """
    engine = get_db_engine()

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO risk_orders 
                (time, ticker, signal_type, original_quantity, adjusted_quantity,
                 position_size_pct, risk_score, approved, rejection_reason)
                VALUES 
                (:time, :ticker, :signal_type, :original_quantity, :adjusted_quantity,
                 :position_size_pct, :risk_score, :approved, :rejection_reason)
            """),
            order.to_dict(),
        )


def apply_risk_to_all_signals(
    signals: dict[str, list[Signal]],
    policy: Optional[RiskPolicy] = None,
    save_to_db: bool = True,
) -> list[RiskOrder]:
    """Apply risk policy to all signals.

    Args:
        signals: Dict mapping ticker to list of signals
        policy: Risk policy to apply
        save_to_db: Whether to save orders to database

    Returns:
        List of all risk orders (approved and rejected)
    """
    orders = []

    for ticker, ticker_signals in signals.items():
        for signal in ticker_signals:
            try:
                order = apply_risk_to_signal(signal, policy, save_to_db)
                orders.append(order)
            except Exception as e:
                logger.error(f"Failed to process signal for {ticker}: {e}")

    # Summary
    approved_count = sum(1 for o in orders if o.approved)
    rejected_count = len(orders) - approved_count
    logger.info(f"Risk processing complete: {approved_count} approved, {rejected_count} rejected")

    return orders


def get_pending_orders() -> list[RiskOrder]:
    """Get approved risk orders that haven't been executed yet.

    Returns:
        List of pending risk orders
    """
    engine = get_db_engine()

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT ro.time, ro.ticker, ro.signal_type, ro.original_quantity,
                       ro.adjusted_quantity, ro.position_size_pct, ro.risk_score,
                       ro.approved, ro.rejection_reason
                FROM risk_orders ro
                LEFT JOIN paper_executions pe ON ro.time = pe.time AND ro.ticker = pe.ticker
                WHERE ro.approved = TRUE AND pe.id IS NULL
                ORDER BY ro.time DESC
            """)
        )

        orders = []
        for row in result:
            orders.append(
                RiskOrder(
                    time=row[0],
                    ticker=row[1],
                    signal_type=row[2],
                    original_quantity=row[3],
                    adjusted_quantity=row[4],
                    position_size_pct=row[5],
                    risk_score=row[6],
                    approved=row[7],
                    rejection_reason=row[8],
                )
            )

    return orders


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: create a mock signal and apply risk
    signal = Signal(
        time=datetime.now(),
        ticker="AAPL",
        signal_type=SignalType.BUY,
        strength=0.7,
        price_at_signal=175.0,
    )

    order = apply_risk_to_signal(signal, save_to_db=False)
    print(f"Order: {order}")
