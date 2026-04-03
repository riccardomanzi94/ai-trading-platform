"""Paper trading execution simulation.

Simulates order execution with:
- Commission costs
- Slippage modeling
- Portfolio state updates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import text

from ai_trading.shared.config import config, get_db_engine
from ai_trading.risk_engine.apply_risk import RiskOrder, get_pending_orders

logger = logging.getLogger(__name__)


@dataclass
class Execution:
    """Represents an executed trade."""

    time: datetime
    ticker: str
    side: str  # "BUY" or "SELL"
    quantity: int
    price: float
    commission: float
    slippage: float
    total_value: float

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "time": self.time,
            "ticker": self.ticker,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "slippage": self.slippage,
            "total_value": self.total_value,
        }


def simulate_slippage(price: float, side: str, slippage_rate: float) -> float:
    """Simulate price slippage.

    Args:
        price: Target price
        side: "BUY" or "SELL"
        slippage_rate: Slippage rate (e.g., 0.0005 for 0.05%)

    Returns:
        Adjusted price after slippage
    """
    # BUY: price goes up (pay more)
    # SELL: price goes down (receive less)
    slippage_factor = 1 + slippage_rate if side == "BUY" else 1 - slippage_rate
    return price * slippage_factor


def calculate_commission(quantity: int, price: float, commission_rate: float) -> float:
    """Calculate commission for a trade.

    Args:
        quantity: Number of shares
        price: Price per share
        commission_rate: Commission rate (e.g., 0.001 for 0.1%)

    Returns:
        Commission amount
    """
    return abs(quantity * price * commission_rate)


def execute_order(
    order: RiskOrder,
    current_price: Optional[float] = None,
    save_to_db: bool = True,
) -> Optional[Execution]:
    """Execute a single risk-approved order.

    Args:
        order: Risk-approved order to execute
        current_price: Current market price (uses order price if not provided)
        save_to_db: Whether to save execution to database

    Returns:
        Execution record, or None if order rejected/failed
    """
    if not order.approved:
        logger.warning(f"Order for {order.ticker} not approved: {order.rejection_reason}")
        return None

    if order.adjusted_quantity <= 0:
        logger.warning(f"Order for {order.ticker} has zero quantity")
        return None

    # Use current price or fall back to signal price
    base_price = current_price if current_price else get_current_price(order.ticker)
    if base_price is None:
        logger.error(f"Could not determine price for {order.ticker}")
        return None

    # Simulate slippage
    executed_price = simulate_slippage(
        base_price,
        order.signal_type,
        config.trading.slippage_rate,
    )

    # Calculate costs
    slippage_cost = abs(executed_price - base_price) * order.adjusted_quantity
    commission = calculate_commission(
        order.adjusted_quantity,
        executed_price,
        config.trading.commission_rate,
    )

    # Total value
    gross_value = order.adjusted_quantity * executed_price
    if order.signal_type == "BUY":
        total_value = gross_value + commission
    else:
        total_value = gross_value - commission

    execution = Execution(
        time=order.time,
        ticker=order.ticker,
        side=order.signal_type,
        quantity=order.adjusted_quantity,
        price=executed_price,
        commission=commission,
        slippage=slippage_cost,
        total_value=total_value,
    )

    if save_to_db:
        _save_execution_to_db(execution)
        update_portfolio(execution)

    logger.info(
        f"Executed {order.signal_type} {order.adjusted_quantity} {order.ticker} "
        f"@ ${executed_price:.2f} (commission: ${commission:.2f}, slippage: ${slippage_cost:.2f})"
    )

    return execution


def _save_execution_to_db(execution: Execution) -> None:
    """Save execution to database.

    Args:
        execution: Execution to save
    """
    engine = get_db_engine()

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO paper_executions 
                (time, ticker, side, quantity, price, commission, slippage, total_value)
                VALUES 
                (:time, :ticker, :side, :quantity, :price, :commission, :slippage, :total_value)
            """),
            execution.to_dict(),
        )


def get_current_price(ticker: str) -> Optional[float]:
    """Get the most recent price for a ticker.

    Args:
        ticker: Stock/ETF ticker symbol

    Returns:
        Latest adj_close price
    """
    engine = get_db_engine()

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT adj_close
                FROM prices
                WHERE ticker = :ticker
                ORDER BY time DESC
                LIMIT 1
            """),
            {"ticker": ticker},
        )
        row = result.fetchone()
        return row[0] if row else None


def update_portfolio(execution: Execution) -> None:
    """Update portfolio state after execution.

    Args:
        execution: Executed trade
    """
    engine = get_db_engine()

    with engine.begin() as conn:
        # Get current position
        result = conn.execute(
            text("""
                SELECT quantity, avg_cost
                FROM portfolio
                WHERE ticker = :ticker
            """),
            {"ticker": execution.ticker},
        )
        row = result.fetchone()

        if execution.side == "BUY":
            if row:
                # Update existing position
                old_qty = row[0] or 0
                old_cost = row[1] or 0
                new_qty = old_qty + execution.quantity
                # Weighted average cost
                new_cost = (
                    (old_qty * old_cost + execution.quantity * execution.price) / new_qty
                    if new_qty > 0
                    else 0
                )
                conn.execute(
                    text("""
                        UPDATE portfolio
                        SET quantity = :quantity,
                            avg_cost = :avg_cost,
                            current_value = :current_value
                        WHERE ticker = :ticker
                    """),
                    {
                        "ticker": execution.ticker,
                        "quantity": new_qty,
                        "avg_cost": new_cost,
                        "current_value": new_qty * execution.price,
                    },
                )
            else:
                # Create new position
                conn.execute(
                    text("""
                        INSERT INTO portfolio (ticker, quantity, avg_cost, current_value)
                        VALUES (:ticker, :quantity, :avg_cost, :current_value)
                    """),
                    {
                        "ticker": execution.ticker,
                        "quantity": execution.quantity,
                        "avg_cost": execution.price,
                        "current_value": execution.quantity * execution.price,
                    },
                )

            # Reduce cash
            _update_cash(-execution.total_value, conn)

        elif execution.side == "SELL":
            if row and row[0] >= execution.quantity:
                new_qty = row[0] - execution.quantity
                conn.execute(
                    text("""
                        UPDATE portfolio
                        SET quantity = :quantity,
                            current_value = :current_value
                        WHERE ticker = :ticker
                    """),
                    {
                        "ticker": execution.ticker,
                        "quantity": new_qty,
                        "current_value": new_qty * execution.price,
                    },
                )
                # Increase cash
                _update_cash(execution.total_value, conn)


def _update_cash(amount: float, conn) -> None:
    """Update cash balance.

    Args:
        amount: Amount to add (positive) or subtract (negative)
        conn: Database connection
    """
    result = conn.execute(
        text("SELECT current_value FROM portfolio WHERE ticker = '_CASH'")
    )
    row = result.fetchone()

    if row:
        new_cash = row[0] + amount
        conn.execute(
            text("UPDATE portfolio SET current_value = :value WHERE ticker = '_CASH'"),
            {"value": new_cash},
        )
    else:
        # Initialize cash
        new_cash = config.trading.initial_capital + amount
        conn.execute(
            text("""
                INSERT INTO portfolio (ticker, quantity, avg_cost, current_value)
                VALUES ('_CASH', 1, 0, :value)
            """),
            {"value": new_cash},
        )


def execute_all_orders(
    orders: Optional[list[RiskOrder]] = None,
    save_to_db: bool = True,
) -> list[Execution]:
    """Execute all pending approved orders.

    Args:
        orders: List of orders to execute (default: pending orders from DB)
        save_to_db: Whether to save executions to database

    Returns:
        List of executions
    """
    if orders is None:
        orders = get_pending_orders()

    executions = []
    for order in orders:
        if order.approved:
            execution = execute_order(order, save_to_db=save_to_db)
            if execution:
                executions.append(execution)

    # Summary
    total_bought = sum(e.total_value for e in executions if e.side == "BUY")
    total_sold = sum(e.total_value for e in executions if e.side == "SELL")
    logger.info(
        f"Executed {len(executions)} orders: "
        f"${total_bought:.2f} bought, ${total_sold:.2f} sold"
    )

    return executions


def get_executions(
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Get execution history.

    Args:
        ticker: Filter by ticker (optional)
        start_date: Start date filter
        end_date: End date filter
        limit: Maximum number of records

    Returns:
        DataFrame with execution history
    """
    engine = get_db_engine()

    where_clauses = []
    params = {"limit": limit}

    if ticker:
        where_clauses.append("ticker = :ticker")
        params["ticker"] = ticker
    if start_date:
        where_clauses.append("time >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_clauses.append("time <= :end_date")
        params["end_date"] = end_date

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = text(f"""
        SELECT time, ticker, side, quantity, price, commission, slippage, total_value, created_at
        FROM paper_executions
        {where_sql}
        ORDER BY time DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    return df


def get_portfolio_summary() -> pd.DataFrame:
    """Get current portfolio summary.

    Returns:
        DataFrame with portfolio positions
    """
    engine = get_db_engine()

    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT ticker, quantity, avg_cost, current_value
                FROM portfolio
                ORDER BY ticker
            """),
            conn,
        )

    return df


def initialize_portfolio(initial_capital: Optional[float] = None) -> None:
    """Initialize portfolio with starting capital.

    Args:
        initial_capital: Starting cash (default: from config)
    """
    if initial_capital is None:
        initial_capital = config.trading.initial_capital

    engine = get_db_engine()

    with engine.begin() as conn:
        # Clear existing portfolio
        conn.execute(text("DELETE FROM portfolio"))

        # Set initial cash
        conn.execute(
            text("""
                INSERT INTO portfolio (ticker, quantity, avg_cost, current_value)
                VALUES ('_CASH', 1, 0, :value)
            """),
            {"value": initial_capital},
        )

    logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: initialize portfolio
    initialize_portfolio()
    print(get_portfolio_summary())
