"""EMA Crossover Backtesting Strategy.

Implements a simple EMA crossover strategy for backtesting:
- BUY when EMA12 crosses above EMA26
- SELL when EMA12 crosses below EMA26

Tracks performance metrics including:
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from ai_trading.shared.config import config, get_db_engine
from ai_trading.feature_store.build_features import compute_ema

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade (entry + exit)."""

    entry_date: datetime
    exit_date: datetime
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Backtest Results for {self.ticker}
{'=' * 40}
Period: {self.start_date} to {self.end_date}
Initial Capital: ${self.initial_capital:,.2f}
Final Value: ${self.final_value:,.2f}
Total Return: {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}
Win Rate: {self.win_rate:.2%}
Total Trades: {self.total_trades}
Winning/Losing: {self.winning_trades}/{self.losing_trades}
Avg Trade Return: {self.avg_trade_return:.2%}
"""


def load_price_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load price data for backtesting.

    Args:
        ticker: Stock/ETF ticker symbol
        start_date: Start date (default: from config)
        end_date: End date (default: today)

    Returns:
        DataFrame with price data
    """
    if start_date is None:
        start_date = config.backtest.start_date

    engine = get_db_engine()

    query = text("""
        SELECT time, open, high, low, close, adj_close, volume
        FROM prices
        WHERE ticker = :ticker
        AND time >= :start_date
        AND (:end_date IS NULL OR time <= :end_date)
        ORDER BY time ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df


def run_ema_crossover_backtest(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: Optional[float] = None,
    ema_short: Optional[int] = None,
    ema_long: Optional[int] = None,
) -> BacktestResult:
    """Run EMA crossover backtest.

    Args:
        ticker: Stock/ETF ticker symbol
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        ema_short: Short EMA period (default: from config)
        ema_long: Long EMA period (default: from config)

    Returns:
        BacktestResult with performance metrics
    """
    if initial_capital is None:
        initial_capital = config.trading.initial_capital
    if ema_short is None:
        ema_short = config.features.ema_short
    if ema_long is None:
        ema_long = config.features.ema_long

    logger.info(f"Running EMA crossover backtest for {ticker}")

    # Load data
    df = load_price_data(ticker, start_date, end_date)
    if df.empty or len(df) < ema_long + 1:
        raise ValueError(f"Insufficient data for {ticker} backtest")

    # Calculate EMAs
    df["ema_short"] = compute_ema(df["adj_close"], ema_short)
    df["ema_long"] = compute_ema(df["adj_close"], ema_long)

    # Generate signals
    df["signal"] = 0
    df.loc[df["ema_short"] > df["ema_long"], "signal"] = 1  # Long
    df.loc[df["ema_short"] < df["ema_long"], "signal"] = -1  # Short/Out

    # Drop NaN rows
    df = df.dropna()

    # Run backtest simulation
    trades, equity_curve = _simulate_trades(df, initial_capital)

    # Calculate metrics
    result = calculate_metrics(
        ticker=ticker,
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=initial_capital,
        start_date=df.index[0].strftime("%Y-%m-%d"),
        end_date=df.index[-1].strftime("%Y-%m-%d"),
    )

    logger.info(f"Backtest complete: {result.total_return:.2%} return")
    return result


def _simulate_trades(
    df: pd.DataFrame,
    initial_capital: float,
) -> tuple[list[Trade], pd.DataFrame]:
    """Simulate trades based on signals.

    Args:
        df: DataFrame with signals
        initial_capital: Starting capital

    Returns:
        Tuple of (trades list, equity curve DataFrame)
    """
    cash = initial_capital
    shares = 0
    position = None  # "LONG" or None
    entry_price = 0.0
    entry_date = None

    trades = []
    equity_values = []
    equity_dates = []

    commission_rate = config.trading.commission_rate
    slippage_rate = config.trading.slippage_rate

    prev_signal = 0

    for date, row in df.iterrows():
        price = row["adj_close"]
        signal = row["signal"]

        # Track equity
        equity = cash + shares * price
        equity_values.append(equity)
        equity_dates.append(date)

        # Entry: signal turns positive and we're not in position
        if signal == 1 and prev_signal != 1 and position is None:
            # Buy
            buy_price = price * (1 + slippage_rate)
            shares_to_buy = int(cash * 0.95 / buy_price)  # Use 95% of cash
            if shares_to_buy > 0:
                cost = shares_to_buy * buy_price
                commission = cost * commission_rate
                cash -= cost + commission
                shares = shares_to_buy
                position = "LONG"
                entry_price = buy_price
                entry_date = date

        # Exit: signal turns negative and we're in position
        elif signal == -1 and prev_signal != -1 and position == "LONG":
            # Sell
            sell_price = price * (1 - slippage_rate)
            proceeds = shares * sell_price
            commission = proceeds * commission_rate
            cash += proceeds - commission

            # Record trade
            pnl = (sell_price - entry_price) * shares - commission * 2
            return_pct = (sell_price - entry_price) / entry_price

            trades.append(
                Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    side="LONG",
                    entry_price=entry_price,
                    exit_price=sell_price,
                    shares=shares,
                    pnl=pnl,
                    return_pct=return_pct,
                )
            )

            shares = 0
            position = None

        prev_signal = signal

    # Close any open position at end
    if position == "LONG" and shares > 0:
        final_price = df["adj_close"].iloc[-1] * (1 - slippage_rate)
        proceeds = shares * final_price
        commission = proceeds * commission_rate
        cash += proceeds - commission

        pnl = (final_price - entry_price) * shares - commission * 2
        return_pct = (final_price - entry_price) / entry_price

        trades.append(
            Trade(
                entry_date=entry_date,
                exit_date=df.index[-1],
                side="LONG",
                entry_price=entry_price,
                exit_price=final_price,
                shares=shares,
                pnl=pnl,
                return_pct=return_pct,
            )
        )

    equity_curve = pd.DataFrame({"equity": equity_values}, index=equity_dates)
    return trades, equity_curve


def calculate_metrics(
    ticker: str,
    trades: list[Trade],
    equity_curve: pd.DataFrame,
    initial_capital: float,
    start_date: str,
    end_date: str,
) -> BacktestResult:
    """Calculate backtest performance metrics.

    Args:
        ticker: Ticker symbol
        trades: List of completed trades
        equity_curve: Equity curve DataFrame
        initial_capital: Starting capital
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        BacktestResult with all metrics
    """
    final_value = equity_curve["equity"].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # Calculate annualized return
    days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Calculate Sharpe ratio
    daily_returns = equity_curve["equity"].pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Calculate max drawdown
    rolling_max = equity_curve["equity"].expanding().max()
    drawdown = (equity_curve["equity"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Trade statistics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl > 0)
    losing_trades = sum(1 for t in trades if t.pnl <= 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    avg_trade_return = (
        sum(t.return_pct for t in trades) / total_trades if total_trades > 0 else 0
    )

    return BacktestResult(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_value=final_value,
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        avg_trade_return=avg_trade_return,
        trades=trades,
        equity_curve=equity_curve,
    )


def compare_to_benchmark(
    result: BacktestResult,
    benchmark_ticker: Optional[str] = None,
) -> dict:
    """Compare backtest results to a benchmark.

    Args:
        result: Backtest result
        benchmark_ticker: Benchmark ticker (default: from config)

    Returns:
        Dict with comparison metrics
    """
    if benchmark_ticker is None:
        benchmark_ticker = config.backtest.benchmark_ticker

    # Load benchmark data
    df = load_price_data(benchmark_ticker, result.start_date, result.end_date)
    if df.empty:
        return {}

    benchmark_return = (df["adj_close"].iloc[-1] - df["adj_close"].iloc[0]) / df[
        "adj_close"
    ].iloc[0]

    alpha = result.total_return - benchmark_return

    return {
        "benchmark_ticker": benchmark_ticker,
        "benchmark_return": benchmark_return,
        "strategy_return": result.total_return,
        "alpha": alpha,
        "outperformed": result.total_return > benchmark_return,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run backtest on SPY
    result = run_ema_crossover_backtest("SPY")
    print(result.summary())

    # Compare to benchmark
    comparison = compare_to_benchmark(result)
    if comparison:
        print(f"\nBenchmark comparison ({comparison['benchmark_ticker']}):")
        print(f"Benchmark Return: {comparison['benchmark_return']:.2%}")
        print(f"Alpha: {comparison['alpha']:.2%}")
