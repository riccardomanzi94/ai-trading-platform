"""Prefect orchestration flows for AI Trading Platform.

Defines the daily end-to-end pipeline:
1. Data ingestion from Yahoo Finance
2. Feature computation
3. Signal generation
4. Risk management
5. Paper execution
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

# Import platform modules
from ai_trading.shared.config import config
from ai_trading.data_ingestion import ingest_all_tickers, incremental_ingest
from ai_trading.feature_store import build_all_features
from ai_trading.signals import generate_all_signals
from ai_trading.risk_engine import apply_risk_to_all_signals
from ai_trading.execution import execute_all_orders, initialize_portfolio
from ai_trading.backtest import run_ema_crossover_backtest

logger = logging.getLogger(__name__)


@task(
    name="ingest_market_data",
    description="Download OHLCV data from Yahoo Finance",
    retries=3,
    retry_delay_seconds=60,
)
def ingest_data_task(
    tickers: Optional[list[str]] = None,
    incremental: bool = True,
) -> dict[str, int]:
    """Ingest market data for all tickers.

    Args:
        tickers: List of ticker symbols (default: from config)
        incremental: If True, only fetch new data since last ingestion

    Returns:
        Dict mapping ticker to number of rows ingested
    """
    log = get_run_logger()
    log.info(f"Starting data ingestion (incremental={incremental})")

    if incremental:
        results = incremental_ingest(tickers)
    else:
        results = ingest_all_tickers(tickers)

    total = sum(results.values())
    log.info(f"Ingested {total} total rows")
    return results


@task(
    name="compute_features",
    description="Calculate technical indicators (EMA, RSI, ATR, volatility)",
)
def compute_features_task(tickers: Optional[list[str]] = None) -> dict[str, int]:
    """Compute technical features for all tickers.

    Args:
        tickers: List of ticker symbols (default: from config)

    Returns:
        Dict mapping ticker to number of feature rows computed
    """
    log = get_run_logger()
    log.info("Starting feature computation")

    results = build_all_features(tickers)

    # Return row counts
    counts = {ticker: len(df) for ticker, df in results.items()}
    total = sum(counts.values())
    log.info(f"Computed {total} total feature rows")
    return counts


@task(
    name="generate_signals",
    description="Generate trading signals from features",
)
def generate_signals_task(tickers: Optional[list[str]] = None) -> dict:
    """Generate trading signals for all tickers.

    Args:
        tickers: List of ticker symbols (default: from config)

    Returns:
        Summary of signals generated
    """
    log = get_run_logger()
    log.info("Starting signal generation")

    results = generate_all_signals(tickers)

    # Return summary
    summary = {
        "total_signals": sum(len(s) for s in results.values()),
        "by_ticker": {
            ticker: [
                {"type": s.signal_type.value, "price": s.price_at_signal}
                for s in signals
            ]
            for ticker, signals in results.items()
            if signals
        },
    }

    log.info(f"Generated {summary['total_signals']} signals")
    return summary


@task(
    name="apply_risk_management",
    description="Apply risk policy to signals",
)
def apply_risk_task(signals_summary: dict) -> dict:
    """Apply risk management to generated signals.

    Args:
        signals_summary: Summary from signal generation task

    Returns:
        Summary of risk orders
    """
    log = get_run_logger()
    log.info("Applying risk management")

    # Re-fetch signals from database and apply risk
    from ai_trading.signals import get_recent_signals, Signal, SignalType
    from ai_trading.risk_engine import apply_risk_to_all_signals

    # Get recent signals
    signals_df = get_recent_signals(limit=50)

    # Convert to Signal objects
    signals_dict = {}
    for _, row in signals_df.iterrows():
        ticker = row["ticker"]
        if ticker not in signals_dict:
            signals_dict[ticker] = []
        signals_dict[ticker].append(
            Signal(
                time=row["time"],
                ticker=ticker,
                signal_type=SignalType(row["signal_type"]),
                strength=row["strength"],
                price_at_signal=row["price_at_signal"],
            )
        )

    # Apply risk
    orders = apply_risk_to_all_signals(signals_dict)

    summary = {
        "total_orders": len(orders),
        "approved": sum(1 for o in orders if o.approved),
        "rejected": sum(1 for o in orders if not o.approved),
    }

    log.info(f"Risk processed: {summary['approved']} approved, {summary['rejected']} rejected")
    return summary


@task(
    name="execute_orders",
    description="Execute approved orders in paper trading",
)
def execute_orders_task(risk_summary: dict) -> dict:
    """Execute approved risk orders.

    Args:
        risk_summary: Summary from risk management task

    Returns:
        Summary of executions
    """
    log = get_run_logger()
    log.info("Executing approved orders")

    executions = execute_all_orders()

    summary = {
        "total_executions": len(executions),
        "total_bought": sum(e.total_value for e in executions if e.side == "BUY"),
        "total_sold": sum(e.total_value for e in executions if e.side == "SELL"),
    }

    log.info(
        f"Executed {summary['total_executions']} orders: "
        f"${summary['total_bought']:.2f} bought, ${summary['total_sold']:.2f} sold"
    )
    return summary


@flow(
    name="daily_e2e_pipeline",
    description="Daily end-to-end trading pipeline",
)
def daily_pipeline(
    tickers: Optional[list[str]] = None,
    incremental: bool = True,
) -> dict:
    """Run the complete daily trading pipeline.

    Args:
        tickers: List of ticker symbols (default: from config)
        incremental: If True, only fetch new data

    Returns:
        Summary of pipeline execution
    """
    log = get_run_logger()
    log.info("Starting daily E2E pipeline")
    start_time = datetime.now()

    # Step 1: Ingest data
    ingestion_result = ingest_data_task(tickers, incremental)

    # Step 2: Compute features
    features_result = compute_features_task(tickers)

    # Step 3: Generate signals
    signals_result = generate_signals_task(tickers)

    # Step 4: Apply risk management
    risk_result = apply_risk_task(signals_result)

    # Step 5: Execute orders
    execution_result = execute_orders_task(risk_result)

    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    summary = {
        "status": "completed",
        "duration_seconds": duration,
        "ingestion": ingestion_result,
        "features": features_result,
        "signals": signals_result,
        "risk": risk_result,
        "execution": execution_result,
    }

    log.info(f"Pipeline completed in {duration:.2f} seconds")
    return summary


@flow(
    name="run_backtest",
    description="Run EMA crossover backtest",
)
def run_backtest_flow(
    ticker: str = "SPY",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: Optional[float] = None,
) -> dict:
    """Run a backtest on historical data.

    Args:
        ticker: Ticker symbol to backtest
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital

    Returns:
        Backtest results summary
    """
    log = get_run_logger()
    log.info(f"Running backtest for {ticker}")

    result = run_ema_crossover_backtest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )

    summary = {
        "ticker": result.ticker,
        "total_return": result.total_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
    }

    log.info(f"Backtest complete: {result.total_return:.2%} return")
    return summary


@flow(
    name="initialize_system",
    description="Initialize the trading system (portfolio, etc.)",
)
def initialize_system_flow(
    initial_capital: Optional[float] = None,
    full_data_load: bool = False,
) -> dict:
    """Initialize the trading system.

    Args:
        initial_capital: Starting portfolio capital
        full_data_load: If True, load all historical data

    Returns:
        Initialization summary
    """
    log = get_run_logger()
    log.info("Initializing trading system")

    # Initialize portfolio
    initialize_portfolio(initial_capital)
    log.info("Portfolio initialized")

    # Optionally load full historical data
    if full_data_load:
        ingestion_result = ingest_data_task(incremental=False)
        features_result = compute_features_task()
    else:
        ingestion_result = {}
        features_result = {}

    return {
        "status": "initialized",
        "initial_capital": initial_capital or config.trading.initial_capital,
        "data_loaded": full_data_load,
        "ingestion": ingestion_result,
        "features": features_result,
    }


if __name__ == "__main__":
    # Example: run the daily pipeline locally
    result = daily_pipeline()
    print(result)
