"""RSI Mean Reversion Strategy.

Generates signals based on RSI overbought/oversold conditions:
- BUY: RSI < 30 (oversold) and starts recovering
- SELL: RSI > 70 (overbought) and starts declining
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import text

from ai_trading.shared.config import config, get_db_engine
from ai_trading.signals.generate_signals import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class RSIStrategyConfig:
    """Configuration for RSI Mean Reversion strategy."""
    
    oversold_threshold: float = 30.0
    overbought_threshold: float = 70.0
    confirmation_periods: int = 2  # Bars to confirm reversal
    min_rsi_change: float = 3.0  # Minimum RSI change for confirmation


def load_rsi_data(ticker: str, limit: int = 10) -> pd.DataFrame:
    """Load recent RSI data for a ticker.

    Args:
        ticker: Stock/ETF ticker symbol
        limit: Number of recent rows to fetch

    Returns:
        DataFrame with RSI data
    """
    engine = get_db_engine()
    query = text("""
        SELECT f.time, f.rsi_14, p.adj_close
        FROM features f
        JOIN prices p ON f.time = p.time AND f.ticker = p.ticker
        WHERE f.ticker = :ticker
        ORDER BY f.time DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "limit": limit})

    return df.sort_values("time").reset_index(drop=True)


def generate_rsi_signal(
    ticker: str,
    strategy_config: Optional[RSIStrategyConfig] = None,
) -> Optional[Signal]:
    """Generate RSI mean reversion signal.

    Args:
        ticker: Stock/ETF ticker symbol
        strategy_config: Strategy configuration

    Returns:
        Signal if conditions are met, None otherwise
    """
    if strategy_config is None:
        strategy_config = RSIStrategyConfig()

    df = load_rsi_data(ticker, limit=strategy_config.confirmation_periods + 2)

    if len(df) < strategy_config.confirmation_periods + 1:
        logger.warning(f"Insufficient data for {ticker}")
        return None

    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    current_rsi = current["rsi_14"]
    prev_rsi = previous["rsi_14"]
    rsi_change = current_rsi - prev_rsi

    # Check for oversold recovery (BUY signal)
    if (
        prev_rsi < strategy_config.oversold_threshold
        and rsi_change > strategy_config.min_rsi_change
    ):
        # Calculate strength based on how oversold it was
        strength = min(1.0, (strategy_config.oversold_threshold - prev_rsi) / 20)
        strength = max(0.5, strength)  # Minimum strength 0.5
        
        logger.info(
            f"RSI BUY signal for {ticker}: RSI {prev_rsi:.1f} -> {current_rsi:.1f}"
        )
        
        return Signal(
            time=pd.to_datetime(current["time"]),
            ticker=ticker,
            signal_type=SignalType.BUY,
            strength=strength,
            price_at_signal=current["adj_close"],
        )

    # Check for overbought decline (SELL signal)
    if (
        prev_rsi > strategy_config.overbought_threshold
        and rsi_change < -strategy_config.min_rsi_change
    ):
        # Calculate strength based on how overbought it was
        strength = min(1.0, (prev_rsi - strategy_config.overbought_threshold) / 20)
        strength = max(0.5, strength)

        logger.info(
            f"RSI SELL signal for {ticker}: RSI {prev_rsi:.1f} -> {current_rsi:.1f}"
        )

        return Signal(
            time=pd.to_datetime(current["time"]),
            ticker=ticker,
            signal_type=SignalType.SELL,
            strength=strength,
            price_at_signal=current["adj_close"],
        )

    return None


def generate_rsi_signals_all(
    tickers: Optional[list[str]] = None,
    strategy_config: Optional[RSIStrategyConfig] = None,
) -> dict[str, list[Signal]]:
    """Generate RSI signals for all tickers.

    Args:
        tickers: List of ticker symbols
        strategy_config: Strategy configuration

    Returns:
        Dict mapping ticker to list of signals
    """
    if tickers is None:
        tickers = config.trading.tickers

    results = {}
    for ticker in tickers:
        signal = generate_rsi_signal(ticker, strategy_config)
        results[ticker] = [signal] if signal else []

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    signals = generate_rsi_signals_all()
    for ticker, sigs in signals.items():
        for s in sigs:
            print(f"{ticker}: {s.signal_type.value} @ ${s.price_at_signal:.2f}")
