"""Momentum Breakout Strategy.

Generates signals based on ATR breakouts:
- BUY: Price breaks above recent high + ATR multiplier
- SELL: Price breaks below recent low - ATR multiplier
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
class MomentumConfig:
    """Configuration for Momentum Breakout strategy."""
    
    lookback_period: int = 20  # Bars for high/low calculation
    atr_multiplier: float = 1.5  # ATR multiplier for breakout threshold
    volume_confirmation: bool = True  # Require above-average volume
    volume_multiplier: float = 1.2  # Volume must be this times average


def load_momentum_data(ticker: str, limit: int = 30) -> pd.DataFrame:
    """Load price and ATR data for momentum analysis.

    Args:
        ticker: Stock/ETF ticker symbol
        limit: Number of recent rows to fetch

    Returns:
        DataFrame with price, ATR, and volume data
    """
    engine = get_db_engine()
    query = text("""
        SELECT p.time, p.high, p.low, p.adj_close, p.volume, f.atr_14
        FROM prices p
        JOIN features f ON p.time = f.time AND p.ticker = f.ticker
        WHERE p.ticker = :ticker
        ORDER BY p.time DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "limit": limit})

    return df.sort_values("time").reset_index(drop=True)


def generate_momentum_signal(
    ticker: str,
    strategy_config: Optional[MomentumConfig] = None,
) -> Optional[Signal]:
    """Generate momentum breakout signal.

    Args:
        ticker: Stock/ETF ticker symbol
        strategy_config: Strategy configuration

    Returns:
        Signal if breakout conditions are met, None otherwise
    """
    if strategy_config is None:
        strategy_config = MomentumConfig()

    df = load_momentum_data(ticker, limit=strategy_config.lookback_period + 5)

    if len(df) < strategy_config.lookback_period:
        logger.warning(f"Insufficient data for {ticker}")
        return None

    current = df.iloc[-1]
    lookback = df.iloc[-(strategy_config.lookback_period + 1):-1]
    
    recent_high = lookback["high"].max()
    recent_low = lookback["low"].min()
    current_price = current["adj_close"]
    current_atr = current["atr_14"]
    
    # Calculate breakout levels
    upper_breakout = recent_high + (current_atr * strategy_config.atr_multiplier)
    lower_breakout = recent_low - (current_atr * strategy_config.atr_multiplier)
    
    # Volume confirmation
    if strategy_config.volume_confirmation:
        avg_volume = lookback["volume"].mean()
        volume_ok = current["volume"] > avg_volume * strategy_config.volume_multiplier
    else:
        volume_ok = True

    # Check for bullish breakout
    if current_price > upper_breakout and volume_ok:
        # Strength based on how far above breakout level
        breakout_magnitude = (current_price - upper_breakout) / current_atr
        strength = min(1.0, 0.5 + breakout_magnitude * 0.2)

        logger.info(
            f"MOMENTUM BUY for {ticker}: ${current_price:.2f} > ${upper_breakout:.2f}"
        )

        return Signal(
            time=pd.to_datetime(current["time"]),
            ticker=ticker,
            signal_type=SignalType.BUY,
            strength=strength,
            price_at_signal=current_price,
        )

    # Check for bearish breakdown
    if current_price < lower_breakout and volume_ok:
        breakout_magnitude = (lower_breakout - current_price) / current_atr
        strength = min(1.0, 0.5 + breakout_magnitude * 0.2)

        logger.info(
            f"MOMENTUM SELL for {ticker}: ${current_price:.2f} < ${lower_breakout:.2f}"
        )

        return Signal(
            time=pd.to_datetime(current["time"]),
            ticker=ticker,
            signal_type=SignalType.SELL,
            strength=strength,
            price_at_signal=current_price,
        )

    return None


def generate_momentum_signals_all(
    tickers: Optional[list[str]] = None,
    strategy_config: Optional[MomentumConfig] = None,
) -> dict[str, list[Signal]]:
    """Generate momentum signals for all tickers.

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
        signal = generate_momentum_signal(ticker, strategy_config)
        results[ticker] = [signal] if signal else []

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    signals = generate_momentum_signals_all()
    for ticker, sigs in signals.items():
        for s in sigs:
            print(f"{ticker}: {s.signal_type.value} @ ${s.price_at_signal:.2f}")
