"""Trading signal generation based on technical indicators.

Generates BUY/SELL/HOLD signals using EMA crossover strategy:
- BUY: EMA12 crosses above EMA26
- SELL: EMA12 crosses below EMA26
- HOLD: No crossover
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd
from sqlalchemy import text

from ai_trading.shared.config import config, get_db_engine

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Trading signal types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Represents a trading signal."""

    time: datetime
    ticker: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    price_at_signal: float

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "time": self.time,
            "ticker": self.ticker,
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "price_at_signal": self.price_at_signal,
        }


def load_features_from_db(ticker: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Load feature data from database for a ticker.

    Args:
        ticker: Stock/ETF ticker symbol
        limit: Maximum number of rows to fetch (most recent)

    Returns:
        DataFrame with feature data sorted by time
    """
    engine = get_db_engine()

    limit_clause = f"LIMIT {limit}" if limit else ""
    query = text(f"""
        SELECT f.time, f.ticker, f.ema_12, f.ema_26, f.rsi_14, f.atr_14, f.volatility_20,
               p.adj_close
        FROM features f
        JOIN prices p ON f.time = p.time AND f.ticker = p.ticker
        WHERE f.ticker = :ticker
        ORDER BY f.time DESC
        {limit_clause}
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})

    return df.sort_values("time").reset_index(drop=True)


def generate_ema_crossover_signal(
    ema_short: float,
    ema_long: float,
    prev_ema_short: float,
    prev_ema_long: float,
    rsi: float,
) -> tuple[SignalType, float]:
    """Generate signal based on EMA crossover.

    Args:
        ema_short: Current short EMA (12)
        ema_long: Current long EMA (26)
        prev_ema_short: Previous short EMA
        prev_ema_long: Previous long EMA
        rsi: Current RSI for signal strength

    Returns:
        Tuple of (SignalType, strength)
    """
    # Detect crossover
    current_diff = ema_short - ema_long
    prev_diff = prev_ema_short - prev_ema_long

    # Calculate signal strength based on RSI
    # RSI > 70: overbought (weaker buy, stronger sell)
    # RSI < 30: oversold (stronger buy, weaker sell)
    def calculate_strength(signal_type: SignalType) -> float:
        if signal_type == SignalType.BUY:
            # Stronger signal when oversold (low RSI)
            if rsi < 30:
                return 0.9
            elif rsi > 70:
                return 0.3
            else:
                return 0.5 + (50 - rsi) / 100
        elif signal_type == SignalType.SELL:
            # Stronger signal when overbought (high RSI)
            if rsi > 70:
                return 0.9
            elif rsi < 30:
                return 0.3
            else:
                return 0.5 + (rsi - 50) / 100
        return 0.0

    # Golden cross: short EMA crosses above long EMA
    if prev_diff <= 0 and current_diff > 0:
        return SignalType.BUY, calculate_strength(SignalType.BUY)

    # Death cross: short EMA crosses below long EMA
    if prev_diff >= 0 and current_diff < 0:
        return SignalType.SELL, calculate_strength(SignalType.SELL)

    return SignalType.HOLD, 0.0


def generate_signals_for_ticker(
    ticker: str,
    lookback: int = 2,
    save_to_db: bool = True,
) -> list[Signal]:
    """Generate signals for a single ticker based on recent features.

    Args:
        ticker: Stock/ETF ticker symbol
        lookback: Number of recent rows to analyze
        save_to_db: Whether to save signals to database

    Returns:
        List of generated signals
    """
    logger.info(f"Generating signals for {ticker}")

    df = load_features_from_db(ticker, limit=lookback + 1)

    if len(df) < 2:
        logger.warning(f"Not enough data for {ticker} to generate signals")
        return []

    signals = []

    # Generate signal for the most recent data point
    current = df.iloc[-1]
    previous = df.iloc[-2]

    signal_type, strength = generate_ema_crossover_signal(
        ema_short=current["ema_12"],
        ema_long=current["ema_26"],
        prev_ema_short=previous["ema_12"],
        prev_ema_long=previous["ema_26"],
        rsi=current["rsi_14"],
    )

    if signal_type != SignalType.HOLD:
        signal = Signal(
            time=pd.to_datetime(current["time"]),
            ticker=ticker,
            signal_type=signal_type,
            strength=strength,
            price_at_signal=current["adj_close"],
        )
        signals.append(signal)

        if save_to_db:
            _save_signal_to_db(signal)

        logger.info(
            f"Generated {signal_type.value} signal for {ticker} "
            f"at ${signal.price_at_signal:.2f} (strength: {strength:.2f})"
        )
    else:
        logger.info(f"No signal for {ticker} (HOLD)")

    return signals


def _save_signal_to_db(signal: Signal) -> None:
    """Save a signal to the database.

    Args:
        signal: Signal to save
    """
    engine = get_db_engine()

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO signals (time, ticker, signal_type, strength, price_at_signal)
                VALUES (:time, :ticker, :signal_type, :strength, :price_at_signal)
            """),
            signal.to_dict(),
        )


def generate_all_signals(
    tickers: Optional[list[str]] = None,
    save_to_db: bool = True,
) -> dict[str, list[Signal]]:
    """Generate signals for all configured tickers.

    Args:
        tickers: List of ticker symbols (default: from config)
        save_to_db: Whether to save signals to database

    Returns:
        Dict mapping ticker to list of signals
    """
    if tickers is None:
        tickers = config.trading.tickers

    results = {}
    for ticker in tickers:
        try:
            signals = generate_signals_for_ticker(ticker, save_to_db=save_to_db)
            results[ticker] = signals
        except Exception as e:
            logger.error(f"Failed to generate signals for {ticker}: {e}")
            results[ticker] = []

    # Summary
    total_signals = sum(len(s) for s in results.values())
    buy_count = sum(
        1 for signals in results.values()
        for s in signals if s.signal_type == SignalType.BUY
    )
    sell_count = sum(
        1 for signals in results.values()
        for s in signals if s.signal_type == SignalType.SELL
    )

    logger.info(
        f"Generated {total_signals} signals: {buy_count} BUY, {sell_count} SELL"
    )
    return results


def get_recent_signals(
    ticker: Optional[str] = None,
    limit: int = 10,
) -> pd.DataFrame:
    """Get recent signals from the database.

    Args:
        ticker: Filter by ticker (optional)
        limit: Maximum number of signals to fetch

    Returns:
        DataFrame with recent signals
    """
    engine = get_db_engine()

    where_clause = "WHERE ticker = :ticker" if ticker else ""
    query = text(f"""
        SELECT time, ticker, signal_type, strength, price_at_signal, created_at
        FROM signals
        {where_clause}
        ORDER BY time DESC
        LIMIT :limit
    """)

    params = {"limit": limit}
    if ticker:
        params["ticker"] = ticker

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: generate signals for all tickers
    results = generate_all_signals()
    for ticker, signals in results.items():
        for s in signals:
            print(f"{ticker}: {s.signal_type.value} at ${s.price_at_signal:.2f}")
