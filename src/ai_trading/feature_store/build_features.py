"""Feature engineering for technical indicators.

Computes:
- EMA (Exponential Moving Average) - 12 and 26 periods
- RSI (Relative Strength Index) - 14 periods
- ATR (Average True Range) - 14 periods
- Rolling Volatility (standard deviation) - 20 periods
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from ai_trading.shared.config import config, get_db_engine

logger = logging.getLogger(__name__)


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average.

    Args:
        series: Price series (typically Adj Close)
        period: Number of periods for EMA

    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    where RS = avg_gain / avg_loss over the period

    Args:
        series: Price series
        period: Number of periods (default: 14)

    Returns:
        RSI series (values between 0 and 100)
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle division by zero (when avg_loss is 0)
    rsi = rsi.replace([np.inf, -np.inf], 100)
    rsi = rsi.fillna(50)  # Neutral RSI when undefined

    return rsi


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range.

    True Range is the greatest of:
    - Current High - Current Low
    - |Current High - Previous Close|
    - |Current Low - Previous Close|

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods (default: 14)

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def compute_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Compute rolling volatility (annualized standard deviation of returns).

    Args:
        series: Price series
        period: Rolling window size (default: 20)

    Returns:
        Annualized volatility series
    """
    returns = series.pct_change()
    volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
    return volatility


def load_prices_from_db(ticker: str) -> pd.DataFrame:
    """Load price data from database for a ticker.

    Args:
        ticker: Stock/ETF ticker symbol

    Returns:
        DataFrame with price data sorted by time
    """
    engine = get_db_engine()
    query = text("""
        SELECT time, open, high, low, close, adj_close, volume
        FROM prices
        WHERE ticker = :ticker
        ORDER BY time ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})

    df["time"] = pd.to_datetime(df["time"])
    return df


def build_features_for_ticker(ticker: str, save_to_db: bool = True) -> pd.DataFrame:
    """Build all technical features for a single ticker.

    Args:
        ticker: Stock/ETF ticker symbol
        save_to_db: Whether to save features to database

    Returns:
        DataFrame with features
    """
    logger.info(f"Building features for {ticker}")

    df = load_prices_from_db(ticker)

    if df.empty:
        logger.warning(f"No price data for {ticker}")
        return pd.DataFrame()

    # Compute features
    features = pd.DataFrame()
    features["time"] = df["time"]
    features["ticker"] = ticker

    # EMA
    features["ema_12"] = compute_ema(df["adj_close"], config.features.ema_short)
    features["ema_26"] = compute_ema(df["adj_close"], config.features.ema_long)

    # RSI
    features["rsi_14"] = compute_rsi(df["adj_close"], config.features.rsi_period)

    # ATR
    features["atr_14"] = compute_atr(
        df["high"],
        df["low"],
        df["close"],
        config.features.atr_period,
    )

    # Volatility
    features["volatility_20"] = compute_volatility(
        df["adj_close"],
        config.features.volatility_period,
    )

    # Drop rows with NaN (first few rows won't have enough data)
    features = features.dropna()

    if save_to_db and not features.empty:
        _save_features_to_db(features)

    logger.info(f"Built {len(features)} feature rows for {ticker}")
    return features


def _save_features_to_db(features: pd.DataFrame) -> None:
    """Save computed features to database.

    Args:
        features: DataFrame with features to save
    """
    engine = get_db_engine()

    with engine.begin() as conn:
        for _, row in features.iterrows():
            conn.execute(
                text("""
                    INSERT INTO features (time, ticker, ema_12, ema_26, rsi_14, atr_14, volatility_20)
                    VALUES (:time, :ticker, :ema_12, :ema_26, :rsi_14, :atr_14, :volatility_20)
                    ON CONFLICT (time, ticker) DO UPDATE SET
                        ema_12 = EXCLUDED.ema_12,
                        ema_26 = EXCLUDED.ema_26,
                        rsi_14 = EXCLUDED.rsi_14,
                        atr_14 = EXCLUDED.atr_14,
                        volatility_20 = EXCLUDED.volatility_20
                """),
                {
                    "time": row["time"],
                    "ticker": row["ticker"],
                    "ema_12": row["ema_12"],
                    "ema_26": row["ema_26"],
                    "rsi_14": row["rsi_14"],
                    "atr_14": row["atr_14"],
                    "volatility_20": row["volatility_20"],
                },
            )


def build_all_features(
    tickers: Optional[list[str]] = None,
    save_to_db: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build features for all configured tickers.

    Args:
        tickers: List of ticker symbols (default: from config)
        save_to_db: Whether to save features to database

    Returns:
        Dict mapping ticker to features DataFrame
    """
    if tickers is None:
        tickers = config.trading.tickers

    results = {}
    for ticker in tickers:
        try:
            features = build_features_for_ticker(ticker, save_to_db)
            results[ticker] = features
        except Exception as e:
            logger.error(f"Failed to build features for {ticker}: {e}")
            results[ticker] = pd.DataFrame()

    return results


def get_latest_features(ticker: str, n_rows: int = 1) -> pd.DataFrame:
    """Get the most recent feature rows for a ticker.

    Args:
        ticker: Stock/ETF ticker symbol
        n_rows: Number of recent rows to fetch

    Returns:
        DataFrame with latest features
    """
    engine = get_db_engine()
    query = text("""
        SELECT time, ticker, ema_12, ema_26, rsi_14, atr_14, volatility_20
        FROM features
        WHERE ticker = :ticker
        ORDER BY time DESC
        LIMIT :n_rows
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "n_rows": n_rows})

    return df.sort_values("time")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: build features for all tickers
    results = build_all_features()
    for ticker, df in results.items():
        print(f"{ticker}: {len(df)} feature rows")
