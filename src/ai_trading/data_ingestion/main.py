"""Data ingestion from Yahoo Finance.

Downloads OHLCV data for configured tickers and stores in TimescaleDB.
Uses yfinance with auto_adjust=False to get raw prices including Adj Close.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import text

from ai_trading.shared.config import config, get_db_engine

logger = logging.getLogger(__name__)


def fetch_prices(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock/ETF ticker symbol
        start_date: Start date in YYYY-MM-DD format (default: 2 years ago)
        end_date: End date in YYYY-MM-DD format (default: today)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
    """
    if start_date is None:
        start_date = config.backtest.start_date
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Fetching {ticker} from {start_date} to {end_date}")

    # Use auto_adjust=False to get raw prices with Adj Close column
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        logger.warning(f"No data returned for {ticker}")
        return pd.DataFrame()

    # Flatten multi-index columns if present (happens with single ticker)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Add ticker column
    data["ticker"] = ticker

    # Reset index to have time as column
    data = data.reset_index()
    data = data.rename(columns={"Date": "time"})

    logger.info(f"Fetched {len(data)} rows for {ticker}")
    return data


def ingest_ticker(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """Ingest price data for a single ticker into the database.

    Args:
        ticker: Stock/ETF ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Number of rows inserted
    """
    df = fetch_prices(ticker, start_date, end_date)

    if df.empty:
        return 0

    # Prepare data for insertion
    df_insert = df[["time", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df_insert.columns = ["time", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

    engine = get_db_engine()

    # Use upsert (INSERT ON CONFLICT) for idempotency
    with engine.begin() as conn:
        for _, row in df_insert.iterrows():
            conn.execute(
                text("""
                    INSERT INTO prices (time, ticker, open, high, low, close, adj_close, volume)
                    VALUES (:time, :ticker, :open, :high, :low, :close, :adj_close, :volume)
                    ON CONFLICT (time, ticker) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        adj_close = EXCLUDED.adj_close,
                        volume = EXCLUDED.volume
                """),
                {
                    "time": row["time"],
                    "ticker": row["ticker"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "adj_close": row["adj_close"],
                    "volume": row["volume"],
                },
            )

    logger.info(f"Ingested {len(df_insert)} rows for {ticker}")
    return len(df_insert)


def ingest_all_tickers(
    tickers: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, int]:
    """Ingest price data for all configured tickers.

    Args:
        tickers: List of ticker symbols (default: from config)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dict mapping ticker to number of rows inserted
    """
    if tickers is None:
        tickers = config.trading.tickers

    results = {}
    for ticker in tickers:
        try:
            count = ingest_ticker(ticker, start_date, end_date)
            results[ticker] = count
        except Exception as e:
            logger.error(f"Failed to ingest {ticker}: {e}")
            results[ticker] = 0

    total = sum(results.values())
    logger.info(f"Total ingested: {total} rows across {len(tickers)} tickers")
    return results


def get_latest_date(ticker: str) -> Optional[datetime]:
    """Get the latest date in the database for a ticker.

    Args:
        ticker: Stock/ETF ticker symbol

    Returns:
        Latest datetime or None if no data exists
    """
    engine = get_db_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(time) FROM prices WHERE ticker = :ticker"),
            {"ticker": ticker},
        )
        row = result.fetchone()
        return row[0] if row and row[0] else None


def incremental_ingest(tickers: Optional[list[str]] = None) -> dict[str, int]:
    """Incrementally ingest only new data since last ingestion.

    Args:
        tickers: List of ticker symbols (default: from config)

    Returns:
        Dict mapping ticker to number of new rows inserted
    """
    if tickers is None:
        tickers = config.trading.tickers

    results = {}
    for ticker in tickers:
        latest = get_latest_date(ticker)
        if latest:
            # Start from day after latest
            start_date = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start_date = config.backtest.start_date

        count = ingest_ticker(ticker, start_date=start_date)
        results[ticker] = count

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: ingest all configured tickers
    results = ingest_all_tickers()
    print(results)
