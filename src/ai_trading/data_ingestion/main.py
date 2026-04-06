"""Data ingestion from Alpaca Markets API.

Downloads OHLCV data for configured tickers from Alpaca and stores in TimescaleDB.
Alpaca provides clean, institutional-grade market data with proper adjustments.

Rate limits (Paper Trading):
- 200 requests per minute for market data
- Automatic retry with exponential backoff
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import text

from ai_trading.shared.config import config, get_db_engine
from ai_trading.broker.alpaca_broker import AlpacaBroker, AlpacaConfig

logger = logging.getLogger(__name__)


class AlpacaDataClient:
    """Client for Alpaca market data with rate limiting and retries."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize Alpaca data client.

        Args:
            api_key: Alpaca API key (optional, uses env var if not provided)
            api_secret: Alpaca API secret (optional, uses env var if not provided)
        """
        # Use provided keys or let AlpacaConfig read from env vars
        alpaca_config = AlpacaConfig()
        if api_key:
            alpaca_config.api_key = api_key
        if api_secret:
            alpaca_config.api_secret = api_secret
        alpaca_config.paper = True  # Data is same for paper/live

        self.broker = AlpacaBroker(alpaca_config)
        self._last_request_time = 0
        self._min_delay = 0.3  # 300ms between requests (200 req/min safe limit)

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_request_time = time.time()

    def fetch_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from Alpaca API.

        Args:
            ticker: Stock/ETF ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Bar timeframe ("1Min", "1Hour", "1Day")

        Returns:
            DataFrame with columns: time, open, high, low, close, volume, ticker
        """
        logger.info(f"Fetching {ticker} from Alpaca: {start_date} to {end_date}")

        # Convert dates to ISO format with timezone
        start_iso = f"{start_date}T00:00:00Z"
        end_iso = f"{end_date}T23:59:59Z"

        all_bars = []
        page_token = None
        max_pages = 50  # Safety limit
        page_count = 0

        while page_count < max_pages:
            self._rate_limit()

            try:
                # Build request with pagination
                url = f"{self.broker.config.data_url}/v2/stocks/{ticker}/bars"
                params = {
                    "timeframe": timeframe,
                    "start": start_iso,
                    "end": end_iso,
                    "limit": 10000,  # Max per request
                }
                if page_token:
                    params["page_token"] = page_token

                response = self.broker._session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                bars = data.get("bars", [])
                if not bars:
                    break

                all_bars.extend(bars)

                # Check for next page
                page_token = data.get("next_page_token")
                if not page_token:
                    break

                page_count += 1
                logger.debug(f"Fetched page {page_count} for {ticker}, {len(bars)} bars")

            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                if "429" in str(e):  # Rate limit
                    logger.warning("Rate limited, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                raise

        if not all_bars:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_bars)

        # Rename columns to match database schema
        df = df.rename(columns={
            "t": "time",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })

        # Parse timestamp
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

        # Add ticker column
        df["ticker"] = ticker

        # Select required columns (Alpaca doesn't provide adj_close separately)
        df = df[["time", "ticker", "open", "high", "low", "close", "volume"]]

        # For Alpaca, close prices are already split-adjusted
        # We'll use close as adj_close for consistency
        df["adj_close"] = df["close"]

        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df


def fetch_prices(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from Alpaca.

    Args:
        ticker: Stock/ETF ticker symbol
        start_date: Start date in YYYY-MM-DD format (default: 2 years ago)
        end_date: End date in YYYY-MM-DD format (default: today)

    Returns:
        DataFrame with columns: time, ticker, open, high, low, close, adj_close, volume
    """
    if start_date is None:
        start_date = config.backtest.start_date
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    client = AlpacaDataClient()
    return client.fetch_bars(ticker, start_date, end_date)


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
    df_insert = df[["time", "ticker", "open", "high", "low", "close", "adj_close", "volume"]].copy()

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
