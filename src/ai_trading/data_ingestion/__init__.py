"""Data ingestion module for fetching market data."""

from .main import (
    fetch_prices,
    ingest_ticker,
    ingest_all_tickers,
    incremental_ingest,
    get_latest_date,
)

__all__ = [
    "fetch_prices",
    "ingest_ticker",
    "ingest_all_tickers",
    "incremental_ingest",
    "get_latest_date",
]
