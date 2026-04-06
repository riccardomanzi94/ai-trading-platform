"""FRED (Federal Reserve Economic Data) API client.

Downloads macroeconomic indicators from the Federal Reserve Bank of St. Louis.
Useful for understanding the broader economic context of trading signals.

Common FRED Series:
- DFF: Effective Federal Funds Rate (daily)
- DGS10: 10-Year Treasury Constant Maturity Rate (daily)
- DGS2: 2-Year Treasury Constant Maturity Rate (daily)
- T10Y2Y: 10-Year minus 2-Year Treasury Yield Spread (recession indicator)
- CPIAUCSL: Consumer Price Index (monthly)
- UNRATE: Unemployment Rate (monthly)
- GDP: Gross Domestic Product (quarterly)
- VIXCLS: CBOE Volatility Index (daily)
- DXY: US Dollar Index (daily)

Requires: FRED_API_KEY environment variable (free at research.stlouisfed.org/useraccount)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from sqlalchemy import text

from ai_trading.shared.config import get_db_engine

logger = logging.getLogger(__name__)


@dataclass
class FREDSeries:
    """Represents a FRED data series."""

    series_id: str
    name: str
    frequency: str  # d=daily, w=weekly, m=monthly, q=quarterly
    unit: str
    description: str = ""


# Common FRED series used for trading analysis
FRED_SERIES = {
    "DFF": FREDSeries(
        series_id="DFF",
        name="Effective Federal Funds Rate",
        frequency="d",
        unit="Percent",
        description="The interest rate at which depository institutions trade federal funds",
    ),
    "DGS10": FREDSeries(
        series_id="DGS10",
        name="10-Year Treasury Constant Maturity Rate",
        frequency="d",
        unit="Percent",
        description="Market yield on U.S. Treasury securities at 10-year constant maturity",
    ),
    "DGS2": FREDSeries(
        series_id="DGS2",
        name="2-Year Treasury Constant Maturity Rate",
        frequency="d",
        unit="Percent",
        description="Market yield on U.S. Treasury securities at 2-year constant maturity",
    ),
    "T10Y2Y": FREDSeries(
        series_id="T10Y2Y",
        name="10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity",
        frequency="d",
        unit="Percent",
        description="Yield curve spread - inverts before recessions",
    ),
    "CPIAUCSL": FREDSeries(
        series_id="CPIAUCSL",
        name="Consumer Price Index for All Urban Consumers",
        frequency="m",
        unit="Index 1982-1984=100",
        description="Measure of average change over time in prices paid by urban consumers",
    ),
    "UNRATE": FREDSeries(
        series_id="UNRATE",
        name="Unemployment Rate",
        frequency="m",
        unit="Percent",
        description="Percentage of labor force that is unemployed",
    ),
    "GDP": FREDSeries(
        series_id="GDP",
        name="Gross Domestic Product",
        frequency="q",
        unit="Billions of Dollars",
        description="Total value of goods produced and services provided in the US",
    ),
    "VIXCLS": FREDSeries(
        series_id="VIXCLS",
        name="CBOE Volatility Index",
        frequency="d",
        unit="Index",
        description="Market's expectation of 30-day volatility implied by S&P 500 index options",
    ),
    "DTWEXBGS": FREDSeries(
        series_id="DTWEXBGS",
        name="Trade Weighted U.S. Dollar Index",
        frequency="d",
        unit="Index",
        description="Broad dollar index against currencies of major US trading partners",
    ),
}


class FREDClient:
    """Client for FRED API."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize FRED client.

        Args:
            api_key: FRED API key (optional, uses FRED_API_KEY env var)

        Raises:
            ValueError: If no API key is provided or found
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Get one free at: "
                "https://research.stlouisfed.org/useraccount/apikey"
            )
        self._session = requests.Session()

    def get_series(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch observations for a FRED series.

        Args:
            series_id: FRED series ID (e.g., "DFF", "CPIAUCSL")
            observation_start: Start date (YYYY-MM-DD), defaults to 5 years ago
            observation_end: End date (YYYY-MM-DD), defaults to today

        Returns:
            DataFrame with columns: time, value
        """
        # Default dates
        if not observation_end:
            observation_end = datetime.now().strftime("%Y-%m-%d")
        if not observation_start:
            observation_start = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "observation_end": observation_end,
            "sort_order": "asc",
        }

        logger.info(f"Fetching FRED series {series_id} from {observation_start} to {observation_end}")

        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "observations" not in data:
                logger.warning(f"No observations found for {series_id}")
                return pd.DataFrame()

            observations = data["observations"]
            df = pd.DataFrame(observations)

            # Parse date and value
            df["time"] = pd.to_datetime(df["date"])
            # FRED uses "." for missing values
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            # Drop rows with missing values
            df = df.dropna(subset=["value"])

            logger.info(f"Fetched {len(df)} observations for {series_id}")
            return df[["time", "value"]]

        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            raise

    def get_current_value(self, series_id: str) -> Optional[float]:
        """Get the most recent value for a series.

        Args:
            series_id: FRED series ID

        Returns:
            Latest value or None if not available
        """
        df = self.get_series(series_id, observation_start=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        if not df.empty:
            return df.iloc[-1]["value"]
        return None

    def get_all_series_info(self) -> pd.DataFrame:
        """Get information about all configured FRED series.

        Returns:
            DataFrame with series metadata
        """
        data = []
        for series in FRED_SERIES.values():
            data.append({
                "series_id": series.series_id,
                "name": series.name,
                "frequency": series.frequency,
                "unit": series.unit,
                "description": series.description,
            })
        return pd.DataFrame(data)


def ingest_macro_series(
    series_id: str,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
) -> int:
    """Ingest a FRED series into the database.

    Args:
        series_id: FRED series ID
        observation_start: Start date (YYYY-MM-DD)
        observation_end: End date (YYYY-MM-DD)

    Returns:
        Number of rows inserted
    """
    client = FREDClient()
    series_info = FRED_SERIES.get(series_id)

    if not series_info:
        logger.warning(f"Unknown series {series_id}, using defaults")
        series_info = FREDSeries(
            series_id=series_id,
            name=series_id,
            frequency="d",
            unit="unknown",
        )

    df = client.get_series(series_id, observation_start, observation_end)

    if df.empty:
        return 0

    # Prepare data for insertion
    df["series_id"] = series_id
    df["series_name"] = series_info.name
    df["unit"] = series_info.unit
    df["frequency"] = series_info.frequency

    engine = get_db_engine()

    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO macro_data (time, series_id, series_name, value, unit, frequency)
                    VALUES (:time, :series_id, :series_name, :value, :unit, :frequency)
                    ON CONFLICT (time, series_id) DO UPDATE SET
                        value = EXCLUDED.value,
                        last_updated = NOW()
                """),
                {
                    "time": row["time"],
                    "series_id": row["series_id"],
                    "series_name": row["series_name"],
                    "value": row["value"],
                    "unit": row["unit"],
                    "frequency": row["frequency"],
                },
            )

    logger.info(f"Ingested {len(df)} rows for {series_id}")
    return len(df)


def ingest_all_macro_series(
    series_ids: Optional[list[str]] = None,
    observation_start: Optional[str] = None,
) -> dict[str, int]:
    """Ingest all configured FRED macro series.

    Args:
        series_ids: List of FRED series IDs (default: all configured)
        observation_start: Start date (YYYY-MM-DD)

    Returns:
        Dict mapping series_id to number of rows inserted
    """
    if series_ids is None:
        series_ids = list(FRED_SERIES.keys())

    results = {}
    for series_id in series_ids:
        try:
            count = ingest_macro_series(series_id, observation_start)
            results[series_id] = count
            # Be nice to FRED API (rate limiting)
            import time
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Failed to ingest {series_id}: {e}")
            results[series_id] = 0

    total = sum(results.values())
    logger.info(f"Total ingested: {total} macro observations across {len(series_ids)} series")
    return results


def get_macro_data(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Get macro data from the database.

    Args:
        series_id: FRED series ID
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)

    Returns:
        DataFrame with macro data
    """
    engine = get_db_engine()

    query = """
        SELECT time, series_id, series_name, value, unit, frequency
        FROM macro_data
        WHERE series_id = :series_id
    """
    params = {"series_id": series_id}

    if start_date:
        query += " AND time >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND time <= :end_date"
        params["end_date"] = end_date

    query += " ORDER BY time ASC"

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    return df


def get_yield_curve_status() -> dict:
    """Get current yield curve status (recession indicator).

    Returns:
        Dict with current 10Y-2Y spread and interpretation
    """
    client = FREDClient()
    spread = client.get_current_value("T10Y2Y")

    if spread is None:
        return {"error": "Could not fetch yield curve data"}

    return {
        "spread_10y_2y": spread,
        "inverted": spread < 0,
        "interpretation": (
            "Inverted (recession warning)" if spread < 0
            else "Flattening" if spread < 0.5
            else "Normal"
        ),
        "last_updated": datetime.now().isoformat(),
    }


def get_economic_summary() -> pd.DataFrame:
    """Get a summary of key economic indicators.

    Returns:
        DataFrame with current values of key indicators
    """
    client = FREDClient()

    key_series = ["DFF", "DGS10", "T10Y2Y", "UNRATE", "CPIAUCSL"]
    data = []

    for series_id in key_series:
        try:
            value = client.get_current_value(series_id)
            series_info = FRED_SERIES.get(series_id)
            if value is not None and series_info:
                data.append({
                    "Indicator": series_info.name,
                    "Value": value,
                    "Unit": series_info.unit,
                    "Series ID": series_id,
                })
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")

    return pd.DataFrame(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: show configured series
    print("Configured FRED Series:")
    print("-" * 80)
    for series in FRED_SERIES.values():
        print(f"\n{series.series_id}: {series.name}")
        print(f"  Frequency: {series.frequency}, Unit: {series.unit}")
        print(f"  {series.description}")

    # Example: fetch and display current economic summary
    print("\n" + "=" * 80)
    print("Current Economic Summary:")
    print("=" * 80)
    try:
        summary = get_economic_summary()
        print(summary.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Set FRED_API_KEY environment variable to use this feature")
        print("Get a free API key at: https://research.stlouisfed.org/useraccount/apikey")
