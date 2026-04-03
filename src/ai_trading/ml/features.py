"""ML Feature Engineering for price prediction.

Creates features for machine learning models:
- Technical indicators (lagged)
- Price momentum features
- Volatility features
- Calendar features
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from ai_trading.shared.config import config, get_db_engine

logger = logging.getLogger(__name__)


def load_full_dataset(ticker: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """Load complete price and feature dataset.

    Args:
        ticker: Stock/ETF ticker symbol
        start_date: Start date for data

    Returns:
        DataFrame with prices and features
    """
    if start_date is None:
        start_date = config.backtest.start_date

    engine = get_db_engine()
    query = text("""
        SELECT 
            p.time, p.open, p.high, p.low, p.close, p.adj_close, p.volume,
            f.ema_12, f.ema_26, f.rsi_14, f.atr_14, f.volatility_20
        FROM prices p
        JOIN features f ON p.time = f.time AND p.ticker = f.ticker
        WHERE p.ticker = :ticker AND p.time >= :start_date
        ORDER BY p.time ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(
            query, conn, params={"ticker": ticker, "start_date": start_date}
        )

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df


def prepare_ml_features(
    df: pd.DataFrame,
    include_lags: bool = True,
    lag_periods: list[int] = None,
) -> pd.DataFrame:
    """Prepare features for ML model.

    Args:
        df: Raw price/feature DataFrame
        include_lags: Whether to include lagged features
        lag_periods: List of lag periods to include

    Returns:
        DataFrame with ML features
    """
    if lag_periods is None:
        lag_periods = [1, 2, 3, 5, 10]

    features = pd.DataFrame(index=df.index)

    # Price-based features
    features["return_1d"] = df["adj_close"].pct_change()
    features["return_5d"] = df["adj_close"].pct_change(5)
    features["return_10d"] = df["adj_close"].pct_change(10)
    features["return_20d"] = df["adj_close"].pct_change(20)

    # Momentum
    features["momentum_5d"] = df["adj_close"] / df["adj_close"].shift(5) - 1
    features["momentum_10d"] = df["adj_close"] / df["adj_close"].shift(10) - 1
    features["momentum_20d"] = df["adj_close"] / df["adj_close"].shift(20) - 1

    # Moving average ratios
    features["price_to_ema12"] = df["adj_close"] / df["ema_12"] - 1
    features["price_to_ema26"] = df["adj_close"] / df["ema_26"] - 1
    features["ema_ratio"] = df["ema_12"] / df["ema_26"] - 1

    # Technical indicators
    features["rsi"] = df["rsi_14"]
    features["rsi_change"] = df["rsi_14"].diff()
    features["atr"] = df["atr_14"]
    features["atr_pct"] = df["atr_14"] / df["adj_close"]
    features["volatility"] = df["volatility_20"]

    # Volume features
    features["volume_change"] = df["volume"].pct_change()
    features["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Price range features
    features["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    features["close_to_high"] = (df["high"] - df["close"]) / (df["high"] - df["low"] + 1e-8)
    features["close_to_low"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

    # Calendar features
    features["day_of_week"] = df.index.dayofweek
    features["month"] = df.index.month
    features["is_month_end"] = df.index.is_month_end.astype(int)
    features["is_quarter_end"] = df.index.is_quarter_end.astype(int)

    # Add lagged features
    if include_lags:
        base_cols = ["return_1d", "rsi", "volume_change", "atr_pct"]
        for col in base_cols:
            for lag in lag_periods:
                features[f"{col}_lag{lag}"] = features[col].shift(lag)

    # RSI zones
    features["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
    features["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)

    # Trend direction
    features["uptrend"] = (df["ema_12"] > df["ema_26"]).astype(int)

    return features


def create_target_variable(
    df: pd.DataFrame,
    horizon: int = 1,
    target_type: str = "direction",
    threshold: float = 0.0,
) -> pd.Series:
    """Create target variable for prediction.

    Args:
        df: DataFrame with adj_close column
        horizon: Prediction horizon in days
        target_type: "direction" (up/down), "return", or "threshold"
        threshold: For threshold type, minimum return to be positive

    Returns:
        Series with target values
    """
    future_return = df["adj_close"].shift(-horizon) / df["adj_close"] - 1

    if target_type == "direction":
        # Binary classification: 1 = up, 0 = down
        target = (future_return > threshold).astype(int)
    elif target_type == "return":
        # Regression target
        target = future_return
    elif target_type == "threshold":
        # 3-class: -1 = down more than threshold, 0 = neutral, 1 = up more than threshold
        target = pd.Series(0, index=df.index)
        target[future_return > threshold] = 1
        target[future_return < -threshold] = -1
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    target.name = "target"
    return target


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train, validation, and test sets.

    Time-series aware split (no shuffling, maintains temporal order).

    Args:
        features: Feature DataFrame
        target: Target Series
        test_ratio: Proportion of data for test
        validation_ratio: Proportion of data for validation

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Remove rows with NaN
    combined = pd.concat([features, target], axis=1).dropna()
    features_clean = combined[features.columns]
    target_clean = combined["target"]

    n = len(features_clean)
    test_size = int(n * test_ratio)
    val_size = int(n * validation_ratio)
    train_size = n - test_size - val_size

    X_train = features_clean.iloc[:train_size]
    X_val = features_clean.iloc[train_size:train_size + val_size]
    X_test = features_clean.iloc[train_size + val_size:]

    y_train = target_clean.iloc[:train_size]
    y_val = target_clean.iloc[train_size:train_size + val_size]
    y_test = target_clean.iloc[train_size + val_size:]

    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Get feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance sorted descending
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).flatten()
    else:
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    return importance_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test feature preparation
    df = load_full_dataset("SPY")
    features = prepare_ml_features(df)
    target = create_target_variable(df)

    print(f"Features shape: {features.shape}")
    print(f"Target distribution:\n{target.value_counts()}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(features, target)
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
