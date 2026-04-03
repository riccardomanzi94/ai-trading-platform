"""ML Models for price direction prediction.

Implements various models:
- Random Forest Classifier
- Gradient Boosting (XGBoost-like with sklearn)
- Simple Neural Network (optional)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from ai_trading.shared.config import config

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics from model evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class PriceDirectionModel:
    """Wrapper for price direction prediction models."""

    model: Any
    scaler: Optional[StandardScaler]
    feature_names: List[str]
    model_type: str
    ticker: str
    trained_at: datetime
    metrics: Optional[ModelMetrics] = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict direction (0 = down, 1 = up).

        Args:
            features: Feature DataFrame

        Returns:
            Array of predictions
        """
        X = features[self.feature_names].values
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probabilities.

        Args:
            features: Feature DataFrame

        Returns:
            Array of probabilities [prob_down, prob_up]
        """
        X = features[self.feature_names].values
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to file.

        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PriceDirectionModel":
        """Load model from file.

        Args:
            path: Path to the saved model

        Returns:
            Loaded PriceDirectionModel
        """
        with open(path, "rb") as f:
            return pickle.load(f)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ticker: str,
    model_type: str = "random_forest",
    scale_features: bool = True,
    **model_params,
) -> PriceDirectionModel:
    """Train a price direction prediction model.

    Args:
        X_train: Training features
        y_train: Training target
        ticker: Ticker symbol
        model_type: Type of model ("random_forest", "gradient_boosting", "logistic")
        scale_features: Whether to scale features
        **model_params: Additional model parameters

    Returns:
        Trained PriceDirectionModel
    """
    logger.info(f"Training {model_type} model for {ticker}")

    feature_names = list(X_train.columns)
    X = X_train.values.copy()

    # Scale features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Create model
    if model_type == "random_forest":
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(model_params)
        model = RandomForestClassifier(**default_params)

    elif model_type == "gradient_boosting":
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_split": 5,
            "random_state": 42,
        }
        default_params.update(model_params)
        model = GradientBoostingClassifier(**default_params)

    elif model_type == "logistic":
        default_params = {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
        }
        default_params.update(model_params)
        model = LogisticRegression(**default_params)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Train
    model.fit(X, y_train)

    return PriceDirectionModel(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        model_type=model_type,
        ticker=ticker,
        trained_at=datetime.now(),
    )


def evaluate_model(
    model: PriceDirectionModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelMetrics:
    """Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        ModelMetrics with evaluation results
    """
    predictions = model.predict(X_test)
    probas = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    try:
        auc_roc = roc_auc_score(y_test, probas[:, 1])
    except Exception:
        auc_roc = None

    cm = confusion_matrix(y_test, predictions)

    metrics = ModelMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        confusion_matrix=cm,
    )

    model.metrics = metrics

    auc_str = f"{auc_roc:.3f}" if auc_roc else "N/A"
    logger.info(
        f"Model evaluation: accuracy={accuracy:.3f}, precision={precision:.3f}, "
        f"recall={recall:.3f}, f1={f1:.3f}, auc_roc={auc_str}"
    )

    return metrics


def predict_direction(
    model: PriceDirectionModel,
    features: pd.DataFrame,
    return_proba: bool = False,
) -> Tuple[int, float]:
    """Predict direction for the most recent data point.

    Args:
        model: Trained model
        features: Feature DataFrame (uses last row)
        return_proba: Whether to return probability

    Returns:
        Tuple of (prediction, confidence)
    """
    last_row = features.iloc[[-1]]
    prediction = model.predict(last_row)[0]
    proba = model.predict_proba(last_row)[0]

    confidence = proba[1] if prediction == 1 else proba[0]

    return int(prediction), float(confidence)


def create_ml_signal(
    model: PriceDirectionModel,
    features: pd.DataFrame,
    ticker: str,
    min_confidence: float = 0.6,
) -> Optional[Dict]:
    """Create trading signal from ML prediction.

    Args:
        model: Trained model
        features: Feature DataFrame
        ticker: Ticker symbol
        min_confidence: Minimum confidence to generate signal

    Returns:
        Signal dict or None
    """
    from ai_trading.signals.generate_signals import Signal, SignalType

    direction, confidence = predict_direction(model, features)

    if confidence < min_confidence:
        return None

    # Get current price
    from ai_trading.feature_store.build_features import load_prices_from_db
    prices = load_prices_from_db(ticker)
    if prices.empty:
        return None
    current_price = prices["adj_close"].iloc[-1]

    signal_type = SignalType.BUY if direction == 1 else SignalType.SELL

    return Signal(
        time=datetime.now(),
        ticker=ticker,
        signal_type=signal_type,
        strength=confidence,
        price_at_signal=current_price,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from ai_trading.ml.features import (
        load_full_dataset,
        prepare_ml_features,
        create_target_variable,
        split_train_test,
        get_feature_importance,
    )

    # Load and prepare data
    df = load_full_dataset("SPY")
    features = prepare_ml_features(df)
    target = create_target_variable(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(features, target)

    # Train model
    model = train_model(X_train, y_train, "SPY", model_type="random_forest")

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1: {metrics.f1:.3f}")
    print(f"  AUC-ROC: {metrics.auc_roc:.3f}")

    # Feature importance
    importance = get_feature_importance(model.model, model.feature_names)
    print(f"\nTop 10 Features:")
    print(importance.head(10).to_string(index=False))

    # Predict
    direction, confidence = predict_direction(model, features)
    print(f"\nLatest prediction: {'UP' if direction == 1 else 'DOWN'} (confidence: {confidence:.2%})")
