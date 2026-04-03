"""Hyperparameter Optimization using Optuna.

Optimizes:
- Strategy parameters (EMA periods, RSI thresholds, etc.)
- ML model hyperparameters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any

from ai_trading.shared.config import config
from ai_trading.backtest.ema_crossover import run_ema_crossover_backtest, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""

    best_params: Dict[str, Any]
    best_value: float
    optimization_history: List[Dict]
    n_trials: int
    ticker: str
    metric_optimized: str
    started_at: datetime
    completed_at: datetime


def _check_optuna():
    """Check if optuna is available."""
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for optimization. "
            "Install with: pip install optuna"
        )


def optimize_ema_strategy(
    ticker: str,
    n_trials: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    metric: str = "sharpe_ratio",
) -> OptimizationResult:
    """Optimize EMA crossover strategy parameters.

    Args:
        ticker: Stock/ETF ticker symbol
        n_trials: Number of optimization trials
        start_date: Backtest start date
        end_date: Backtest end date
        metric: Metric to optimize ("sharpe_ratio", "total_return", "win_rate")

    Returns:
        OptimizationResult with best parameters
    """
    _check_optuna()

    started_at = datetime.now()
    history = []

    def objective(trial: Trial) -> float:
        # Sample parameters
        ema_short = trial.suggest_int("ema_short", 5, 20)
        ema_long = trial.suggest_int("ema_long", 20, 50)

        # Ensure short < long
        if ema_short >= ema_long:
            return float("-inf")

        try:
            result = run_ema_crossover_backtest(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                ema_short=ema_short,
                ema_long=ema_long,
            )

            if metric == "sharpe_ratio":
                value = result.sharpe_ratio
            elif metric == "total_return":
                value = result.total_return
            elif metric == "win_rate":
                value = result.win_rate
            else:
                value = result.sharpe_ratio

            history.append({
                "trial": trial.number,
                "ema_short": ema_short,
                "ema_long": ema_long,
                "value": value,
            })

            return value

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("-inf")

    # Create study and optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    completed_at = datetime.now()

    return OptimizationResult(
        best_params=study.best_params,
        best_value=study.best_value,
        optimization_history=history,
        n_trials=n_trials,
        ticker=ticker,
        metric_optimized=metric,
        started_at=started_at,
        completed_at=completed_at,
    )


def optimize_rsi_strategy(
    ticker: str,
    n_trials: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    metric: str = "sharpe_ratio",
) -> OptimizationResult:
    """Optimize RSI mean reversion strategy parameters.

    Args:
        ticker: Stock/ETF ticker symbol
        n_trials: Number of optimization trials
        start_date: Backtest start date
        end_date: Backtest end date
        metric: Metric to optimize

    Returns:
        OptimizationResult with best parameters
    """
    _check_optuna()

    from ai_trading.signals.rsi_mean_reversion import RSIStrategyConfig

    started_at = datetime.now()
    history = []

    def objective(trial: Trial) -> float:
        oversold = trial.suggest_int("oversold_threshold", 20, 35)
        overbought = trial.suggest_int("overbought_threshold", 65, 80)
        min_change = trial.suggest_float("min_rsi_change", 1.0, 5.0)

        # Run backtest with these parameters
        # Note: This requires a custom backtest function for RSI strategy
        # For now, we'll use a simplified evaluation

        try:
            # Import here to avoid circular imports
            from ai_trading.ml.features import load_full_dataset, prepare_ml_features
            from ai_trading.feature_store.build_features import compute_rsi

            df = load_full_dataset(ticker)

            # Simulate RSI strategy
            df["rsi"] = df["rsi_14"]
            df["signal"] = 0

            # Generate signals
            for i in range(1, len(df)):
                prev_rsi = df["rsi"].iloc[i - 1]
                curr_rsi = df["rsi"].iloc[i]
                rsi_change = curr_rsi - prev_rsi

                if prev_rsi < oversold and rsi_change > min_change:
                    df.iloc[i, df.columns.get_loc("signal")] = 1  # BUY
                elif prev_rsi > overbought and rsi_change < -min_change:
                    df.iloc[i, df.columns.get_loc("signal")] = -1  # SELL

            # Calculate simple returns
            df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
            df["strategy_return"] = df["position"].shift(1) * df["adj_close"].pct_change()
            
            total_return = (1 + df["strategy_return"].fillna(0)).prod() - 1
            sharpe = df["strategy_return"].mean() / (df["strategy_return"].std() + 1e-8) * np.sqrt(252)

            if metric == "sharpe_ratio":
                value = sharpe
            elif metric == "total_return":
                value = total_return
            else:
                value = sharpe

            history.append({
                "trial": trial.number,
                "oversold_threshold": oversold,
                "overbought_threshold": overbought,
                "min_rsi_change": min_change,
                "value": value,
            })

            return value if not np.isnan(value) else float("-inf")

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("-inf")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    completed_at = datetime.now()

    return OptimizationResult(
        best_params=study.best_params,
        best_value=study.best_value,
        optimization_history=history,
        n_trials=n_trials,
        ticker=ticker,
        metric_optimized=metric,
        started_at=started_at,
        completed_at=completed_at,
    )


def optimize_ml_model(
    ticker: str,
    model_type: str = "random_forest",
    n_trials: int = 30,
    metric: str = "f1",
) -> OptimizationResult:
    """Optimize ML model hyperparameters.

    Args:
        ticker: Stock/ETF ticker symbol
        model_type: Type of model to optimize
        n_trials: Number of optimization trials
        metric: Metric to optimize ("accuracy", "f1", "auc_roc")

    Returns:
        OptimizationResult with best parameters
    """
    _check_optuna()

    from ai_trading.ml.features import (
        load_full_dataset,
        prepare_ml_features,
        create_target_variable,
        split_train_test,
    )
    from ai_trading.ml.models import train_model, evaluate_model

    started_at = datetime.now()
    history = []

    # Load data once
    df = load_full_dataset(ticker)
    features = prepare_ml_features(df)
    target = create_target_variable(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(features, target)

    def objective(trial: Trial) -> float:
        if model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
        elif model_type == "gradient_boosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }
        else:
            params = {}

        try:
            model = train_model(X_train, y_train, ticker, model_type, **params)
            metrics = evaluate_model(model, X_val, y_val)

            if metric == "accuracy":
                value = metrics.accuracy
            elif metric == "f1":
                value = metrics.f1
            elif metric == "auc_roc":
                value = metrics.auc_roc or 0.5
            else:
                value = metrics.f1

            history.append({
                "trial": trial.number,
                **params,
                "value": value,
            })

            return value

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    completed_at = datetime.now()

    return OptimizationResult(
        best_params=study.best_params,
        best_value=study.best_value,
        optimization_history=history,
        n_trials=n_trials,
        ticker=ticker,
        metric_optimized=metric,
        started_at=started_at,
        completed_at=completed_at,
    )


def optimize_strategy_params(
    ticker: str,
    strategy: str = "ema",
    n_trials: int = 50,
    **kwargs,
) -> OptimizationResult:
    """Generic strategy parameter optimization.

    Args:
        ticker: Stock/ETF ticker symbol
        strategy: Strategy name ("ema", "rsi", "ml")
        n_trials: Number of optimization trials
        **kwargs: Additional arguments for specific optimizers

    Returns:
        OptimizationResult with best parameters
    """
    if strategy == "ema":
        return optimize_ema_strategy(ticker, n_trials, **kwargs)
    elif strategy == "rsi":
        return optimize_rsi_strategy(ticker, n_trials, **kwargs)
    elif strategy == "ml":
        return optimize_ml_model(ticker, n_trials=n_trials, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test EMA optimization
    print("Optimizing EMA strategy for SPY...")
    result = optimize_ema_strategy("SPY", n_trials=20)
    print(f"\nBest params: {result.best_params}")
    print(f"Best Sharpe: {result.best_value:.3f}")
