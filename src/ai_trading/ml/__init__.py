"""Machine Learning module for AI Trading Platform."""

from .features import (
    prepare_ml_features,
    create_target_variable,
    split_train_test,
)
from .models import (
    PriceDirectionModel,
    train_model,
    predict_direction,
    evaluate_model,
)
from .optimizer import (
    optimize_strategy_params,
    OptimizationResult,
)

__all__ = [
    # Features
    "prepare_ml_features",
    "create_target_variable",
    "split_train_test",
    # Models
    "PriceDirectionModel",
    "train_model",
    "predict_direction",
    "evaluate_model",
    # Optimizer
    "optimize_strategy_params",
    "OptimizationResult",
]
