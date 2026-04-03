"""Backtest module for strategy testing."""

from .ema_crossover import (
    BacktestResult,
    run_ema_crossover_backtest,
    calculate_metrics,
)

__all__ = [
    "BacktestResult",
    "run_ema_crossover_backtest",
    "calculate_metrics",
]
