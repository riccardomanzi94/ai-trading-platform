"""Signals module for trading signal generation."""

from .generate_signals import (
    generate_ema_crossover_signal,
    generate_signals_for_ticker,
    generate_all_signals,
    get_recent_signals,
    Signal,
    SignalType,
)
from .rsi_mean_reversion import (
    generate_rsi_signal,
    generate_rsi_signals_all,
    RSIStrategyConfig,
)
from .momentum_breakout import (
    generate_momentum_signal,
    generate_momentum_signals_all,
    MomentumConfig,
)
from .multi_strategy import (
    generate_combined_signal,
    generate_combined_signals_all,
    MultiStrategyConfig,
    CombineMethod,
    StrategyWeight,
)

__all__ = [
    # EMA Crossover
    "generate_ema_crossover_signal",
    "generate_signals_for_ticker",
    "generate_all_signals",
    "get_recent_signals",
    "Signal",
    "SignalType",
    # RSI Mean Reversion
    "generate_rsi_signal",
    "generate_rsi_signals_all",
    "RSIStrategyConfig",
    # Momentum Breakout
    "generate_momentum_signal",
    "generate_momentum_signals_all",
    "MomentumConfig",
    # Multi-Strategy
    "generate_combined_signal",
    "generate_combined_signals_all",
    "MultiStrategyConfig",
    "CombineMethod",
    "StrategyWeight",
]
