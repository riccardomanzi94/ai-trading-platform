"""Multi-Strategy Combiner.

Combines signals from multiple strategies with configurable weights:
- EMA Crossover
- RSI Mean Reversion
- Momentum Breakout

Produces ensemble signals with aggregated strength.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from ai_trading.shared.config import config
from ai_trading.signals.generate_signals import (
    Signal,
    SignalType,
    generate_signals_for_ticker,
)
from ai_trading.signals.rsi_mean_reversion import generate_rsi_signal
from ai_trading.signals.momentum_breakout import generate_momentum_signal

logger = logging.getLogger(__name__)


class CombineMethod(str, Enum):
    """Methods for combining multiple strategy signals."""
    
    UNANIMOUS = "unanimous"  # All strategies must agree
    MAJORITY = "majority"  # Majority vote
    WEIGHTED = "weighted"  # Weighted average
    ANY = "any"  # Any signal triggers


@dataclass
class StrategyWeight:
    """Weight configuration for a strategy."""
    
    name: str
    weight: float
    enabled: bool = True


@dataclass
class MultiStrategyConfig:
    """Configuration for multi-strategy combiner."""
    
    strategies: list[StrategyWeight] = field(default_factory=lambda: [
        StrategyWeight("ema_crossover", 0.4, True),
        StrategyWeight("rsi_mean_reversion", 0.3, True),
        StrategyWeight("momentum_breakout", 0.3, True),
    ])
    combine_method: CombineMethod = CombineMethod.WEIGHTED
    min_strength: float = 0.3  # Minimum combined strength to emit signal
    conflict_resolution: str = "abstain"  # "abstain", "stronger", "recent"


def get_strategy_signal(strategy_name: str, ticker: str) -> Optional[Signal]:
    """Get signal from a specific strategy.

    Args:
        strategy_name: Name of the strategy
        ticker: Stock/ETF ticker symbol

    Returns:
        Signal from the strategy, or None
    """
    if strategy_name == "ema_crossover":
        signals = generate_signals_for_ticker(ticker, save_to_db=False)
        return signals[0] if signals else None
    elif strategy_name == "rsi_mean_reversion":
        return generate_rsi_signal(ticker)
    elif strategy_name == "momentum_breakout":
        return generate_momentum_signal(ticker)
    else:
        logger.warning(f"Unknown strategy: {strategy_name}")
        return None


def combine_signals_weighted(
    signals: list[tuple[Signal, float]],
    min_strength: float = 0.3,
) -> Optional[Signal]:
    """Combine signals using weighted average.

    Args:
        signals: List of (signal, weight) tuples
        min_strength: Minimum strength threshold

    Returns:
        Combined signal or None if conflicting/weak
    """
    if not signals:
        return None

    # Separate buy and sell signals
    buy_signals = [(s, w) for s, w in signals if s.signal_type == SignalType.BUY]
    sell_signals = [(s, w) for s, w in signals if s.signal_type == SignalType.SELL]

    # Calculate weighted scores
    buy_score = sum(s.strength * w for s, w in buy_signals)
    sell_score = sum(s.strength * w for s, w in sell_signals)
    total_weight = sum(w for _, w in signals)

    if total_weight == 0:
        return None

    # Determine direction
    if buy_score > sell_score and buy_score / total_weight >= min_strength:
        # Use the most recent signal as template
        base_signal = buy_signals[0][0]
        combined_strength = buy_score / total_weight
        
        return Signal(
            time=base_signal.time,
            ticker=base_signal.ticker,
            signal_type=SignalType.BUY,
            strength=min(1.0, combined_strength),
            price_at_signal=base_signal.price_at_signal,
        )
    elif sell_score > buy_score and sell_score / total_weight >= min_strength:
        base_signal = sell_signals[0][0]
        combined_strength = sell_score / total_weight

        return Signal(
            time=base_signal.time,
            ticker=base_signal.ticker,
            signal_type=SignalType.SELL,
            strength=min(1.0, combined_strength),
            price_at_signal=base_signal.price_at_signal,
        )

    return None


def combine_signals_majority(
    signals: list[tuple[Signal, float]],
) -> Optional[Signal]:
    """Combine signals using majority vote.

    Args:
        signals: List of (signal, weight) tuples

    Returns:
        Signal if majority agrees, None otherwise
    """
    if not signals:
        return None

    buy_count = sum(1 for s, _ in signals if s.signal_type == SignalType.BUY)
    sell_count = sum(1 for s, _ in signals if s.signal_type == SignalType.SELL)
    total = len(signals)

    if buy_count > total / 2:
        buy_signals = [s for s, _ in signals if s.signal_type == SignalType.BUY]
        avg_strength = sum(s.strength for s in buy_signals) / len(buy_signals)
        base = buy_signals[0]
        return Signal(
            time=base.time,
            ticker=base.ticker,
            signal_type=SignalType.BUY,
            strength=avg_strength,
            price_at_signal=base.price_at_signal,
        )
    elif sell_count > total / 2:
        sell_signals = [s for s, _ in signals if s.signal_type == SignalType.SELL]
        avg_strength = sum(s.strength for s in sell_signals) / len(sell_signals)
        base = sell_signals[0]
        return Signal(
            time=base.time,
            ticker=base.ticker,
            signal_type=SignalType.SELL,
            strength=avg_strength,
            price_at_signal=base.price_at_signal,
        )

    return None


def combine_signals_unanimous(
    signals: list[tuple[Signal, float]],
) -> Optional[Signal]:
    """Combine signals requiring unanimous agreement.

    Args:
        signals: List of (signal, weight) tuples

    Returns:
        Signal if all agree, None otherwise
    """
    if not signals:
        return None

    signal_types = set(s.signal_type for s, _ in signals)
    
    if len(signal_types) != 1:
        return None  # Not unanimous

    # All agree
    all_signals = [s for s, _ in signals]
    avg_strength = sum(s.strength for s in all_signals) / len(all_signals)
    base = all_signals[0]

    return Signal(
        time=base.time,
        ticker=base.ticker,
        signal_type=base.signal_type,
        strength=avg_strength,
        price_at_signal=base.price_at_signal,
    )


def generate_combined_signal(
    ticker: str,
    strategy_config: Optional[MultiStrategyConfig] = None,
) -> Optional[Signal]:
    """Generate combined signal from multiple strategies.

    Args:
        ticker: Stock/ETF ticker symbol
        strategy_config: Multi-strategy configuration

    Returns:
        Combined signal or None
    """
    if strategy_config is None:
        strategy_config = MultiStrategyConfig()

    # Collect signals from enabled strategies
    signals_with_weights = []
    
    for strategy in strategy_config.strategies:
        if not strategy.enabled:
            continue
            
        try:
            signal = get_strategy_signal(strategy.name, ticker)
            if signal:
                signals_with_weights.append((signal, strategy.weight))
                logger.debug(
                    f"{ticker} {strategy.name}: {signal.signal_type.value} "
                    f"strength={signal.strength:.2f}"
                )
        except Exception as e:
            logger.warning(f"Strategy {strategy.name} failed for {ticker}: {e}")

    if not signals_with_weights:
        return None

    # Combine based on method
    if strategy_config.combine_method == CombineMethod.WEIGHTED:
        return combine_signals_weighted(
            signals_with_weights, 
            strategy_config.min_strength
        )
    elif strategy_config.combine_method == CombineMethod.MAJORITY:
        return combine_signals_majority(signals_with_weights)
    elif strategy_config.combine_method == CombineMethod.UNANIMOUS:
        return combine_signals_unanimous(signals_with_weights)
    elif strategy_config.combine_method == CombineMethod.ANY:
        # Return strongest signal
        strongest = max(signals_with_weights, key=lambda x: x[0].strength)
        return strongest[0]

    return None


def generate_combined_signals_all(
    tickers: Optional[list[str]] = None,
    strategy_config: Optional[MultiStrategyConfig] = None,
) -> dict[str, list[Signal]]:
    """Generate combined signals for all tickers.

    Args:
        tickers: List of ticker symbols
        strategy_config: Multi-strategy configuration

    Returns:
        Dict mapping ticker to list of signals
    """
    if tickers is None:
        tickers = config.trading.tickers

    results = {}
    for ticker in tickers:
        try:
            signal = generate_combined_signal(ticker, strategy_config)
            results[ticker] = [signal] if signal else []
        except Exception as e:
            logger.error(f"Failed to generate combined signal for {ticker}: {e}")
            results[ticker] = []

    # Summary
    total_signals = sum(len(s) for s in results.values())
    logger.info(f"Generated {total_signals} combined signals from multi-strategy")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with default config
    signals = generate_combined_signals_all()
    for ticker, sigs in signals.items():
        for s in sigs:
            print(f"{ticker}: {s.signal_type.value} @ ${s.price_at_signal:.2f} (strength: {s.strength:.2f})")
