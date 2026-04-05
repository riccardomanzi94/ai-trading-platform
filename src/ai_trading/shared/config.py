"""Shared configuration for AI Trading Platform."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ai_trading"
        )
    )
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class TradingConfig:
    """Trading parameters configuration."""

    tickers: list[str] = field(
        default_factory=lambda: os.getenv(
            "TICKERS",
            "SPY,QQQ,IWM,VOO,VTI,XLF,XLK,XLE,XLU,XLI,XLP,XLV,XLY,TLT,BND,AGG,LQD,VEA,VWO,GLD,USO,AAPL,MSFT,GOOGL,TSLA"
        ).split(",")
    )
    initial_capital: float = field(
        default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "100000.0"))
    )
    max_position_size: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    )
    max_portfolio_risk: float = field(
        default_factory=lambda: float(os.getenv("MAX_PORTFOLIO_RISK", "0.02"))
    )
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005  # 0.05% slippage


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""

    ema_short: int = 12
    ema_long: int = 26
    rsi_period: int = 14
    atr_period: int = 14
    volatility_period: int = 20


@dataclass
class BacktestConfig:
    """Backtesting parameters."""

    start_date: str = "2020-01-01"
    end_date: Optional[str] = None  # None means today
    benchmark_ticker: str = "SPY"


@dataclass
class Config:
    """Main configuration container."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


# Global config instance
config = Config()


def get_db_engine():
    """Create SQLAlchemy engine with connection pooling."""
    from sqlalchemy import create_engine

    return create_engine(
        config.database.url,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow,
        pool_pre_ping=True,
    )


def get_db_connection():
    """Get a database connection from the pool."""
    engine = get_db_engine()
    return engine.connect()
