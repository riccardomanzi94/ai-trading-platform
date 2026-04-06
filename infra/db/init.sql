-- AI Trading Platform - Database Schema
-- PostgreSQL with TimescaleDB extension

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create database (run as superuser)
-- CREATE DATABASE ai_trading;

-- OHLCV price data
CREATE TABLE IF NOT EXISTS prices (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    adj_close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (time, ticker)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('prices', 'time', if_not_exists => TRUE);

-- Create index for ticker lookups
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices (ticker, time DESC);

-- Technical features computed from price data
CREATE TABLE IF NOT EXISTS features (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    ema_12 DOUBLE PRECISION,
    ema_26 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    volatility_20 DOUBLE PRECISION,
    PRIMARY KEY (time, ticker)
);

SELECT create_hypertable('features', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_features_ticker ON features (ticker, time DESC);

-- Trading signals
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),
    strength DOUBLE PRECISION,
    price_at_signal DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, time)
);

SELECT create_hypertable('signals', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals (ticker, time DESC);

-- Risk-adjusted orders (after risk engine)
CREATE TABLE IF NOT EXISTS risk_orders (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    original_quantity INTEGER,
    adjusted_quantity INTEGER,
    position_size_pct DOUBLE PRECISION,
    risk_score DOUBLE PRECISION,
    approved BOOLEAN DEFAULT FALSE,
    rejection_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, time)
);

SELECT create_hypertable('risk_orders', 'time', if_not_exists => TRUE);

-- Paper trading executions
CREATE TABLE IF NOT EXISTS paper_executions (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION DEFAULT 0,
    slippage DOUBLE PRECISION DEFAULT 0,
    total_value DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, time)
);

SELECT create_hypertable('paper_executions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_executions_ticker ON paper_executions (ticker, time DESC);

-- Portfolio state (current holdings)
CREATE TABLE IF NOT EXISTS portfolio (
    ticker VARCHAR(10) PRIMARY KEY,
    quantity INTEGER NOT NULL DEFAULT 0,
    avg_cost DOUBLE PRECISION,
    current_value DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- Portfolio history for tracking performance
CREATE TABLE IF NOT EXISTS portfolio_history (
    time TIMESTAMPTZ NOT NULL,
    total_value DOUBLE PRECISION,
    cash DOUBLE PRECISION,
    positions_value DOUBLE PRECISION,
    daily_pnl DOUBLE PRECISION,
    cumulative_pnl DOUBLE PRECISION,
    PRIMARY KEY (time)
);

SELECT create_hypertable('portfolio_history', 'time', if_not_exists => TRUE);

-- Backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DOUBLE PRECISION,
    final_capital DOUBLE PRECISION,
    total_return DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    total_trades INTEGER,
    parameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Retention policy: keep detailed data for 2 years, then compress
SELECT add_retention_policy('prices', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('features', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('signals', INTERVAL '1 year', if_not_exists => TRUE);

-- Enable compression for older data
ALTER TABLE prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker'
);

SELECT add_compression_policy('prices', INTERVAL '30 days', if_not_exists => TRUE);

-- Macroeconomic data from FRED (Federal Reserve)
CREATE TABLE IF NOT EXISTS macro_data (
    time TIMESTAMPTZ NOT NULL,
    series_id VARCHAR(20) NOT NULL,
    series_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(50),
    frequency VARCHAR(20), -- daily, weekly, monthly, quarterly
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, series_id)
);

SELECT create_hypertable('macro_data', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_macro_series ON macro_data (series_id, time DESC);

-- Common FRED series for reference
-- DFF: Effective Federal Funds Rate (daily)
-- DGS10: 10-Year Treasury Constant Maturity Rate (daily)
-- DGS2: 2-Year Treasury Constant Maturity Rate (daily)
-- T10Y2Y: 10-Year minus 2-Year Treasury spread (daily)
-- CPIAUCSL: Consumer Price Index (monthly)
-- UNRATE: Unemployment Rate (monthly)
-- GDP: Gross Domestic Product (quarterly)
-- VIXCLS: CBOE Volatility Index (daily)
