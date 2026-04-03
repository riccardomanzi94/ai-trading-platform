"""Streamlit Dashboard for AI Trading Platform.

Provides visualizations for:
- Portfolio performance
- Recent signals and executions
- Technical indicators
- Backtest results
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
from sqlalchemy import text

# Import will fail if running outside streamlit, so we handle gracefully
try:
    from ai_trading.shared.config import config, get_db_engine
    from ai_trading.backtest.ema_crossover import run_ema_crossover_backtest
except ImportError:
    pass

logger = logging.getLogger(__name__)


def get_portfolio_data() -> pd.DataFrame:
    """Get current portfolio positions."""
    engine = get_db_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT ticker, quantity, avg_cost, current_value
                FROM portfolio
                WHERE ticker != '_CASH' AND quantity > 0
                ORDER BY current_value DESC
            """),
            conn,
        )
    return df


def get_cash_balance() -> float:
    """Get current cash balance."""
    engine = get_db_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT current_value FROM portfolio WHERE ticker = '_CASH'")
        )
        row = result.fetchone()
        return row[0] if row else config.trading.initial_capital


def get_recent_signals(limit: int = 20) -> pd.DataFrame:
    """Get recent trading signals."""
    engine = get_db_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT time, ticker, signal_type, strength, price_at_signal
                FROM signals
                ORDER BY time DESC
                LIMIT :limit
            """),
            conn,
            params={"limit": limit},
        )
    return df


def get_recent_executions(limit: int = 20) -> pd.DataFrame:
    """Get recent trade executions."""
    engine = get_db_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT time, ticker, side, quantity, price, commission, total_value
                FROM paper_executions
                ORDER BY time DESC
                LIMIT :limit
            """),
            conn,
            params={"limit": limit},
        )
    return df


def get_price_data(ticker: str, days: int = 90) -> pd.DataFrame:
    """Get price data for a ticker."""
    engine = get_db_engine()
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT time, adj_close as price
                FROM prices
                WHERE ticker = :ticker AND time >= :start_date
                ORDER BY time ASC
            """),
            conn,
            params={"ticker": ticker, "start_date": start_date},
        )
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df


def get_features_data(ticker: str, days: int = 90) -> pd.DataFrame:
    """Get feature data for a ticker."""
    engine = get_db_engine()
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT f.time, p.adj_close as price, f.ema_12, f.ema_26, 
                       f.rsi_14, f.atr_14, f.volatility_20
                FROM features f
                JOIN prices p ON f.time = p.time AND f.ticker = p.ticker
                WHERE f.ticker = :ticker AND f.time >= :start_date
                ORDER BY f.time ASC
            """),
            conn,
            params={"ticker": ticker, "start_date": start_date},
        )
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df


def get_equity_curve(days: int = 365) -> pd.DataFrame:
    """Calculate equity curve from executions."""
    engine = get_db_engine()
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT time, 
                       SUM(CASE WHEN side = 'BUY' THEN -total_value ELSE total_value END) 
                       OVER (ORDER BY time) as cumulative_pnl
                FROM paper_executions
                WHERE time >= :start_date
                ORDER BY time ASC
            """),
            conn,
            params={"start_date": start_date},
        )

    if df.empty:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"])
    df["equity"] = config.trading.initial_capital + df["cumulative_pnl"]
    df = df.set_index("time")
    return df[["equity"]]


def run_dashboard():
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="AI Trading Platform",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 AI Trading Platform Dashboard")

    # Sidebar
    st.sidebar.header("Settings")
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        options=config.trading.tickers,
        index=0,
    )
    chart_days = st.sidebar.slider("Chart Days", 30, 365, 90)

    # Main layout
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Portfolio", "Signals & Executions", "Technical Analysis", "Backtest"]
    )

    # Tab 1: Portfolio
    with tab1:
        st.header("Portfolio Overview")

        col1, col2, col3 = st.columns(3)

        # Get portfolio data
        portfolio_df = get_portfolio_data()
        cash = get_cash_balance()
        total_invested = portfolio_df["current_value"].sum() if not portfolio_df.empty else 0
        total_value = cash + total_invested

        with col1:
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")
        with col2:
            st.metric("Cash", f"${cash:,.2f}")
        with col3:
            st.metric("Invested", f"${total_invested:,.2f}")

        if not portfolio_df.empty:
            st.subheader("Current Positions")
            st.dataframe(portfolio_df, use_container_width=True)

            # Portfolio allocation chart
            st.subheader("Portfolio Allocation")
            allocation = portfolio_df.set_index("ticker")["current_value"]
            allocation["Cash"] = cash
            st.bar_chart(allocation)

        # Equity curve
        equity_df = get_equity_curve(chart_days)
        if not equity_df.empty:
            st.subheader("Equity Curve")
            st.line_chart(equity_df)

    # Tab 2: Signals & Executions
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Recent Signals")
            signals_df = get_recent_signals()
            if not signals_df.empty:
                # Color-code signals
                def highlight_signal(row):
                    if row["signal_type"] == "BUY":
                        return ["background-color: #90EE90"] * len(row)
                    elif row["signal_type"] == "SELL":
                        return ["background-color: #FFB6C1"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    signals_df.style.apply(highlight_signal, axis=1),
                    use_container_width=True,
                )
            else:
                st.info("No signals yet")

        with col2:
            st.subheader("Recent Executions")
            executions_df = get_recent_executions()
            if not executions_df.empty:
                st.dataframe(executions_df, use_container_width=True)
            else:
                st.info("No executions yet")

    # Tab 3: Technical Analysis
    with tab3:
        st.header(f"Technical Analysis: {selected_ticker}")

        features_df = get_features_data(selected_ticker, chart_days)

        if not features_df.empty:
            # Price with EMAs
            st.subheader("Price & EMAs")
            price_ema_df = features_df[["price", "ema_12", "ema_26"]]
            st.line_chart(price_ema_df)

            # RSI
            st.subheader("RSI (14)")
            rsi_df = features_df[["rsi_14"]]
            st.line_chart(rsi_df)
            st.caption("Overbought > 70, Oversold < 30")

            # Volatility
            st.subheader("Volatility (20-day)")
            vol_df = features_df[["volatility_20"]]
            st.line_chart(vol_df)

            # ATR
            st.subheader("ATR (14)")
            atr_df = features_df[["atr_14"]]
            st.line_chart(atr_df)
        else:
            st.warning(f"No feature data available for {selected_ticker}")

    # Tab 4: Backtest
    with tab4:
        st.header("Backtest Runner")

        col1, col2 = st.columns(2)

        with col1:
            bt_ticker = st.selectbox(
                "Backtest Ticker",
                options=config.trading.tickers,
                key="bt_ticker",
            )
            bt_start = st.date_input(
                "Start Date",
                value=datetime.strptime(config.backtest.start_date, "%Y-%m-%d"),
            )

        with col2:
            bt_capital = st.number_input(
                "Initial Capital",
                value=config.trading.initial_capital,
                min_value=1000.0,
            )
            bt_end = st.date_input("End Date", value=datetime.now())

        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    result = run_ema_crossover_backtest(
                        ticker=bt_ticker,
                        start_date=bt_start.strftime("%Y-%m-%d"),
                        end_date=bt_end.strftime("%Y-%m-%d"),
                        initial_capital=bt_capital,
                    )

                    # Display results
                    st.success("Backtest complete!")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{result.total_return:.2%}")
                    with col2:
                        st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
                    with col4:
                        st.metric("Win Rate", f"{result.win_rate:.2%}")

                    # Equity curve
                    st.subheader("Equity Curve")
                    st.line_chart(result.equity_curve)

                    # Trade statistics
                    st.subheader("Trade Statistics")
                    stats_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Initial Capital",
                                "Final Value",
                                "Total Trades",
                                "Winning Trades",
                                "Losing Trades",
                                "Avg Trade Return",
                            ],
                            "Value": [
                                f"${result.initial_capital:,.2f}",
                                f"${result.final_value:,.2f}",
                                result.total_trades,
                                result.winning_trades,
                                result.losing_trades,
                                f"{result.avg_trade_return:.2%}",
                            ],
                        }
                    )
                    st.dataframe(stats_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("AI Trading Platform v0.1.0")


if __name__ == "__main__":
    run_dashboard()
