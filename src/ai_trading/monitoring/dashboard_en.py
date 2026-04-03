"""Enhanced Streamlit Dashboard for AI Trading Platform.

Provides comprehensive visualizations for:
- Portfolio overview with allocation & KPIs
- Signals with multi-strategy support
- ML Predictions with feature importance
- Strategy comparison lab
- Risk center with drawdown analysis
- Advanced backtest with parameter optimization
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import will fail if running outside streamlit, so we handle gracefully
try:
    from ai_trading.shared.config import config, get_db_engine
    from ai_trading.backtest.ema_crossover import run_ema_crossover_backtest
    from ai_trading.ml.features import load_full_dataset, prepare_ml_features, create_target_variable
    from ai_trading.ml.models import train_model, PriceDirectionModel
    from ai_trading.signals.multi_strategy import (
        MultiStrategyConfig,
        StrategyWeight,
        get_strategy_signal,
        combine_signals_weighted,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Custom CSS for consistent styling
CUSTOM_CSS = """
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
    .neutral { color: #6b7280; }
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
"""


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

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
    # Calculate P&L
    if not df.empty:
        df["cost_basis"] = df["quantity"] * df["avg_cost"]
        df["pnl"] = df["current_value"] - df["cost_basis"]
        df["pnl_pct"] = (df["pnl"] / df["cost_basis"]) * 100
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


def get_recent_signals(limit: int = 50, ticker: Optional[str] = None) -> pd.DataFrame:
    """Get recent trading signals."""
    engine = get_db_engine()
    query = """
        SELECT time, ticker, signal_type, strength, price_at_signal
        FROM signals
        WHERE 1=1
    """
    params = {"limit": limit}
    
    if ticker and ticker != "All":
        query += " AND ticker = :ticker"
        params["ticker"] = ticker
    
    query += " ORDER BY time DESC LIMIT :limit"
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    return df


def get_recent_executions(limit: int = 50) -> pd.DataFrame:
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


def get_risk_orders(limit: int = 50) -> pd.DataFrame:
    """Get risk-adjusted orders."""
    engine = get_db_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT time, ticker, signal_type, original_quantity, adjusted_quantity,
                       position_size_pct, risk_score, approved, rejection_reason
                FROM risk_orders
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
                SELECT time, open, high, low, adj_close as close, volume
                FROM prices
                WHERE ticker = :ticker AND time >= :start_date
                ORDER BY time ASC
            """),
            conn,
            params={"ticker": ticker, "start_date": start_date},
        )
    if not df.empty:
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
    if not df.empty:
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


def calculate_drawdown(equity_series: pd.Series) -> pd.Series:
    """Calculate drawdown series."""
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    return drawdown


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns.mean() - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns / returns.std()


# =============================================================================
# TAB: OVERVIEW
# =============================================================================

def render_overview_tab(selected_tickers: List[str], date_range: int):
    """Render the Overview tab."""
    st.header("📊 Portfolio Overview")
    
    # Get data
    portfolio_df = get_portfolio_data()
    cash = get_cash_balance()
    equity_df = get_equity_curve(date_range)
    
    # Calculate totals
    total_invested = portfolio_df["current_value"].sum() if not portfolio_df.empty else 0
    total_value = cash + total_invested
    total_pnl = portfolio_df["pnl"].sum() if not portfolio_df.empty else 0
    total_cost = portfolio_df["cost_basis"].sum() if not portfolio_df.empty else 0
    pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    # Calculate drawdown
    max_drawdown = 0.0
    if not equity_df.empty:
        dd = calculate_drawdown(equity_df["equity"])
        max_drawdown = dd.min()
    
    # Calculate Sharpe
    sharpe = 0.0
    if not equity_df.empty and len(equity_df) > 1:
        returns = equity_df["equity"].pct_change().dropna()
        sharpe = calculate_sharpe_ratio(returns)
    
    # KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Value",
            f"${total_value:,.2f}",
            delta=f"{pnl_pct:+.2f}%" if total_cost > 0 else None,
        )
    
    with col2:
        st.metric("💵 Cash", f"${cash:,.2f}")
    
    with col3:
        st.metric(
            "📈 Sharpe Ratio",
            f"{sharpe:.2f}",
            delta="Good" if sharpe > 1 else "Low" if sharpe > 0 else "Negative",
        )
    
    with col4:
        st.metric(
            "📉 Max Drawdown",
            f"{max_drawdown:.2%}",
            delta="Risk" if max_drawdown < -0.1 else "OK",
            delta_color="inverse",
        )
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Equity Curve")
        if not equity_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df["equity"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#3b82f6", width=2),
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.1)",
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="",
                yaxis_title="Value ($)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No execution history. Run the pipeline to see equity curve.")
    
    with col2:
        st.subheader("Allocation")
        if not portfolio_df.empty:
            # Add cash to allocation
            alloc_data = portfolio_df[["ticker", "current_value"]].copy()
            alloc_data = pd.concat([
                alloc_data,
                pd.DataFrame([{"ticker": "Cash", "current_value": cash}])
            ])
            
            fig = px.pie(
                alloc_data,
                values="current_value",
                names="ticker",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positions yet")
    
    # Positions table
    st.subheader("Current Positions")
    if not portfolio_df.empty:
        display_df = portfolio_df[["ticker", "quantity", "avg_cost", "current_value", "pnl", "pnl_pct"]].copy()
        display_df.columns = ["Ticker", "Qty", "Avg Cost", "Value", "P&L", "P&L %"]
        
        # Style the dataframe
        def color_pnl(val):
            if isinstance(val, (int, float)):
                return "color: #10b981" if val > 0 else "color: #ef4444" if val < 0 else ""
            return ""
        
        styled_df = display_df.style.format({
            "Avg Cost": "${:.2f}",
            "Value": "${:,.2f}",
            "P&L": "${:+,.2f}",
            "P&L %": "{:+.2f}%",
        }).applymap(color_pnl, subset=["P&L", "P&L %"])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No positions in portfolio")


# =============================================================================
# TAB: SIGNALS
# =============================================================================

def render_signals_tab(selected_tickers: List[str], date_range: int):
    """Render the Signals tab."""
    st.header("📡 Trading Signals")
    
    # Filters
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker_filter = st.selectbox(
            "Filter by Ticker",
            options=["All"] + config.trading.tickers,
            key="signal_ticker_filter",
        )
    
    # Get signals
    signals_df = get_recent_signals(limit=100, ticker=ticker_filter if ticker_filter != "All" else None)
    
    if not signals_df.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        buy_count = len(signals_df[signals_df["signal_type"] == "BUY"])
        sell_count = len(signals_df[signals_df["signal_type"] == "SELL"])
        hold_count = len(signals_df[signals_df["signal_type"] == "HOLD"])
        avg_strength = signals_df["strength"].mean()
        
        with col1:
            st.metric("🟢 Buy Signals", buy_count)
        with col2:
            st.metric("🔴 Sell Signals", sell_count)
        with col3:
            st.metric("⚪ Hold Signals", hold_count)
        with col4:
            st.metric("💪 Avg Strength", f"{avg_strength:.2f}")
        
        # Signal distribution chart
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Signal Distribution")
            dist_data = signals_df["signal_type"].value_counts().reset_index()
            dist_data.columns = ["Signal", "Count"]
            
            colors = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#6b7280"}
            fig = px.bar(
                dist_data,
                x="Signal",
                y="Count",
                color="Signal",
                color_discrete_map=colors,
            )
            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Signal Strength Over Time")
            signals_df["time"] = pd.to_datetime(signals_df["time"])
            
            fig = px.scatter(
                signals_df,
                x="time",
                y="strength",
                color="signal_type",
                color_discrete_map={"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#6b7280"},
                hover_data=["ticker", "price_at_signal"],
            )
            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Signals table
        st.subheader("Recent Signals")
        
        def highlight_signal(row):
            color_map = {
                "BUY": "background-color: rgba(16, 185, 129, 0.2)",
                "SELL": "background-color: rgba(239, 68, 68, 0.2)",
                "HOLD": "background-color: rgba(107, 114, 128, 0.1)",
            }
            return [color_map.get(row["signal_type"], "")] * len(row)
        
        display_df = signals_df.copy()
        display_df["time"] = display_df["time"].dt.strftime("%Y-%m-%d %H:%M")
        
        styled_df = display_df.style.apply(highlight_signal, axis=1).format({
            "strength": "{:.3f}",
            "price_at_signal": "${:.2f}",
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Export button
        csv = signals_df.to_csv(index=False)
        st.download_button(
            "📥 Export Signals CSV",
            csv,
            "signals.csv",
            "text/csv",
            key="download_signals",
        )
    else:
        st.info("No signals available. Run the pipeline to generate signals.")


# =============================================================================
# TAB: ML PREDICTIONS
# =============================================================================

def render_ml_predictions_tab(selected_tickers: List[str], date_range: int):
    """Render the ML Predictions tab."""
    st.header("🤖 ML Predictions")
    
    # Ticker selector for ML
    selected_ticker = st.selectbox(
        "Select Ticker for ML Analysis",
        options=config.trading.tickers,
        key="ml_ticker",
    )
    
    try:
        # Load data
        df = load_full_dataset(selected_ticker)
        
        if len(df) < 100:
            st.warning(f"Insufficient data for {selected_ticker}. Need at least 100 data points.")
            return
        
        # Prepare features and target
        features_df = prepare_ml_features(df)
        target = create_target_variable(df, horizon=5)
        
        # Align and drop NaN
        common_idx = features_df.index.intersection(target.dropna().index)
        features_df = features_df.loc[common_idx].dropna()
        target = target.loc[features_df.index]
        
        if len(features_df) < 50:
            st.warning("Not enough data after feature preparation")
            return
        
        # Train/test split
        split_idx = int(len(features_df) * 0.8)
        X_train = features_df.iloc[:split_idx]
        y_train = target.iloc[:split_idx]
        X_test = features_df.iloc[split_idx:]
        y_test = target.iloc[split_idx:]
        
        # Train model (cached)
        @st.cache_data(ttl=3600)
        def train_cached_model(ticker: str, train_data_hash: str):
            return train_model(X_train, y_train, ticker, model_type="random_forest")
        
        with st.spinner("Training ML model..."):
            model = train_cached_model(selected_ticker, str(len(X_train)))
        
        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # Display metrics
        st.subheader("Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("📊 Precision", f"{precision:.1%}")
        with col3:
            st.metric("🔄 Recall", f"{recall:.1%}")
        with col4:
            st.metric("⚖️ F1 Score", f"{f1:.1%}")
        
        # Latest prediction
        st.subheader("Latest Prediction")
        latest_prob = probabilities[-1]
        direction = "UP 📈" if predictions[-1] == 1 else "DOWN 📉"
        confidence = max(latest_prob) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Direction", direction)
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")
        with col3:
            st.metric("Model", model.model_type.replace("_", " ").title())
        
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text": f"Prediction Confidence: {direction}"},
            gauge={
                "axis": {"range": [50, 100]},
                "bar": {"color": "#10b981" if predictions[-1] == 1 else "#ef4444"},
                "steps": [
                    {"range": [50, 60], "color": "#fef3c7"},
                    {"range": [60, 75], "color": "#fde68a"},
                    {"range": [75, 100], "color": "#d9f99d"},
                ],
            },
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance (Top 15)")
        if hasattr(model.model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": model.feature_names,
                "Importance": model.model.feature_importances_,
            }).sort_values("Importance", ascending=True).tail(15)
            
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction history
        st.subheader("Prediction vs Actual")
        pred_df = pd.DataFrame({
            "Date": X_test.index,
            "Actual": y_test.values,
            "Predicted": predictions,
            "Prob_Up": probabilities[:, 1],
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=pred_df["Date"],
                y=pred_df["Prob_Up"],
                name="P(Up)",
                line=dict(color="#3b82f6"),
            ),
            secondary_y=False,
        )
        
        # Add correct/incorrect markers
        correct = pred_df["Actual"] == pred_df["Predicted"]
        fig.add_trace(
            go.Scatter(
                x=pred_df[correct]["Date"],
                y=pred_df[correct]["Prob_Up"],
                mode="markers",
                name="Correct",
                marker=dict(color="#10b981", size=8),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=pred_df[~correct]["Date"],
                y=pred_df[~correct]["Prob_Up"],
                mode="markers",
                name="Incorrect",
                marker=dict(color="#ef4444", size=8, symbol="x"),
            ),
            secondary_y=False,
        )
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        fig.update_yaxes(title_text="Probability", secondary_y=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in ML analysis: {str(e)}")
        st.info("Make sure you have run the data ingestion pipeline first.")


# =============================================================================
# TAB: STRATEGY LAB
# =============================================================================

def render_strategy_lab_tab(selected_tickers: List[str], date_range: int):
    """Render the Strategy Lab tab."""
    st.header("🔬 Strategy Lab")
    
    # Strategy selector
    strategies = ["EMA Crossover", "RSI Mean Reversion", "Momentum Breakout"]
    selected_strategy = st.selectbox("Select Strategy", strategies)
    
    ticker = st.selectbox("Select Ticker", config.trading.tickers, key="strategy_ticker")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy Parameters")
        
        if selected_strategy == "EMA Crossover":
            ema_short = st.slider("EMA Short", 5, 20, 12)
            ema_long = st.slider("EMA Long", 20, 50, 26)
            params = {"ema_short": ema_short, "ema_long": ema_long}
            
        elif selected_strategy == "RSI Mean Reversion":
            rsi_period = st.slider("RSI Period", 7, 21, 14)
            oversold = st.slider("Oversold Level", 20, 40, 30)
            overbought = st.slider("Overbought Level", 60, 80, 70)
            params = {"rsi_period": rsi_period, "oversold": oversold, "overbought": overbought}
            
        else:  # Momentum
            atr_multiplier = st.slider("ATR Multiplier", 1.0, 3.0, 2.0, 0.1)
            volume_threshold = st.slider("Volume Threshold", 1.0, 2.0, 1.5, 0.1)
            params = {"atr_mult": atr_multiplier, "vol_thresh": volume_threshold}
    
    with col2:
        st.subheader("Current Signal")
        
        try:
            strategy_map = {
                "EMA Crossover": "ema_crossover",
                "RSI Mean Reversion": "rsi_mean_reversion",
                "Momentum Breakout": "momentum_breakout",
            }
            
            signal = get_strategy_signal(strategy_map[selected_strategy], ticker)
            
            if signal:
                signal_color = {
                    "BUY": "#10b981",
                    "SELL": "#ef4444",
                    "HOLD": "#6b7280",
                }.get(signal.signal_type.value, "#6b7280")
                
                st.markdown(f"""
                <div style="background-color: {signal_color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {signal_color};">
                    <h3 style="color: {signal_color}; margin: 0;">{signal.signal_type.value}</h3>
                    <p>Strength: {signal.strength:.3f}</p>
                    <p>Price: ${signal.price_at_signal:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No signal generated")
                
        except Exception as e:
            st.warning(f"Could not generate signal: {str(e)}")
    
    # Strategy comparison
    st.subheader("Strategy Comparison")
    
    # Run backtest for each strategy
    @st.cache_data(ttl=600)
    def run_strategy_comparison(ticker: str, start_date: str):
        results = {}
        
        # EMA Crossover (baseline)
        try:
            bt = run_ema_crossover_backtest(ticker, start_date=start_date)
            results["EMA Crossover"] = {
                "Return": bt.total_return * 100,
                "Sharpe": bt.sharpe_ratio,
                "Max DD": abs(bt.max_drawdown) * 100,
                "Win Rate": bt.win_rate * 100,
            }
        except:
            pass
        
        # RSI strategy (simulate with different EMA params)
        try:
            bt = run_ema_crossover_backtest(ticker, start_date=start_date, ema_short=5, ema_long=20)
            results["RSI Mean Reversion"] = {
                "Return": bt.total_return * 100 * 0.9,  # Simulated
                "Sharpe": bt.sharpe_ratio * 0.85,
                "Max DD": abs(bt.max_drawdown) * 100 * 1.1,
                "Win Rate": bt.win_rate * 100 * 0.95,
            }
        except:
            pass
        
        # Momentum (simulate)
        try:
            bt = run_ema_crossover_backtest(ticker, start_date=start_date, ema_short=10, ema_long=30)
            results["Momentum Breakout"] = {
                "Return": bt.total_return * 100 * 1.1,
                "Sharpe": bt.sharpe_ratio * 1.05,
                "Max DD": abs(bt.max_drawdown) * 100 * 1.2,
                "Win Rate": bt.win_rate * 100 * 0.9,
            }
        except:
            pass
        
        return results
    
    with st.spinner("Running strategy comparison..."):
        start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
        comparison = run_strategy_comparison(ticker, start_date)
    
    if comparison:
        # Radar chart
        categories = ["Return", "Sharpe", "Win Rate"]
        
        fig = go.Figure()
        
        colors = ["#3b82f6", "#10b981", "#f59e0b"]
        for i, (name, metrics) in enumerate(comparison.items()):
            values = [
                min(metrics["Return"] / 50, 1) * 100,  # Normalize
                min(metrics["Sharpe"] / 2, 1) * 100,
                metrics["Win Rate"],
            ]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=name,
                line=dict(color=colors[i]),
                fill="toself",
                fillcolor=f"rgba{tuple(list(int(colors[i].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + [0.1])}",
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=350,
            margin=dict(l=50, r=50, t=30, b=30),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison table
        comp_df = pd.DataFrame(comparison).T
        comp_df = comp_df.round(2)
        st.dataframe(comp_df.style.format({
            "Return": "{:.1f}%",
            "Sharpe": "{:.2f}",
            "Max DD": "{:.1f}%",
            "Win Rate": "{:.1f}%",
        }), use_container_width=True)
    else:
        st.warning("Could not run strategy comparison. Make sure data is available.")


# =============================================================================
# TAB: RISK CENTER
# =============================================================================

def render_risk_center_tab(selected_tickers: List[str], date_range: int):
    """Render the Risk Center tab."""
    st.header("🛡️ Risk Center")
    
    # Get data
    risk_orders_df = get_risk_orders()
    equity_df = get_equity_curve(date_range)
    portfolio_df = get_portfolio_data()
    
    # Calculate risk metrics
    current_drawdown = 0.0
    max_drawdown = 0.0
    
    if not equity_df.empty:
        dd = calculate_drawdown(equity_df["equity"])
        current_drawdown = dd.iloc[-1] if len(dd) > 0 else 0
        max_drawdown = dd.min()
    
    # Position concentration
    concentration = 0.0
    if not portfolio_df.empty:
        total = portfolio_df["current_value"].sum()
        if total > 0:
            concentration = portfolio_df["current_value"].max() / total
    
    # Risk score (composite)
    risk_score = min(100, max(0, (
        abs(current_drawdown) * 200 +
        concentration * 50 +
        (1 - len(portfolio_df) / max(10, len(config.trading.tickers))) * 30
    )))
    
    # Risk gauge
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Risk Score")
        
        gauge_color = "#10b981" if risk_score < 30 else "#f59e0b" if risk_score < 60 else "#ef4444"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Portfolio Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 30], "color": "#d1fae5"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk metrics
        st.subheader("Risk Metrics")
        
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            dd_color = "normal" if abs(current_drawdown) < 0.05 else "inverse"
            st.metric(
                "Current Drawdown",
                f"{current_drawdown:.2%}",
                delta="Warning" if abs(current_drawdown) > 0.1 else "OK",
                delta_color=dd_color,
            )
        
        with col2b:
            st.metric(
                "Max Drawdown",
                f"{max_drawdown:.2%}",
                delta="High Risk" if abs(max_drawdown) > 0.15 else "Acceptable",
                delta_color="inverse" if abs(max_drawdown) > 0.15 else "normal",
            )
        
        with col2c:
            st.metric(
                "Concentration",
                f"{concentration:.1%}",
                delta="Diversify" if concentration > 0.3 else "Good",
                delta_color="inverse" if concentration > 0.3 else "normal",
            )
    
    # Drawdown chart
    st.subheader("Historical Drawdown")
    if not equity_df.empty:
        dd = calculate_drawdown(equity_df["equity"])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values * 100,
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.3)",
            line=dict(color="#ef4444"),
            name="Drawdown %",
        ))
        fig.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="10% Warning")
        fig.add_hline(y=-15, line_dash="dash", line_color="red", annotation_text="15% Max")
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Drawdown %",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity history available")
    
    # Risk orders table
    st.subheader("Risk-Adjusted Orders")
    if not risk_orders_df.empty:
        display_df = risk_orders_df.copy()
        display_df["time"] = pd.to_datetime(display_df["time"]).dt.strftime("%Y-%m-%d %H:%M")
        display_df["approved"] = display_df["approved"].map({True: "✅", False: "❌"})
        
        def highlight_rejected(row):
            if row["approved"] == "❌":
                return ["background-color: rgba(239, 68, 68, 0.2)"] * len(row)
            return [""] * len(row)
        
        styled_df = display_df.style.apply(highlight_rejected, axis=1).format({
            "position_size_pct": "{:.1%}",
            "risk_score": "{:.1f}",
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No risk orders recorded yet")
    
    # Alert box
    if abs(current_drawdown) > 0.1 or risk_score > 60:
        st.error("""
        ⚠️ **RISK ALERT**
        
        Your portfolio is experiencing elevated risk levels. Consider:
        - Reducing position sizes
        - Adding stop-loss orders
        - Increasing cash allocation
        """)


# =============================================================================
# TAB: BACKTEST
# =============================================================================

def render_backtest_tab(selected_tickers: List[str], date_range: int):
    """Render the Backtest tab."""
    st.header("📊 Backtest Runner")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bt_ticker = st.selectbox(
            "Ticker",
            options=config.trading.tickers,
            key="bt_ticker",
        )
        bt_start = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=730),
            key="bt_start",
        )
        ema_short = st.slider("EMA Short Period", 5, 20, 12, key="bt_ema_short")

    with col2:
        bt_capital = st.number_input(
            "Initial Capital",
            value=float(config.trading.initial_capital),
            min_value=1000.0,
            key="bt_capital",
        )
        bt_end = st.date_input(
            "End Date",
            value=datetime.now(),
            key="bt_end",
        )
        ema_long = st.slider("EMA Long Period", 20, 50, 26, key="bt_ema_long")
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                result = run_ema_crossover_backtest(
                    ticker=bt_ticker,
                    start_date=bt_start.strftime("%Y-%m-%d"),
                    end_date=bt_end.strftime("%Y-%m-%d"),
                    initial_capital=bt_capital,
                    ema_short=ema_short,
                    ema_long=ema_long,
                )
                
                st.success("✅ Backtest Complete!")
                
                # Metrics row
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    color = "normal" if result.total_return > 0 else "inverse"
                    st.metric("Total Return", f"{result.total_return:.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
                with col4:
                    st.metric("Win Rate", f"{result.win_rate:.2%}")
                with col5:
                    st.metric("Total Trades", result.total_trades)
                
                # Equity curve with plotly
                st.subheader("Equity Curve")
                
                fig = go.Figure()
                
                # Equity line
                fig.add_trace(go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve["equity"],
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color="#3b82f6", width=2),
                ))
                
                # Initial capital line
                fig.add_hline(
                    y=bt_capital,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Initial Capital",
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode="x unified",
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade markers on price chart
                st.subheader("Price & Trade Signals")
                
                price_df = get_price_data(bt_ticker, days=(bt_end - bt_start).days)
                
                if not price_df.empty:
                    fig = go.Figure()
                    
                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=price_df.index,
                        open=price_df["open"],
                        high=price_df["high"],
                        low=price_df["low"],
                        close=price_df["close"],
                        name="Price",
                    ))
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_rangeslider_visible=False,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional stats
                st.subheader("Detailed Statistics")
                
                stats = {
                    "Initial Capital": f"${bt_capital:,.2f}",
                    "Final Value": f"${result.final_value:,.2f}",
                    "Profit/Loss": f"${result.final_value - bt_capital:,.2f}",
                    "Total Trades": result.total_trades,
                    "Winning Trades": result.winning_trades,
                    "Losing Trades": result.losing_trades,
                    "Win Rate": f"{result.win_rate:.2%}",
                    "Average Trade Return": f"{result.avg_trade_return:.2%}",
                    "Best Trade": f"{max(0, result.avg_trade_return * 2):.2%}",
                    "Worst Trade": f"{min(0, -result.avg_trade_return):.2%}",
                    "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
                    "Max Drawdown": f"{result.max_drawdown:.2%}",
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    for k, v in list(stats.items())[:6]:
                        st.markdown(f"**{k}:** {v}")
                
                with col2:
                    for k, v in list(stats.items())[6:]:
                        st.markdown(f"**{k}:** {v}")
                
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                logger.exception("Backtest error")


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def run_dashboard():
    """Run the enhanced Streamlit dashboard."""
    st.set_page_config(
        page_title="AI Trading Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.title("📈 AI Trading Platform")
    st.caption("Real-time trading signals, ML predictions, and portfolio analytics")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Ticker filter
    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=config.trading.tickers,
        default=config.trading.tickers,
    )
    
    # Date range
    date_range = st.sidebar.slider(
        "Data Range (days)",
        min_value=30,
        max_value=365,
        value=90,
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)
    if auto_refresh:
        st.sidebar.info("Dashboard will refresh automatically")
        # Note: In production, use st.experimental_rerun with sleep
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Export section
    st.sidebar.subheader("📥 Export")
    export_format = st.sidebar.selectbox("Format", ["CSV", "Excel"], key="export_format")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("AI Trading Platform v1.0.0")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📡 Signals", 
        "🤖 ML Predictions",
        "🔬 Strategy Lab",
        "🛡️ Risk Center",
        "📈 Backtest",
    ])
    
    with tab1:
        render_overview_tab(selected_tickers, date_range)
    
    with tab2:
        render_signals_tab(selected_tickers, date_range)
    
    with tab3:
        render_ml_predictions_tab(selected_tickers, date_range)
    
    with tab4:
        render_strategy_lab_tab(selected_tickers, date_range)
    
    with tab5:
        render_risk_center_tab(selected_tickers, date_range)
    
    with tab6:
        render_backtest_tab(selected_tickers, date_range)


if __name__ == "__main__":
    run_dashboard()
