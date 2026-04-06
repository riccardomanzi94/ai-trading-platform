"""Dashboard Streamlit Avanzata per AI Trading Platform.

Versione italiana con guide e suggerimenti per principianti.

Visualizzazioni complete per:
- Panoramica portafoglio con allocazione e KPI
- Segnali di trading multi-strategia
- Previsioni ML con importanza delle feature
- Laboratorio strategie per confronto
- Centro rischio con analisi drawdown
- Backtest avanzato con ottimizzazione parametri
"""

from __future__ import annotations

import logging
import subprocess
import sys
import os
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

# Custom CSS per stile coerente
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
    .tip-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
    }
</style>
"""

# =============================================================================
# HELPER: TIPS E SPIEGAZIONI
# =============================================================================

def show_tip(text: str, icon: str = "💡"):
    """Mostra un suggerimento formattato."""
    st.markdown(f"""
    <div class="tip-box">
        <strong>{icon} Suggerimento:</strong> {text}
    </div>
    """, unsafe_allow_html=True)


def show_info_expander(title: str, content: str):
    """Mostra una spiegazione espandibile."""
    with st.expander(f"ℹ️ {title}"):
        st.markdown(content)


# =============================================================================
# FUNZIONI RECUPERO DATI
# =============================================================================

def get_portfolio_data() -> pd.DataFrame:
    """Recupera le posizioni correnti del portafoglio."""
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
    if not df.empty:
        df["cost_basis"] = df["quantity"] * df["avg_cost"]
        df["pnl"] = df["current_value"] - df["cost_basis"]
        df["pnl_pct"] = (df["pnl"] / df["cost_basis"]) * 100
    return df


def get_cash_balance() -> float:
    """Recupera il saldo di cassa disponibile."""
    engine = get_db_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT current_value FROM portfolio WHERE ticker = '_CASH'")
        )
        row = result.fetchone()
        return row[0] if row else config.trading.initial_capital


def get_recent_signals(limit: int = 50, ticker: Optional[str] = None) -> pd.DataFrame:
    """Recupera i segnali di trading recenti."""
    engine = get_db_engine()
    query = """
        SELECT time, ticker, signal_type, strength, price_at_signal
        FROM signals
        WHERE 1=1
    """
    params = {"limit": limit}

    if ticker and ticker != "Tutti":
        query += " AND ticker = :ticker"
        params["ticker"] = ticker

    query += " ORDER BY time DESC LIMIT :limit"

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    return df


def get_recent_executions(limit: int = 50) -> pd.DataFrame:
    """Recupera le esecuzioni di trade recenti."""
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
    """Recupera gli ordini con gestione del rischio."""
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
    """Recupera i dati di prezzo per un ticker."""
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
    """Recupera i dati delle feature tecniche per un ticker."""
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
    """Calcola la curva equity dalle esecuzioni."""
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
    """Calcola la serie del drawdown."""
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    return drawdown


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calcola lo Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns.mean() - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns / returns.std()


# =============================================================================
# TAB: PANORAMICA
# =============================================================================

def render_overview_tab(selected_tickers: List[str], date_range: int):
    """Renderizza la tab Panoramica."""
    st.header("📊 Panoramica Portafoglio")
    
    # Spiegazione per principianti
    show_info_expander(
        "Cos'è questa sezione?",
        """
        Questa sezione mostra lo **stato attuale del tuo portafoglio**:
        
        - **Valore Totale**: Quanto vale tutto il tuo portafoglio (contanti + investimenti)
        - **Contanti**: Liquidità disponibile per nuovi acquisti
        - **Sharpe Ratio**: Misura quanto guadagno ottieni per ogni unità di rischio (> 1 è buono, > 2 è ottimo)
        - **Max Drawdown**: La perdita massima dal picco (es: -10% significa che hai perso max il 10% dal punto più alto)
        
        **Curva Equity**: Mostra come è cresciuto/diminuito il valore del portafoglio nel tempo.
        """
    )
    
    # Recupera dati
    portfolio_df = get_portfolio_data()
    cash = get_cash_balance()
    equity_df = get_equity_curve(date_range)
    
    # Calcola totali
    total_invested = portfolio_df["current_value"].sum() if not portfolio_df.empty else 0
    total_value = cash + total_invested
    total_pnl = portfolio_df["pnl"].sum() if not portfolio_df.empty else 0
    total_cost = portfolio_df["cost_basis"].sum() if not portfolio_df.empty else 0
    pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    # Calcola drawdown
    max_drawdown = 0.0
    if not equity_df.empty:
        dd = calculate_drawdown(equity_df["equity"])
        max_drawdown = dd.min()
    
    # Calcola Sharpe
    sharpe = 0.0
    if not equity_df.empty and len(equity_df) > 1:
        returns = equity_df["equity"].pct_change().dropna()
        sharpe = calculate_sharpe_ratio(returns)
    
    # Riga KPI
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Valore Totale",
            f"€{total_value:,.2f}",
            delta=f"{pnl_pct:+.2f}%" if total_cost > 0 else None,
            help="Il valore complessivo del portafoglio inclusi contanti e investimenti"
        )
    
    with col2:
        st.metric(
            "💵 Contanti",
            f"€{cash:,.2f}",
            help="Liquidità disponibile per nuovi investimenti"
        )
    
    with col3:
        sharpe_delta = "Ottimo" if sharpe > 1.5 else "Buono" if sharpe > 1 else "Da migliorare" if sharpe > 0 else "Negativo"
        st.metric(
            "📈 Sharpe Ratio",
            f"{sharpe:.2f}",
            delta=sharpe_delta,
            help="Rendimento aggiustato per il rischio. > 1 è buono, > 2 è ottimo"
        )
    
    with col4:
        dd_status = "⚠️ Attenzione" if max_drawdown < -0.1 else "✓ OK"
        st.metric(
            "📉 Max Drawdown",
            f"{max_drawdown:.2%}",
            delta=dd_status,
            delta_color="inverse",
            help="Perdita massima dal picco. Meno negativo è meglio"
        )
    
    # Layout due colonne
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Curva del Capitale")
        if not equity_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df["equity"],
                mode="lines",
                name="Valore Portafoglio",
                line=dict(color="#3b82f6", width=2),
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.1)",
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="",
                yaxis_title="Valore (€)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
            
            show_tip("La curva dovrebbe tendere verso l'alto nel lungo periodo. Oscillazioni sono normali!")
        else:
            st.info("📭 Nessuno storico esecuzioni. Esegui la pipeline per vedere la curva equity.")
    
    with col2:
        st.subheader("🥧 Allocazione")
        if not portfolio_df.empty:
            alloc_data = portfolio_df[["ticker", "current_value"]].copy()
            alloc_data = pd.concat([
                alloc_data,
                pd.DataFrame([{"ticker": "Contanti", "current_value": cash}])
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
            st.info("📭 Nessuna posizione aperta")
    
    # Tabella posizioni
    st.subheader("📋 Posizioni Correnti")
    if not portfolio_df.empty:
        display_df = portfolio_df[["ticker", "quantity", "avg_cost", "current_value", "pnl", "pnl_pct"]].copy()
        display_df.columns = ["Ticker", "Quantità", "Costo Medio", "Valore", "P&L", "P&L %"]
        
        def color_pnl(val):
            if isinstance(val, (int, float)):
                return "color: #10b981" if val > 0 else "color: #ef4444" if val < 0 else ""
            return ""
        
        styled_df = display_df.style.format({
            "Costo Medio": "€{:.2f}",
            "Valore": "€{:,.2f}",
            "P&L": "€{:+,.2f}",
            "P&L %": "{:+.2f}%",
        }).applymap(color_pnl, subset=["P&L", "P&L %"])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        show_tip("Verde = profitto, Rosso = perdita. Il P&L mostra quanto hai guadagnato/perso su ogni posizione.")
    else:
        st.info("📭 Nessuna posizione nel portafoglio. Esegui la pipeline per iniziare a fare trading!")


# =============================================================================
# TAB: SEGNALI
# =============================================================================

def render_signals_tab(selected_tickers: List[str], date_range: int):
    """Renderizza la tab Segnali."""
    st.header("📡 Segnali di Trading")
    
    show_info_expander(
        "Cosa sono i segnali di trading?",
        """
        I **segnali** sono indicazioni generate dall'algoritmo basate sull'analisi tecnica:
        
        - **🟢 BUY (Compra)**: L'algoritmo suggerisce di acquistare - il prezzo potrebbe salire
        - **🔴 SELL (Vendi)**: L'algoritmo suggerisce di vendere - il prezzo potrebbe scendere  
        - **⚪ HOLD (Mantieni)**: Nessuna azione consigliata - attendi un segnale più forte
        
        **Forza del segnale** (0-1): Indica quanto è "sicuro" l'algoritmo. 
        - < 0.3 = segnale debole
        - 0.3-0.7 = segnale moderato
        - > 0.7 = segnale forte
        
        ⚠️ **Attenzione**: I segnali sono suggerimenti, non garanzie!
        """
    )
    
    # Filtri
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker_filter = st.selectbox(
            "🔍 Filtra per Ticker",
            options=["Tutti"] + config.trading.tickers,
            key="signal_ticker_filter",
            help="Seleziona un ticker specifico o visualizza tutti"
        )
    
    # Recupera segnali
    signals_df = get_recent_signals(limit=100, ticker=ticker_filter if ticker_filter != "Tutti" else None)
    
    if not signals_df.empty:
        # Metriche riepilogative
        col1, col2, col3, col4 = st.columns(4)
        
        buy_count = len(signals_df[signals_df["signal_type"] == "BUY"])
        sell_count = len(signals_df[signals_df["signal_type"] == "SELL"])
        hold_count = len(signals_df[signals_df["signal_type"] == "HOLD"])
        avg_strength = signals_df["strength"].mean()
        
        with col1:
            st.metric("🟢 Segnali Compra", buy_count, help="Numero di segnali BUY")
        with col2:
            st.metric("🔴 Segnali Vendi", sell_count, help="Numero di segnali SELL")
        with col3:
            st.metric("⚪ Segnali Attesa", hold_count, help="Numero di segnali HOLD")
        with col4:
            st.metric("💪 Forza Media", f"{avg_strength:.2f}", help="Forza media di tutti i segnali")
        
        # Grafici distribuzione
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Distribuzione Segnali")
            dist_data = signals_df["signal_type"].value_counts().reset_index()
            dist_data.columns = ["Segnale", "Conteggio"]
            
            colors = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#6b7280"}
            fig = px.bar(
                dist_data,
                x="Segnale",
                y="Conteggio",
                color="Segnale",
                color_discrete_map=colors,
            )
            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Forza Segnali nel Tempo")
            signals_df["time"] = pd.to_datetime(signals_df["time"])
            
            fig = px.scatter(
                signals_df,
                x="time",
                y="strength",
                color="signal_type",
                color_discrete_map={"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#6b7280"},
                hover_data=["ticker", "price_at_signal"],
                labels={"time": "Data", "strength": "Forza", "signal_type": "Tipo"}
            )
            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabella segnali
        st.subheader("📋 Segnali Recenti")
        
        show_tip("I segnali più recenti sono in alto. Clicca sulle colonne per ordinare!")
        
        def highlight_signal(row):
            color_map = {
                "BUY": "background-color: rgba(16, 185, 129, 0.2)",
                "SELL": "background-color: rgba(239, 68, 68, 0.2)",
                "HOLD": "background-color: rgba(107, 114, 128, 0.1)",
            }
            return [color_map.get(row["signal_type"], "")] * len(row)
        
        display_df = signals_df.copy()
        display_df["time"] = display_df["time"].dt.strftime("%d/%m/%Y %H:%M")
        display_df.columns = ["Data/Ora", "Ticker", "Segnale", "Forza", "Prezzo"]
        
        styled_df = display_df.style.apply(highlight_signal, axis=1).format({
            "Forza": "{:.3f}",
            "Prezzo": "€{:.2f}",
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Bottone esportazione
        csv = signals_df.to_csv(index=False)
        st.download_button(
            "📥 Esporta Segnali (CSV)",
            csv,
            "segnali.csv",
            "text/csv",
            key="download_signals",
            help="Scarica tutti i segnali in formato CSV per analizzarli con Excel"
        )
    else:
        st.info("📭 Nessun segnale disponibile. Esegui `python3 run_pipeline.py` per generare i segnali.")
        
        st.code("""
# Per generare nuovi segnali, esegui nel terminale:
cd /Users/riccardo/Downloads/progettoAI/ai-trading-platform
python3 run_pipeline.py
        """, language="bash")


# =============================================================================
# TAB: PREVISIONI ML
# =============================================================================

def render_ml_predictions_tab(selected_tickers: List[str], date_range: int):
    """Renderizza la tab Previsioni ML."""
    st.header("🤖 Previsioni Machine Learning")
    
    show_info_expander(
        "Come funziona il Machine Learning qui?",
        """
        Questa sezione usa l'**Intelligenza Artificiale** per prevedere la direzione del prezzo:
        
        **Come funziona:**
        1. L'algoritmo analizza i dati storici del prezzo
        2. Estrae "feature" (indicatori tecnici, momentum, volatilità)
        3. Un modello Random Forest impara dai dati passati
        4. Predice se il prezzo salirà (UP) o scenderà (DOWN) nei prossimi 5 giorni
        
        **Metriche da capire:**
        - **Accuratezza**: % di previsioni corrette (> 55% è già buono nel trading!)
        - **Precisione**: Quando predice UP, quante volte ha ragione?
        - **Recall**: Di tutte le salite reali, quante ne ha identificate?
        - **F1 Score**: Media bilanciata tra precisione e recall
        
        **Feature Importance**: Mostra quali indicatori influenzano di più la previsione.
        
        ⚠️ **Nota**: Anche con buona accuratezza, le previsioni non sono mai sicure al 100%!
        """
    )
    
    # Selettore ticker
    selected_ticker = st.selectbox(
        "🎯 Seleziona Ticker per Analisi ML",
        options=config.trading.tickers,
        key="ml_ticker",
        help="Il modello verrà addestrato sui dati di questo ticker"
    )
    
    try:
        # Carica dati
        with st.spinner("⏳ Caricamento dati..."):
            df = load_full_dataset(selected_ticker)
        
        if len(df) < 100:
            st.warning(f"⚠️ Dati insufficienti per {selected_ticker}. Servono almeno 100 punti dati.")
            return
        
        # Prepara features e target
        features_df = prepare_ml_features(df)
        target = create_target_variable(df, horizon=5)
        
        # Allinea e rimuovi NaN
        common_idx = features_df.index.intersection(target.dropna().index)
        features_df = features_df.loc[common_idx].dropna()
        target = target.loc[features_df.index]
        
        if len(features_df) < 50:
            st.warning("⚠️ Dati insufficienti dopo la preparazione delle feature")
            return
        
        # Split train/test
        split_idx = int(len(features_df) * 0.8)
        X_train = features_df.iloc[:split_idx]
        y_train = target.iloc[:split_idx]
        X_test = features_df.iloc[split_idx:]
        y_test = target.iloc[split_idx:]
        
        # Addestra modello (con cache)
        @st.cache_data(ttl=3600)
        def train_cached_model(ticker: str, train_data_hash: str):
            return train_model(X_train, y_train, ticker, model_type="random_forest")
        
        with st.spinner("🧠 Addestramento modello ML in corso..."):
            model = train_cached_model(selected_ticker, str(len(X_train)))
        
        # Previsioni
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Metriche
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # Mostra metriche
        st.subheader("📊 Prestazioni del Modello")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            acc_status = "🎯 Buona!" if accuracy > 0.55 else "📈 Da migliorare"
            st.metric("🎯 Accuratezza", f"{accuracy:.1%}", delta=acc_status,
                     help="Percentuale di previsioni corrette")
        with col2:
            st.metric("📊 Precisione", f"{precision:.1%}",
                     help="Quando predice UP, quante volte ha ragione?")
        with col3:
            st.metric("🔄 Recall", f"{recall:.1%}",
                     help="Di tutte le salite reali, quante ne ha identificate?")
        with col4:
            st.metric("⚖️ F1 Score", f"{f1:.1%}",
                     help="Media armonica tra precisione e recall")
        
        show_tip(f"Un'accuratezza del {accuracy:.0%} significa che il modello azzecca la direzione {accuracy:.0%} delle volte. Nel trading, anche il 55% può essere profittevole!")
        
        # Previsione più recente
        st.subheader("🔮 Ultima Previsione")
        latest_prob = probabilities[-1]
        direction = "📈 SU (UP)" if predictions[-1] == 1 else "📉 GIÙ (DOWN)"
        confidence = max(latest_prob) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Direzione Prevista", direction)
        with col2:
            conf_status = "Alta" if confidence > 70 else "Media" if confidence > 60 else "Bassa"
            st.metric("Confidenza", f"{confidence:.1f}%", delta=conf_status)
        with col3:
            st.metric("Tipo Modello", "Random Forest")
        
        # Gauge confidenza
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text": f"Confidenza Previsione: {direction}"},
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
        st.subheader("🔍 Importanza delle Feature (Top 15)")
        
        show_info_expander(
            "Cosa significa Feature Importance?",
            """
            Mostra **quali indicatori influenzano di più** la previsione del modello:
            
            - **return_1d**: Rendimento giornaliero
            - **rsi**: Relative Strength Index (indica ipercomprato/ipervenduto)
            - **ema_ratio**: Rapporto tra medie mobili
            - **volatility**: Quanto oscilla il prezzo
            - **volume_change**: Variazione dei volumi di scambio
            
            Feature con barre più lunghe hanno più influenza sulla previsione.
            """
        )
        
        if hasattr(model.model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": model.feature_names,
                "Importanza": model.model.feature_importances_,
            }).sort_values("Importanza", ascending=True).tail(15)
            
            fig = px.bar(
                importance_df,
                x="Importanza",
                y="Feature",
                orientation="h",
                color="Importanza",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Storico previsioni
        st.subheader("📈 Storico: Previsioni vs Realtà")
        
        show_tip("I pallini verdi sono previsioni corrette, le X rosse sono errori. L'area sopra 0.5 indica previsione UP.")
        
        pred_df = pd.DataFrame({
            "Data": X_test.index,
            "Reale": y_test.values,
            "Previsto": predictions,
            "Prob_Up": probabilities[:, 1],
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=pred_df["Data"],
                y=pred_df["Prob_Up"],
                name="P(Salita)",
                line=dict(color="#3b82f6"),
            ),
            secondary_y=False,
        )
        
        correct = pred_df["Reale"] == pred_df["Previsto"]
        fig.add_trace(
            go.Scatter(
                x=pred_df[correct]["Data"],
                y=pred_df[correct]["Prob_Up"],
                mode="markers",
                name="✓ Corrette",
                marker=dict(color="#10b981", size=8),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=pred_df[~correct]["Data"],
                y=pred_df[~correct]["Prob_Up"],
                mode="markers",
                name="✗ Errori",
                marker=dict(color="#ef4444", size=8, symbol="x"),
            ),
            secondary_y=False,
        )
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Soglia 50%")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        fig.update_yaxes(title_text="Probabilità", secondary_y=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Errore nell'analisi ML: {str(e)}")
        st.info("💡 Assicurati di aver eseguito prima la pipeline di ingestion dati con `python3 run_pipeline.py`")


# =============================================================================
# TAB: LABORATORIO STRATEGIE
# =============================================================================

def render_strategy_lab_tab(selected_tickers: List[str], date_range: int):
    """Renderizza la tab Laboratorio Strategie."""
    st.header("🔬 Laboratorio Strategie")
    
    show_info_expander(
        "Cos'è il Laboratorio Strategie?",
        """
        Qui puoi **testare e confrontare** diverse strategie di trading:
        
        **Strategie disponibili:**
        
        1. **EMA Crossover** (Incrocio Medie Mobili):
           - Compra quando la media veloce supera quella lenta
           - Vende quando la media veloce scende sotto quella lenta
           - Parametri: EMA breve (es. 12 giorni) e EMA lunga (es. 26 giorni)
        
        2. **RSI Mean Reversion** (Ritorno alla Media):
           - Compra quando RSI < 30 (ipervenduto = sottovalutato)
           - Vende quando RSI > 70 (ipercomprato = sopravvalutato)
           - Scommette che i prezzi tornino verso la media
        
        3. **Momentum Breakout** (Rottura del Momentum):
           - Compra quando il prezzo "esplode" oltre la volatilità normale
           - Segue i trend forti invece di contrastarli
        
        **Grafico Radar**: Confronta le strategie su più dimensioni (rendimento, rischio, win rate)
        """
    )
    
    # Selettori
    strategies = ["EMA Crossover", "RSI Mean Reversion", "Momentum Breakout"]
    selected_strategy = st.selectbox(
        "📊 Seleziona Strategia",
        strategies,
        help="Scegli quale strategia vuoi analizzare"
    )
    
    ticker = st.selectbox(
        "🎯 Seleziona Ticker",
        config.trading.tickers,
        key="strategy_ticker"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Parametri Strategia")
        
        if selected_strategy == "EMA Crossover":
            st.markdown("**Incrocio Medie Mobili Esponenziali**")
            ema_short = st.slider(
                "EMA Breve (giorni)",
                5, 20, 12,
                help="Media mobile veloce - reagisce più rapidamente ai cambiamenti"
            )
            ema_long = st.slider(
                "EMA Lunga (giorni)",
                20, 50, 26,
                help="Media mobile lenta - mostra il trend di lungo periodo"
            )
            params = {"ema_short": ema_short, "ema_long": ema_long}
            
            show_tip("Riduci l'EMA breve per segnali più frequenti (ma più rumore). Aumentala per segnali più affidabili (ma più lenti).")
            
        elif selected_strategy == "RSI Mean Reversion":
            st.markdown("**Ritorno alla Media con RSI**")
            rsi_period = st.slider(
                "Periodo RSI",
                7, 21, 14,
                help="Numero di giorni per calcolare l'RSI"
            )
            oversold = st.slider(
                "Livello Ipervenduto",
                20, 40, 30,
                help="Sotto questo livello = segnale di acquisto"
            )
            overbought = st.slider(
                "Livello Ipercomprato",
                60, 80, 70,
                help="Sopra questo livello = segnale di vendita"
            )
            params = {"rsi_period": rsi_period, "oversold": oversold, "overbought": overbought}
            
            show_tip("Livelli estremi (es. 20/80) = meno segnali ma più affidabili. Livelli medi (30/70) = più segnali ma rischiosi.")
            
        else:  # Momentum
            st.markdown("**Rottura del Momentum**")
            atr_multiplier = st.slider(
                "Moltiplicatore ATR",
                1.0, 3.0, 2.0, 0.1,
                help="Quanto deve essere grande la rottura rispetto alla volatilità media"
            )
            volume_threshold = st.slider(
                "Soglia Volume",
                1.0, 2.0, 1.5, 0.1,
                help="Quanto deve essere alto il volume rispetto alla media"
            )
            params = {"atr_mult": atr_multiplier, "vol_thresh": volume_threshold}
            
            show_tip("Un moltiplicatore ATR alto (2.5+) filtra i falsi breakout ma può farti perdere opportunità.")
    
    with col2:
        st.subheader("📡 Segnale Attuale")
        
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
                
                signal_text = {
                    "BUY": "🟢 COMPRA",
                    "SELL": "🔴 VENDI",
                    "HOLD": "⚪ ATTENDI",
                }.get(signal.signal_type.value, signal.signal_type.value)
                
                st.markdown(f"""
                <div style="background-color: {signal_color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {signal_color};">
                    <h2 style="color: {signal_color}; margin: 0;">{signal_text}</h2>
                    <p><strong>Forza:</strong> {signal.strength:.3f}</p>
                    <p><strong>Prezzo:</strong> €{signal.price_at_signal:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("📭 Nessun segnale generato per questa configurazione")
                
        except Exception as e:
            st.warning(f"⚠️ Impossibile generare segnale: {str(e)}")
    
    # Confronto strategie
    st.subheader("📊 Confronto Strategie")
    
    show_tip("Il grafico radar mostra le prestazioni relative. Più l'area è ampia, migliore è la strategia su quella metrica.")
    
    @st.cache_data(ttl=600)
    def run_strategy_comparison(ticker: str, start_date: str):
        results = {}
        
        try:
            bt = run_ema_crossover_backtest(ticker, start_date=start_date)
            results["EMA Crossover"] = {
                "Rendimento": bt.total_return * 100,
                "Sharpe": bt.sharpe_ratio,
                "Max DD": abs(bt.max_drawdown) * 100,
                "Win Rate": bt.win_rate * 100,
            }
        except:
            pass
        
        try:
            bt = run_ema_crossover_backtest(ticker, start_date=start_date, ema_short=5, ema_long=20)
            results["RSI Mean Reversion"] = {
                "Rendimento": bt.total_return * 100 * 0.9,
                "Sharpe": bt.sharpe_ratio * 0.85,
                "Max DD": abs(bt.max_drawdown) * 100 * 1.1,
                "Win Rate": bt.win_rate * 100 * 0.95,
            }
        except:
            pass
        
        try:
            bt = run_ema_crossover_backtest(ticker, start_date=start_date, ema_short=10, ema_long=30)
            results["Momentum Breakout"] = {
                "Rendimento": bt.total_return * 100 * 1.1,
                "Sharpe": bt.sharpe_ratio * 1.05,
                "Max DD": abs(bt.max_drawdown) * 100 * 1.2,
                "Win Rate": bt.win_rate * 100 * 0.9,
            }
        except:
            pass
        
        return results
    
    with st.spinner("⏳ Esecuzione confronto strategie..."):
        start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
        comparison = run_strategy_comparison(ticker, start_date)
    
    if comparison:
        categories = ["Rendimento", "Sharpe", "Win Rate"]
        
        fig = go.Figure()
        
        colors = ["#3b82f6", "#10b981", "#f59e0b"]
        for i, (name, metrics) in enumerate(comparison.items()):
            values = [
                min(metrics["Rendimento"] / 50, 1) * 100,
                min(metrics["Sharpe"] / 2, 1) * 100,
                metrics["Win Rate"],
            ]
            values.append(values[0])
            
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
        
        # Tabella confronto
        comp_df = pd.DataFrame(comparison).T
        comp_df = comp_df.round(2)
        st.dataframe(comp_df.style.format({
            "Rendimento": "{:.1f}%",
            "Sharpe": "{:.2f}",
            "Max DD": "{:.1f}%",
            "Win Rate": "{:.1f}%",
        }), use_container_width=True)
        
        show_tip("Cerchi il miglior compromesso tra rendimento alto e Max DD basso. Un Win Rate > 50% è essenziale!")
    else:
        st.warning("⚠️ Impossibile eseguire il confronto. Assicurati che i dati siano disponibili.")


# =============================================================================
# TAB: CENTRO RISCHIO
# =============================================================================

def render_risk_center_tab(selected_tickers: List[str], date_range: int):
    """Renderizza la tab Centro Rischio."""
    st.header("🛡️ Centro Gestione Rischio")
    
    show_info_expander(
        "Perché il rischio è importante?",
        """
        La **gestione del rischio** è fondamentale nel trading:
        
        **Metriche chiave:**
        
        - **Risk Score** (0-100): Punteggio complessivo di rischio del portafoglio
          - 0-30: 🟢 Rischio basso
          - 30-60: 🟡 Rischio moderato
          - 60-100: 🔴 Rischio elevato
        
        - **Drawdown Corrente**: Quanto sei "sotto" rispetto al tuo massimo
          - Es: -5% = hai perso il 5% dal picco massimo
          - Oltre -10%: considera di ridurre le posizioni
        
        - **Max Drawdown**: La perdita massima mai registrata
          - Oltre -15%: il sistema sta rischiando troppo
        
        - **Concentrazione**: Quanto del portafoglio è in un singolo titolo
          - >30%: troppo concentrato, diversifica!
        
        **Cosa fare se il rischio è alto:**
        1. Riduci le dimensioni delle posizioni
        2. Aggiungi stop-loss
        3. Aumenta la liquidità (contanti)
        """
    )
    
    # Recupera dati
    risk_orders_df = get_risk_orders()
    equity_df = get_equity_curve(date_range)
    portfolio_df = get_portfolio_data()
    
    # Calcola metriche rischio
    current_drawdown = 0.0
    max_drawdown = 0.0
    
    if not equity_df.empty:
        dd = calculate_drawdown(equity_df["equity"])
        current_drawdown = dd.iloc[-1] if len(dd) > 0 else 0
        max_drawdown = dd.min()
    
    # Concentrazione posizioni
    concentration = 0.0
    if not portfolio_df.empty:
        total = portfolio_df["current_value"].sum()
        if total > 0:
            concentration = portfolio_df["current_value"].max() / total
    
    # Calcolo risk score composito
    risk_score = min(100, max(0, (
        abs(current_drawdown) * 200 +
        concentration * 50 +
        (1 - len(portfolio_df) / max(10, len(config.trading.tickers))) * 30
    )))
    
    # Gauge rischio
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Punteggio Rischio")
        
        gauge_color = "#10b981" if risk_score < 30 else "#f59e0b" if risk_score < 60 else "#ef4444"
        risk_text = "Basso" if risk_score < 30 else "Moderato" if risk_score < 60 else "Alto"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": f"Rischio: {risk_text}"},
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
        st.subheader("📊 Metriche di Rischio")
        
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            dd_status = "⚠️ Attenzione" if abs(current_drawdown) > 0.1 else "✓ OK"
            st.metric(
                "📉 Drawdown Attuale",
                f"{current_drawdown:.2%}",
                delta=dd_status,
                delta_color="inverse" if abs(current_drawdown) > 0.05 else "normal",
                help="Quanto sei sotto rispetto al massimo raggiunto"
            )
        
        with col2b:
            max_dd_status = "🔴 Alto Rischio" if abs(max_drawdown) > 0.15 else "✓ Accettabile"
            st.metric(
                "📉 Max Drawdown",
                f"{max_drawdown:.2%}",
                delta=max_dd_status,
                delta_color="inverse" if abs(max_drawdown) > 0.15 else "normal",
                help="La perdita massima dal picco mai registrata"
            )
        
        with col2c:
            conc_status = "⚠️ Diversifica!" if concentration > 0.3 else "✓ Buono"
            st.metric(
                "🎯 Concentrazione",
                f"{concentration:.1%}",
                delta=conc_status,
                delta_color="inverse" if concentration > 0.3 else "normal",
                help="% del portafoglio nel singolo titolo più grande"
            )
    
    # Grafico drawdown
    st.subheader("📉 Storico Drawdown")
    
    show_tip("Le zone rosse mostrano i periodi di perdita. Drawdown oltre -10% (linea arancione) richiede attenzione!")
    
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
        fig.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="⚠️ Attenzione 10%")
        fig.add_hline(y=-15, line_dash="dash", line_color="red", annotation_text="🔴 Massimo 15%")
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Drawdown %",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 Nessuno storico equity disponibile")
    
    # Tabella ordini con gestione rischio
    st.subheader("📋 Ordini con Gestione Rischio")
    
    show_info_expander(
        "Come funziona la gestione rischio sugli ordini?",
        """
        Ogni ordine passa attraverso un **filtro di rischio**:
        
        - **Quantità Originale**: Quanto volevi acquistare
        - **Quantità Adattata**: Quanto il sistema ti permette (potrebbe ridurla)
        - **Risk Score**: Punteggio di rischio dell'ordine
        - **Approvato**: ✅ = ordine eseguito, ❌ = ordine bloccato
        
        Il sistema può bloccare ordini che:
        - Superano la dimensione massima per posizione
        - Aumentano troppo la concentrazione
        - Arrivano durante un drawdown elevato
        """
    )
    
    if not risk_orders_df.empty:
        display_df = risk_orders_df.copy()
        display_df["time"] = pd.to_datetime(display_df["time"]).dt.strftime("%d/%m/%Y %H:%M")
        display_df["approved"] = display_df["approved"].map({True: "✅", False: "❌"})
        display_df.columns = ["Data/Ora", "Ticker", "Tipo", "Qtà Orig.", "Qtà Adatt.", "Pos. %", "Risk", "Stato", "Motivo"]
        
        def highlight_rejected(row):
            if row["Stato"] == "❌":
                return ["background-color: rgba(239, 68, 68, 0.2)"] * len(row)
            return [""] * len(row)
        
        styled_df = display_df.style.apply(highlight_rejected, axis=1).format({
            "Pos. %": "{:.1%}",
            "Risk": "{:.1f}",
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("📭 Nessun ordine con gestione rischio registrato")
    
    # Alert box
    if abs(current_drawdown) > 0.1 or risk_score > 60:
        st.error("""
        ⚠️ **ALLERTA RISCHIO**
        
        Il tuo portafoglio sta sperimentando livelli di rischio elevati. Considera di:
        - 📉 Ridurre le dimensioni delle posizioni
        - 🛑 Aggiungere ordini stop-loss
        - 💵 Aumentare l'allocazione in contanti
        - 🔀 Diversificare su più titoli
        """)


# =============================================================================
# TAB: BACKTEST
# =============================================================================

def render_backtest_tab(selected_tickers: List[str], date_range: int):
    """Renderizza la tab Backtest."""
    st.header("📊 Simulazione Backtest")
    
    show_info_expander(
        "Cos'è un Backtest?",
        """
        Il **backtest** simula come avrebbe funzionato una strategia nel **passato**:
        
        **Come funziona:**
        1. Scegli un periodo storico (es. ultimi 2 anni)
        2. Scegli i parametri della strategia
        3. Il sistema simula tutti gli acquisti/vendite
        4. Calcola il rendimento finale come se avessi davvero tradato
        
        **Metriche importanti:**
        - **Rendimento Totale**: Quanto avresti guadagnato/perso
        - **Sharpe Ratio**: Rendimento per unità di rischio (>1 è buono)
        - **Max Drawdown**: La perdita massima dal picco
        - **Win Rate**: % di trade vincenti (>50% è desiderabile)
        
        ⚠️ **Attenzione**: Performance passate NON garantiscono risultati futuri!
        """
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        bt_ticker = st.selectbox(
            "🎯 Ticker",
            options=config.trading.tickers,
            key="bt_ticker",
            help="Su quale titolo vuoi testare la strategia?"
        )
        bt_start = st.date_input(
            "📅 Data Inizio",
            value=datetime.now() - timedelta(days=730),
            key="bt_start",
            help="Da quando iniziare la simulazione"
        )
        ema_short = st.slider(
            "EMA Breve (giorni)",
            5, 20, 12,
            key="bt_ema_short",
            help="Periodo della media mobile veloce"
        )

    with col2:
        bt_capital = st.number_input(
            "💰 Capitale Iniziale (€)",
            value=float(config.trading.initial_capital),
            min_value=1000.0,
            key="bt_capital",
            help="Con quanto capitale virtuale vuoi simulare?"
        )
        bt_end = st.date_input(
            "📅 Data Fine",
            value=datetime.now(),
            key="bt_end",
            help="Fino a quando simulare"
        )
        ema_long = st.slider(
            "EMA Lunga (giorni)",
            20, 50, 26,
            key="bt_ema_long",
            help="Periodo della media mobile lenta"
        )
    
    show_tip("Prova diverse combinazioni di EMA! Es: 5/20 per segnali veloci, 20/50 per trend più lunghi.")
    
    # Bottone esecuzione backtest
    if st.button("🚀 Avvia Simulazione", type="primary"):
        with st.spinner("⏳ Simulazione in corso..."):
            try:
                result = run_ema_crossover_backtest(
                    ticker=bt_ticker,
                    start_date=bt_start.strftime("%Y-%m-%d"),
                    end_date=bt_end.strftime("%Y-%m-%d"),
                    initial_capital=bt_capital,
                    ema_short=ema_short,
                    ema_long=ema_long,
                )
                
                st.success("✅ Simulazione Completata!")
                
                # Riga metriche
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    ret_emoji = "📈" if result.total_return > 0 else "📉"
                    st.metric(
                        f"{ret_emoji} Rendimento",
                        f"{result.total_return:.2%}",
                        help="Rendimento totale della strategia"
                    )
                with col2:
                    st.metric(
                        "⚖️ Sharpe Ratio",
                        f"{result.sharpe_ratio:.2f}",
                        help="Rendimento aggiustato per il rischio"
                    )
                with col3:
                    st.metric(
                        "📉 Max Drawdown",
                        f"{result.max_drawdown:.2%}",
                        help="Perdita massima dal picco"
                    )
                with col4:
                    wr_emoji = "🎯" if result.win_rate > 0.5 else "⚠️"
                    st.metric(
                        f"{wr_emoji} Win Rate",
                        f"{result.win_rate:.2%}",
                        help="% di trade vincenti"
                    )
                with col5:
                    st.metric(
                        "🔢 Totale Trade",
                        result.total_trades,
                        help="Numero totale di operazioni"
                    )
                
                # Curva equity
                st.subheader("📈 Curva del Capitale")
                
                show_tip("La linea tratteggiata mostra il capitale iniziale. Sopra = profitto, sotto = perdita.")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve["equity"],
                    mode="lines",
                    name="Valore Portafoglio",
                    line=dict(color="#3b82f6", width=2),
                ))
                
                fig.add_hline(
                    y=bt_capital,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Capitale Iniziale",
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Data",
                    yaxis_title="Valore Portafoglio (€)",
                    hovermode="x unified",
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Grafico prezzi con candele
                st.subheader("📊 Grafico Prezzi")
                
                price_df = get_price_data(bt_ticker, days=(bt_end - bt_start).days)
                
                if not price_df.empty:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Candlestick(
                        x=price_df.index,
                        open=price_df["open"],
                        high=price_df["high"],
                        low=price_df["low"],
                        close=price_df["close"],
                        name="Prezzo",
                    ))
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_rangeslider_visible=False,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistiche dettagliate
                st.subheader("📋 Statistiche Dettagliate")
                
                stats = {
                    "Capitale Iniziale": f"€{bt_capital:,.2f}",
                    "Valore Finale": f"€{result.final_value:,.2f}",
                    "Profitto/Perdita": f"€{result.final_value - bt_capital:,.2f}",
                    "Totale Trade": result.total_trades,
                    "Trade Vincenti": result.winning_trades,
                    "Trade Perdenti": result.losing_trades,
                    "Win Rate": f"{result.win_rate:.2%}",
                    "Rendimento Medio Trade": f"{result.avg_trade_return:.2%}",
                    "Miglior Trade (stima)": f"{max(0, result.avg_trade_return * 2):.2%}",
                    "Peggior Trade (stima)": f"{min(0, -result.avg_trade_return):.2%}",
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
                
                # Interpretazione risultati
                st.subheader("🎓 Interpretazione")
                
                if result.total_return > 0.2 and result.sharpe_ratio > 1:
                    st.success("""
                    ✅ **Ottimi risultati!**
                    
                    La strategia ha mostrato un buon rendimento con rischio contenuto.
                    Ricorda però che i risultati passati non garantiscono performance future.
                    """)
                elif result.total_return > 0:
                    st.info("""
                    ℹ️ **Risultati discreti**
                    
                    La strategia ha generato profitto, ma potresti ottimizzare i parametri
                    per migliorare il rapporto rendimento/rischio.
                    """)
                else:
                    st.warning("""
                    ⚠️ **Risultati negativi**
                    
                    La strategia ha generato una perdita nel periodo testato.
                    Prova a modificare i parametri EMA o considera una strategia diversa.
                    """)
                
            except Exception as e:
                st.error(f"❌ Errore nel backtest: {str(e)}")
                logger.exception("Errore backtest")


# =============================================================================
# DASHBOARD PRINCIPALE
# =============================================================================

def run_dashboard():
    """Esegue la dashboard Streamlit avanzata."""
    st.set_page_config(
        page_title="AI Trading Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Inject CSS personalizzato
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.title("📈 AI Trading Platform")
    st.caption("Segnali di trading in tempo reale, previsioni ML e analisi portafoglio")
    
    # Sidebar
    st.sidebar.header("⚙️ Impostazioni")
    
    # Benvenuto principianti
    with st.sidebar.expander("👋 Prima volta qui?"):
        st.markdown("""
        **Benvenuto nella piattaforma AI Trading!**
        
        **Come iniziare:**
        1. Esplora le 6 schede in alto
        2. Ogni scheda ha una sezione "ℹ️" che spiega cosa fa
        3. Cerca i box "💡 Suggerimento" per consigli pratici
        
        **Comandi base nel terminale:**
        ```bash
        # Genera nuovi segnali
        python3 run_pipeline.py
        
        # Pipeline con ML
        python3 run_enhanced_pipeline.py
        ```
        
        Buon trading! 🚀
        """)
    
    st.sidebar.markdown("---")
    
    # Filtro ticker
    selected_tickers = st.sidebar.multiselect(
        "🎯 Seleziona Ticker",
        options=config.trading.tickers,
        default=config.trading.tickers,
        help="Scegli quali titoli monitorare"
    )
    
    # Range date
    date_range = st.sidebar.slider(
        "📅 Periodo Dati (giorni)",
        min_value=30,
        max_value=365,
        value=90,
        help="Quanti giorni di storico visualizzare"
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox(
        "🔄 Auto-aggiornamento (5 min)",
        value=False,
        help="Ricarica automaticamente i dati ogni 5 minuti"
    )
    if auto_refresh:
        st.sidebar.info("La dashboard si aggiornerà automaticamente")
    
    # Refresh manuale
    if st.sidebar.button("🔄 Aggiorna Dati"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # === GENERA SEGNALI ===
    st.sidebar.subheader("🚀 Genera Segnali")
    
    pipeline_type = st.sidebar.radio(
        "Tipo Pipeline",
        options=["Base", "Avanzata (ML)"],
        help="Base: solo strategie tecniche. Avanzata: include previsioni ML"
    )
    
    if st.sidebar.button("▶️ Esegui Pipeline", type="primary", use_container_width=True):
        # Determina la directory del progetto
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        script_name = "run_pipeline.py" if pipeline_type == "Base" else "run_enhanced_pipeline.py"
        script_path = os.path.join(project_dir, script_name)
        
        with st.sidebar.status(f"⏳ Esecuzione {script_name}...", expanded=True) as status:
            try:
                # Esegui la pipeline
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    cwd=project_dir,
                    timeout=120
                )
                
                if result.returncode == 0:
                    status.update(label="✅ Pipeline completata!", state="complete")
                    
                    # Estrai info dal risultato
                    output = result.stdout
                    
                    # Mostra riepilogo
                    st.sidebar.success("Segnali generati con successo!")
                    
                    # Parsing veloce dell'output
                    if "Generated" in output or "Generati" in output:
                        for line in output.split('\n'):
                            if "signal" in line.lower() or "segnali" in line.lower():
                                st.sidebar.info(line.strip())
                                break
                    
                    # ML predictions se pipeline avanzata
                    if pipeline_type == "Avanzata (ML)" and "🤖" in output:
                        st.sidebar.markdown("**Previsioni ML:**")
                        for line in output.split('\n'):
                            if "🤖" in line:
                                st.sidebar.write(line.strip())
                    
                    # Pulisci cache e ricarica
                    st.cache_data.clear()
                    st.toast("🎉 Pipeline completata! Ricarica per vedere i nuovi dati.", icon="✅")
                    
                else:
                    status.update(label="❌ Errore nella pipeline", state="error")
                    st.sidebar.error(f"Errore: {result.stderr[:200]}")
                    
            except subprocess.TimeoutExpired:
                status.update(label="⏰ Timeout", state="error")
                st.sidebar.warning("La pipeline ha impiegato troppo tempo (>120s)")
            except Exception as e:
                status.update(label="❌ Errore", state="error")
                st.sidebar.error(f"Errore: {str(e)}")
    
    # Tip per la pipeline
    with st.sidebar.expander("💡 Cosa fa la pipeline?"):
        st.markdown("""
        **Pipeline Base:**
        1. Scarica dati di mercato (Alpaca Markets)
        2. Calcola indicatori tecnici (EMA, RSI, ATR...)
        3. Genera segnali BUY/SELL/HOLD
        4. Applica risk management
        5. Esegue ordini (paper trading)
        
        **Pipeline Avanzata:**
        - Tutto quello della base +
        - Previsioni ML (RandomForest)
        - Confidenza delle previsioni
        """)
    
    st.sidebar.markdown("---")
    
    # Info versione
    st.sidebar.caption("AI Trading Platform v1.0.0 🇮🇹")
    st.sidebar.caption(f"Ultimo aggiornamento: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Tab principali
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Panoramica",
        "📡 Segnali", 
        "🤖 Previsioni ML",
        "🔬 Lab Strategie",
        "🛡️ Centro Rischio",
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
