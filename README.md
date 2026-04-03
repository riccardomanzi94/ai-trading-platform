# AI Trading Platform

Piattaforma Python production-like per analisi storica di azioni/ETF, generazione segnali, risk engine, paper execution e dashboard di monitoraggio.

## Architettura

```
ai-trading-platform/
├── pyproject.toml          # Configurazione progetto e dipendenze
├── prefect.yaml            # Orchestrazione workflow Prefect
├── infra/db/init.sql       # Schema database TimescaleDB
├── orchestrator/flows/     # Pipeline Prefect
├── src/ai_trading/         # Codice applicativo
│   ├── shared/             # Configurazione condivisa
│   ├── data_ingestion/     # Download dati da yfinance
│   ├── feature_store/      # Feature engineering (EMA, RSI, ATR)
│   ├── signals/            # Generazione segnali trading
│   ├── risk_engine/        # Policy e applicazione risk management
│   ├── execution/          # Paper trading simulation
│   ├── backtest/           # Backtesting strategie
│   └── monitoring/         # Dashboard Streamlit
└── tests/                  # Test automatici pytest
```

## Requisiti

- Python 3.11+
- PostgreSQL con estensione TimescaleDB
- Prefect 2.x

## Installazione

```bash
# Crea virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Installa dipendenze
pip install -e ".[dev]"

# Configura database
psql -U postgres -f infra/db/init.sql
```

## Configurazione

Crea un file `.env` nella root del progetto:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_trading
TICKERS=SPY,QQQ,IWM,AAPL,MSFT,GOOGL
INITIAL_CAPITAL=100000.0
MAX_POSITION_SIZE=0.1
MAX_PORTFOLIO_RISK=0.02
```

## Esecuzione

### Pipeline giornaliera

```bash
# Registra e avvia il flow Prefect
prefect deploy --all
prefect agent start -q default
```

### Dashboard

```bash
streamlit run src/ai_trading/monitoring/dashboard.py
```

### Test

```bash
pytest tests/ -v
```

## Moduli

### Data Ingestion
Scarica dati storici OHLCV da Yahoo Finance usando `yfinance` con `auto_adjust=False`.

### Feature Store
Calcola indicatori tecnici:
- EMA (Exponential Moving Average) - 12 e 26 periodi
- RSI (Relative Strength Index) - 14 periodi
- ATR (Average True Range) - 14 periodi
- Rolling Volatility - 20 periodi

### Signals
Genera segnali di trading basati su EMA crossover:
- BUY: EMA12 incrocia EMA26 verso l'alto
- SELL: EMA12 incrocia EMA26 verso il basso

### Risk Engine
Applica regole di risk management:
- Position sizing basato su volatilità
- Max position size per ticker
- Max portfolio risk

### Execution
Paper trading engine per simulazione ordini senza esecuzione reale.

### Backtest
Backtesting della strategia EMA crossover su dati storici.

### Monitoring
Dashboard Streamlit per visualizzare:
- Performance portfolio
- Segnali attivi
- Metriche di rischio
