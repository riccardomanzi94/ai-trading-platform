# 🚀 AI Trading Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)

**Una piattaforma professionale di trading algoritmico con Machine Learning, backtesting avanzato e dashboard interattiva in italiano.**

---

## 📋 Indice

- [Panoramica](#-panoramica)
- [Screenshot](#-screenshot)
- [Caratteristiche](#-caratteristiche)
- [Quick Start](#-quick-start)
- [Guida Dashboard](#-guida-dashboard-passo-passo)
- [Configurazione](#%EF%B8%8F-configurazione)
- [Architettura](#-architettura)
- [Notifiche](#-sistema-notifiche)
- [Disclaimer](#%EF%B8%8F-disclaimer)

---

## 🎯 Panoramica

AI Trading Platform è una soluzione completa per il trading algoritmico che combina:

- **3 Strategie di Trading** testate e configurabili
- **Machine Learning** per previsioni di mercato
- **Risk Management** professionale con controlli VaR
- **Backtesting** completo con metriche dettagliate
- **Dashboard Interattiva** in italiano con tutorial integrati

Perfetta sia per chi inizia sia per trader esperti che vogliono automatizzare le proprie strategie.

---

## 📸 Screenshot

### Dashboard Panoramica
![Dashboard Overview](docs/images/dashboard_overview.png)

### Segnali di Trading
![Trading Signals](docs/images/dashboard_signals.png)

### Previsioni ML
![ML Predictions](docs/images/dashboard_ml_predictions.png)

### Laboratorio Strategie
![Strategy Lab](docs/images/dashboard_strategy_lab.png)

### Centro Rischio
![Risk Center](docs/images/dashboard_risk_center.png)

### Backtest
![Backtest](docs/images/dashboard_backtest.png)

### Sidebar con Controlli
![Sidebar](docs/images/dashboard_sidebar.png)

---

## ✨ Caratteristiche

| Modulo | Descrizione |
|--------|-------------|
| 📊 **Data Ingestion** | Scarica dati OHLCV da Alpaca Markets (real-time, istituzionale) + FRED macroeconomici |
| 🧮 **Feature Store** | Calcola 50+ indicatori tecnici (RSI, MACD, Bollinger, ATR...) |
| 📈 **Signals** | 3 strategie: EMA Crossover, RSI Mean Reversion, Momentum |
| 🤖 **ML Engine** | RandomForest + GradientBoosting con ottimizzazione Optuna |
| ⚠️ **Risk Engine** | Position sizing, stop-loss, VaR, drawdown limits |
| 💼 **Execution** | Motore ordini con slippage simulation |
| 📉 **Backtest** | Walk-forward, metriche Sharpe/Sortino/Calmar |
| 🖥️ **Dashboard** | 6 tab interattive con grafici Plotly |
| 🔔 **Alerts** | Telegram, Slack, Email |
| 🏦 **Broker** | Integrazione Alpaca Markets |

---

## 🚀 Quick Start

### 1. Clona il Repository

```bash
git clone https://github.com/riccardomanzi94/ai-trading-platform.git
cd ai-trading-platform
```

### 2. Crea Ambiente Virtuale

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: venv\Scripts\activate  # Windows
```

### 3. Installa Dipendenze

```bash
pip install -e .
```

### 4. Avvia Database (Opzionale)

```bash
docker-compose up -d
```

### 5. Esegui Pipeline

```bash
# Pipeline base
python3 run_pipeline.py

# Pipeline avanzata con ML
python3 run_enhanced_pipeline.py
```

### 6. Avvia Dashboard

```bash
python3 -m streamlit run src/ai_trading/monitoring/dashboard.py --server.port 8503
```

Apri il browser su: **http://localhost:8503**

---

## 📱 Guida Dashboard Passo Passo

La dashboard è pensata per essere intuitiva anche per chi inizia. Ogni sezione include **💡 Tip** con spiegazioni semplici.

### Tab 1: 🏠 Panoramica

**Cosa trovi:**
- 💰 **Valore Portfolio**: Quanto vale il tuo portafoglio totale
- 📊 **Sharpe Ratio**: Misura il rendimento rispetto al rischio (sopra 1.0 è buono!)
- 📉 **Max Drawdown**: La perdita massima dal picco (più basso = meglio)
- 📈 **Grafico Equity**: L'andamento del tuo capitale nel tempo

**💡 Tip per Neofiti:**
> Lo Sharpe Ratio ti dice se stai guadagnando "bene". Un valore di 2.0 significa rendimenti eccellenti con rischio contenuto!

---

### Tab 2: 📈 Segnali di Trading

**Cosa trovi:**
- Tabella con segnali **BUY** (compra), **SELL** (vendi), **HOLD** (mantieni)
- Confidenza del segnale in percentuale
- Filtri per simbolo e tipo di segnale

**Come usarla:**
1. Seleziona un simbolo dalla sidebar (es. AAPL, MSFT)
2. Guarda la colonna "Segnale" per vedere cosa suggerisce la strategia
3. Controlla la "Confidenza" - più è alta, più il segnale è affidabile

**💡 Tip per Neofiti:**
> Non seguire ciecamente i segnali! Usali come supporto alle tue decisioni. Confidenza > 70% indica segnali più robusti.

---

### Tab 3: 🤖 Previsioni ML

**Cosa trovi:**
- Previsioni del Machine Learning per i prossimi movimenti
- Confronto tra modello RandomForest e GradientBoosting
- Grafico delle probabilità UP/DOWN

**Come usarla:**
1. Scegli il simbolo da analizzare
2. Confronta le previsioni dei due modelli
3. Se entrambi concordano, il segnale è più affidabile

**💡 Tip per Neofiti:**
> Il ML non è magia! Funziona meglio in mercati con trend definiti. In mercati laterali può dare falsi segnali.

---

### Tab 4: 🔬 Strategy Lab (Laboratorio Strategie)

**Cosa trovi:**
- Test e confronto delle 3 strategie disponibili
- Parametri configurabili per ogni strategia
- Grafici comparativi di performance

**Le 3 Strategie:**

| Strategia | Come Funziona | Ideale Per |
|-----------|---------------|------------|
| **EMA Crossover** | Compra quando EMA veloce incrocia EMA lenta verso l'alto | Mercati in trend |
| **RSI Mean Reversion** | Compra quando RSI < 30 (ipervenduto) | Mercati laterali |
| **Momentum Breakout** | Compra su rottura di resistenze con volume | Breakout esplosivi |

**💡 Tip per Neofiti:**
> Inizia con EMA Crossover, è la più semplice da capire. Usa periodi più lunghi (20/50) per segnali più affidabili ma meno frequenti.

---

### Tab 5: ⚠️ Centro Rischio

**Cosa trovi:**
- 📊 **VaR (Value at Risk)**: Quanto potresti perdere nel 95% dei casi
- 📉 **Drawdown Corrente**: Quanto sei sotto dal massimo
- 🎯 **Position Sizing**: Quanto investire per ogni trade
- ⚡ **Esposizione**: Quanto del capitale è investito

**Come usarla:**
1. Controlla il VaR giornaliero - è la tua "perdita massima probabile"
2. Se il drawdown supera il 10%, considera di ridurre le posizioni
3. Non superare mai il 2% di rischio per singolo trade

**💡 Tip per Neofiti:**
> La regola d'oro: non rischiare mai più del 2% del capitale su un singolo trade. Con 10.000€, max 200€ per trade!

---

### Tab 6: 📉 Backtest

**Cosa trovi:**
- Test delle strategie su dati storici
- Metriche complete: Sharpe, Sortino, Calmar, Win Rate
- Grafico equity curve del backtest
- Lista di tutti i trade simulati

**Come usarla:**
1. Seleziona strategia, simbolo e periodo
2. Clicca "Esegui Backtest"
3. Analizza i risultati - Win Rate > 50% è un buon inizio
4. Guarda il Max Drawdown - se è troppo alto, la strategia è rischiosa

**Metriche Spiegate:**

| Metrica | Significato | Valore Buono |
|---------|-------------|--------------|
| **Sharpe Ratio** | Rendimento/Rischio | > 1.0 |
| **Sortino Ratio** | Come Sharpe ma considera solo perdite | > 1.5 |
| **Win Rate** | % trade vincenti | > 50% |
| **Profit Factor** | Guadagni/Perdite totali | > 1.5 |
| **Max Drawdown** | Perdita massima | < 20% |

**💡 Tip per Neofiti:**
> Un backtest positivo NON garantisce profitti futuri! I mercati cambiano. Usa il backtest per capire come si comporta una strategia, non come previsione.

---

## ⚙️ Configurazione

### File .env

Crea un file `.env` nella root del progetto:

```env
# Trading - Asset da analizzare (25 asset: ETF + Azioni)
TICKERS=SPY,QQQ,IWM,VOO,VTI,XLF,XLK,XLE,XLU,XLI,XLP,XLV,XLY,TLT,BND,AGG,LQD,VEA,VWO,GLD,USO,AAPL,MSFT,GOOGL,TSLA
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.1
MAX_PORTFOLIO_RISK=0.02

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/trading

# Broker (Alpaca) - Dati di mercato real-time
# Ottieni API key gratuita: https://alpaca.markets/
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Dati Macroeconomici (FRED)
# Ottieni API key gratuita: https://research.stlouisfed.org/useraccount/apikey
FRED_API_KEY=your_fred_api_key

# Notifiche Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Notifiche Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
```

### ETF e Asset Inclusi

| Categoria | ETF | Descrizione |
|-----------|-----|-------------|
| **Broad Market** | SPY, VOO, QQQ, VTI, IWM | S&P 500, NASDAQ-100, Totale USA, Russell 2000 |
| **Settoriali** | XLF, XLK, XLE, XLU, XLI, XLP, XLV, XLY | Finanziari, Tech, Energia, Utilities, Industriali, Consumer Staples, Healthcare, Consumer Discretionary |
| **Obbligazioni** | TLT, BND, AGG, LQD | Treasury 20+, Bond Market, Aggregate Bond, Corporate IG |
| **Internazionali** | VEA, VWO | Mercati Sviluppati, Emergenti |
| **Commodities** | GLD, USO | Oro, Petrolio |
| **Azioni** | AAPL, MSFT, GOOGL, TSLA | Big Tech + Tesla |

### Configurare Telegram

1. Cerca `@BotFather` su Telegram
2. Invia `/newbot` e segui le istruzioni
3. Copia il token nel file `.env`
4. Cerca `@userinfobot` per ottenere il tuo Chat ID

---

## 📈 Fonti Dati

La piattaforma utilizza fonti dati professionali e affidabili:

### Alpaca Markets (Dati di Mercato)
- **Tipo**: Dati OHLCV (Open, High, Low, Close, Volume) reali
- **Qualità**: Dati istituzionali, clean e senza bias
- **Prezzo**: Gratuito (paper trading), include 15 min di delay o real-time per alcuni exchange
- **Vantaggi**: API stabile, limiti di rate generosi (200 req/min), paginazione automatica
- **Copertura**: Azioni USA, ETF, alcune crypto

### FRED - Federal Reserve Economic Data (Dati Macroeconomici)
- **Tipo**: Indicatori economici (tassi di interesse, inflazione, disoccupazione, ecc.)
- **Prezzo**: Gratuito con registrazione
- **Indicatori principali**:
  - `DFF`: Federal Funds Rate (tasso base della Fed)
  - `DGS10`: Treasury 10 anni (benchmark dei tassi)
  - `T10Y2Y`: Spread 10Y-2Y (indicatore di recessione quando negativo)
  - `CPIAUCSL`: Indice dei prezzi al consumo (inflazione)
  - `UNRATE`: Tasso di disoccupazione
  - `VIXCLS`: Indice di volatilità VIX
  - `DTWEXBGS`: Indice del dollaro USA

### Perché Alpaca invece di Yahoo Finance?
- ✅ Dati più affidabili e consistenti
- ✅ API ufficiale con supporto (non scraping)
- ✅ Split e dividendi gestiti automaticamente
- ✅ Paginazione per dataset storici grandi
- ✅ Rate limiting trasparente
- ✅ No problemi di IP bloccati o CAPTCHA

---

## 🏗️ Architettura

```
ai-trading-platform/
├── src/ai_trading/
│   ├── data_ingestion/    # Download dati mercato (Alpaca) + macro (FRED)
│   │   ├── main.py        # Ingestion dati prezzi da Alpaca
│   │   └── fred_client.py # Ingestion dati macroeconomici
│   ├── feature_store/     # Calcolo indicatori tecnici
│   ├── signals/           # Generazione segnali trading
│   ├── risk_engine/       # Gestione rischio
│   ├── execution/         # Esecuzione ordini
│   ├── backtest/          # Backtesting strategie
│   ├── monitoring/        # Dashboard Streamlit
│   │   └── dashboard.py   # Dashboard italiana
│   ├── ml/                # Machine Learning
│   ├── alerts/            # Sistema notifiche
│   └── broker/            # Integrazione broker (Alpaca)
├── infra/db/
│   └── init.sql           # Schema database (include tabella macro_data)
├── tests/                 # Test suite
├── run_pipeline.py        # Pipeline base
└── run_enhanced_pipeline.py  # Pipeline con ML
```

---

## 🔔 Sistema Notifiche

Ricevi alert in tempo reale quando:
- ✅ Viene generato un nuovo segnale di trading
- ⚠️ Il drawdown supera la soglia
- 🎯 Un ordine viene eseguito
- 📊 Report giornaliero performance

Canali supportati: **Telegram**, **Slack**, **Email**

---

## 🤖 GitHub Copilot Agent

Il progetto include un agent personalizzato per GitHub Copilot che assiste nello sviluppo di strategie di trading.

### Trading Strategy Developer Agent

Situato in `.github/agents/trading-strategy.agent.md`, questo agent è specializzato in:

- **Indicatori Tecnici**: EMA, RSI, ATR, momentum, volatility
- **Generazione Segnali**: crossover, mean reversion, breakout
- **Ensemble Multi-Strategia**: combinazione pesata di più strategie
- **Ottimizzazione ML**: feature engineering e tuning parametri
- **Risk Management**: position sizing, stop-loss, VaR

**Prompt di esempio:**
```
@trading-strategy Crea una strategia MACD divergence
@trading-strategy Ottimizza le soglie RSI mean reversion
@trading-strategy Aggiungi dati macro FRED come filtro regime
```

---

## ⚠️ Disclaimer

> **ATTENZIONE**: Questo software è fornito a scopo educativo e di ricerca. 
> 
> - NON costituisce consulenza finanziaria
> - Il trading comporta rischi significativi di perdita
> - I risultati passati NON garantiscono risultati futuri
> - Testa sempre su paper trading prima di usare denaro reale
> - L'autore NON è responsabile per eventuali perdite finanziarie

---

## 📄 Licenza

MIT License - Vedi [LICENSE](LICENSE) per dettagli.

---

## 🤝 Contribuire

Pull request benvenute! Per modifiche importanti, apri prima una issue.

---

**Made with ❤️ and 🤖 AI**
