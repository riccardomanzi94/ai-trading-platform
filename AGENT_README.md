# 🤖 AI Trader Agent

Un agente AI intelligente che opera come un trader professionista, utilizzando la piattaforma di trading algoritmico in modo efficiente e naturale.

---

## 🎯 Caratteristiche

### 💬 Interazione Naturale
- Comandi in linguaggio naturale (inglese/italiano)
- Risposte chiare e strutturate
- Contesto conversazionale mantenuto

### 🎛️ Tre Modalità Operative

| Modalità | Descrizione | Ideale Per |
|----------|-------------|------------|
| **MANUAL** | Solo analisi e consigli | Apprendimento, verifica strategie |
| **SEMI_AUTO** | Suggerisce, attende approvazione | Controllo + automazione |
| **AUTO** | Esegue completamente autonomo | Trader esperti, cron job |

### 🧠 Capacità Intelligenti
- **Analisi Tecnica**: EMA, RSI, ATR, Volatilità
- **Multi-Strategia**: Combina segnali da più strategie
- **Risk Management**: Position sizing, VaR, stop-loss automatici
- **Portfolio Management**: Monitoraggio e ottimizzazione
- **Apprendimento**: Migliora dalle performance passate

---

## 🚀 Quick Start

### 1. Avvia l'Agent in Modalità Interattiva

```bash
python run_agent.py
```

### 2. Prova i Comandi

```
> help                    # Mostra aiuto
> status                  # Stato agent e portafoglio
> analyze AAPL           # Analizza Apple
> analyze SPY            # Analizza S&P 500 ETF
> portfolio              # Dettaglio portafoglio
> mode auto              # Passa in modalità automatica
> scan                   # Scansiona tutti i ticker
> buy AAPL 10            # Compra 10 azioni Apple
> sell MSFT              # Vendi Microsoft
```

---

## 📚 Comandi Disponibili

### Informazioni
- `help` - Mostra guida completa
- `status` - Stato agent e mercato
- `portfolio` - Dettaglio portafoglio
- `config` - Configurazione attuale

### Analisi
- `analyze <TICKER>` - Analisi completa di un ticker
- `scan` / `run` - Scansiona tutti i ticker configurati

### Trading
- `buy <TICKER> [qty]` - Esegui ordine di acquisto
- `sell <TICKER> [qty]` - Esegui ordine di vendita

### Gestione
- `mode <manual|semi|auto>` - Cambia modalità operativa
- `history` - Storico trade eseguiti

---

## 🔧 Modalità di Esecuzione

### 1. Shell Interattiva (Default)

```bash
python run_agent.py
python run_agent.py --mode auto
python run_agent.py --confidence 0.7
```

### 2. Comando Singolo

```bash
python run_agent.py --cmd "analyze SPY"
python run_agent.py --mode auto --cmd "scan"
```

### 3. API Server

```bash
python run_agent.py --server
python run_agent.py --server --port 8000
```

Endpoints disponibili:
- `GET /status` - Stato agent
- `POST /analyze` - Analizza ticker
- `POST /trade` - Esegui trade
- `POST /scan` - Scansiona mercato
- `POST /chat` - Chat con l'agent
- `POST /mode/{mode}` - Cambia modalità

### 4. Daemon Programmato

```bash
# Esegue ogni 30 minuti (default)
python run_agent.py --daemon

# Esegue ogni ora
python run_agent.py --daemon --interval 60 --mode auto
```

---

## 💡 Esempi di Utilizzo

### Scenario 1: Analisi Prima del Trading

```
> mode manual
🔄 Mode switched to MANUAL. I will only provide analysis and recommendations.

> analyze TSLA
📊 Analisi TSLA
💰 Prezzo: $242.50
📈 Variazione: 1D: +2.34% | 7D: -1.23%
📉 Trend: UPTREND
🟢 Sentiment: BULLISH
📊 Volatilità: 45.2%

🔔 Segnali:
   • BUY (forza: 75%)

🎯 Raccomandazione: STRONG BUY

> mode semi
🔄 Mode switched to SEMI_AUTO.

> buy TSLA
🤖 Suggerimento BUY per TSLA
Confermare l'ordine? (y/n)
> y
✅ Eseguito BUY TSLA: 5 @ $242.50
```

### Scenario 2: Trading Automatico

```bash
# Avvia in modalità auto e esegui scan
python run_agent.py --mode auto --cmd "scan"

# O come daemon
python run_agent.py --daemon --mode auto --interval 30
```

### Scenario 3: Integrazione API

```python
import requests

# Analisi
response = requests.post("http://localhost:8000/analyze", json={"ticker": "AAPL"})
analysis = response.json()

# Trade
response = requests.post("http://localhost:8000/trade", json={
    "ticker": "AAPL",
    "action": "BUY",
    "quantity": 10
})
result = response.json()
```

---

## ⚙️ Configurazione

### Variabili d'Ambiente

Crea un file `.env` nella root:

```env
# Trading - 25 asset: ETF (Broad, Sector, Bonds, Intl, Commodities) + Azioni
TICKERS=SPY,QQQ,IWM,VOO,VTI,XLF,XLK,XLE,XLU,XLI,XLP,XLV,XLY,TLT,BND,AGG,LQD,VEA,VWO,GLD,USO,AAPL,MSFT,GOOGL,TSLA
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.1
MAX_PORTFOLIO_RISK=0.02

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/ai_trading

# Notifiche (opzionale)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Personalizzazione Agent

```python
from ai_trading.agent import TraderAgent, AgentMode

agent = TraderAgent(
    mode=AgentMode.SEMI_AUTO,
    min_confidence=0.7,          # Più conservativo
    max_daily_trades=5,          # Limita trade
)
```

---

## 🏗️ Architettura

```
┌─────────────────────────────────────────────┐
│              AI Trader Agent                  │
├─────────────────────────────────────────────┤
│  • Natural Language Interface               │
│  • Decision Engine                            │
│  • Risk Manager                             │
│  • Portfolio Optimizer                      │
└────────────┬────────────────────────────────┘
             │
┌────────────▼────────────────────────────────┐
│         Trading Platform                    │
│  ┌──────────┬──────────┬──────────┐          │
│  │  Signals │   ML     │   Risk   │          │
│  └──────────┴──────────┴──────────┘          │
│  ┌──────────┬──────────┬──────────┐          │
│  │Execution │ Backtest │  Alerts  │          │
│  └──────────┴──────────┴──────────┘          │
└─────────────────────────────────────────────┘
```

---

## 📊 Flusso Decisionale

```
Utente/Trigger
      │
      ▼
┌─────────────────┐
│  Market Scan    │◄── Analisi multi-ticker
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Signal Gen      │◄── 3 Strategie + ML
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Risk Assessment │◄── Position sizing, VaR
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Decision Point  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────┐  ┌──────┐
│ MANUAL│  │ EXEC │
│Review │  │Trade │
└──────┘  └──────┘
```

---

## 🔐 Sicurezza

- **MANUAL**: Zero rischi, solo analisi
- **SEMI_AUTO**: Controllo umano su ogni trade
- **AUTO**: Limiti configurati (max trade/giorno, position size, ecc.)

### Best Practices

1. Inizia sempre con `mode manual`
2. Testa su paper trading prima del live
3. Imposta `max_daily_trades` ragionevoli
4. Monitora le notifiche Telegram/Slack
5. Rivendi periodicamente le performance

---

## 🐛 Troubleshooting

### Errore: "Portafoglio non inizializzato"
```bash
# Inizializza il database
docker-compose up -d
python run_pipeline.py  # Una volta per inizializzare
```

### Errore: "No module named 'ai_trading'"
```bash
# Installa il pacchetto
pip install -e .
```

### Errore: "Database connection failed"
```bash
# Verifica che PostgreSQL sia running
docker-compose ps
# O usa SQLite modificando DATABASE_URL in .env
```

---

## 📈 Performance Tracking

L'agent traccia automaticamente:
- Win rate
- Profit factor
- Sharpe ratio
- Max drawdown
- Trade history

Accedi con:
```
> history
> portfolio
```

---

## 🤝 Integrazione con la Dashboard

L'agent è complementare alla dashboard Streamlit:
- **Dashboard**: Visualizzazione grafica, backtest
- **Agent**: Interazione, automazione, esecuzione

Usali insieme:
```bash
# Terminale 1: Dashboard
python -m streamlit run src/ai_trading/monitoring/dashboard.py

# Terminale 2: Agent
python run_agent.py --daemon --mode auto
```

---

## 📄 Licenza

MIT License - Vedi LICENSE per dettagli.

---

**Made with 🤖 AI**
