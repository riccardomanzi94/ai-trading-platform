#!/usr/bin/env python3
"""Run the full AI Trading pipeline."""

from ai_trading.data_ingestion import ingest_all_tickers
from ai_trading.feature_store import build_all_features
from ai_trading.signals import generate_all_signals
from ai_trading.risk_engine import apply_risk_to_all_signals
from ai_trading.execution import execute_all_orders, initialize_portfolio

print('=' * 50)
print('AI TRADING PLATFORM - Pipeline Automatica')
print('=' * 50)

# 1. Inizializza portfolio
print('\n[1/5] Inizializzazione portfolio...')
initialize_portfolio(100000)
print('✓ Portfolio inizializzato con $100,000')

# 2. Ingestion dati
print('\n[2/5] Download dati da Yahoo Finance...')
ingestion = ingest_all_tickers(start_date='2024-01-01')
total_rows = sum(ingestion.values())
print(f'✓ Scaricati {total_rows} righe per {len(ingestion)} ticker')
for ticker, count in ingestion.items():
    print(f'   {ticker}: {count} righe')

# 3. Feature engineering
print('\n[3/5] Calcolo indicatori tecnici...')
features = build_all_features()
for ticker, df in features.items():
    print(f'   {ticker}: {len(df)} feature rows')
print('✓ EMA, RSI, ATR, Volatility calcolati')

# 4. Generazione segnali
print('\n[4/5] Generazione segnali trading...')
signals = generate_all_signals()
signal_count = sum(len(s) for s in signals.values())
print(f'✓ Generati {signal_count} segnali')
for ticker, sigs in signals.items():
    for s in sigs:
        print(f'   {ticker}: {s.signal_type.value} @ ${s.price_at_signal:.2f} (strength: {s.strength:.2f})')

# 5. Risk management + Execution
print('\n[5/5] Applicazione risk management ed esecuzione...')
if signal_count > 0:
    orders = apply_risk_to_all_signals(signals)
    approved = [o for o in orders if o.approved]
    print(f'✓ {len(approved)}/{len(orders)} ordini approvati')
    
    executions = execute_all_orders(orders)
    print(f'✓ {len(executions)} trade eseguiti')
    for e in executions:
        print(f'   {e.side} {e.quantity} {e.ticker} @ ${e.price:.2f} = ${e.total_value:.2f}')
else:
    print('✓ Nessun segnale da processare (mercato in HOLD)')

print('\n' + '=' * 50)
print('Pipeline completata!')
print('=' * 50)
