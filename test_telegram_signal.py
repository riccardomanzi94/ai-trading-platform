#!/usr/bin/env python3
"""Test script per inviare un segnale di trading di prova su Telegram."""

import os
from datetime import datetime
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

from ai_trading.alerts.notifier import send_signal_alert, send_execution_alert, AlertLevel, send_alert


class MockSignal:
    """Mock signal per test."""
    def __init__(self, signal_type, ticker, strength, price):
        self.signal_type = type('obj', (object,), {'value': signal_type})()
        self.ticker = ticker
        self.strength = strength
        self.price_at_signal = price
        self.time = datetime.now()


class MockExecution:
    """Mock execution per test."""
    def __init__(self, side, ticker, quantity, price):
        self.side = side
        self.ticker = ticker
        self.quantity = quantity
        self.price = price
        self.commission = price * quantity * 0.001  # 0.1% commissione
        self.total_value = price * quantity


def test_trading_signals():
    """Testa l'invio di vari tipi di notifiche trading."""

    print("=" * 60)
    print("🧪 Test Notifiche Trading Telegram")
    print("=" * 60)

    # Test 1: Segnale BUY
    print("\n📈 Test 1: Invio segnale BUY...")
    buy_signal = MockSignal(
        signal_type="BUY",
        ticker="SPY",
        strength=0.85,
        price=450.25
    )
    success = send_signal_alert(buy_signal)
    print(f"   {'✅ Inviato!' if success else '❌ Errore'}")

    # Test 2: Segnale SELL
    print("\n📉 Test 2: Invio segnale SELL...")
    sell_signal = MockSignal(
        signal_type="SELL",
        ticker="QQQ",
        strength=0.72,
        price=385.50
    )
    success = send_signal_alert(sell_signal)
    print(f"   {'✅ Inviato!' if success else '❌ Errore'}")

    # Test 3: Trade Executed
    print("\n💰 Test 3: Invio notifica trade eseguito...")
    execution = MockExecution(
        side="BUY",
        ticker="AAPL",
        quantity=10,
        price=175.50
    )
    success = send_execution_alert(execution)
    print(f"   {'✅ Inviato!' if success else '❌ Errore'}")

    # Test 4: Alert personalizzato
    print("\n⚠️  Test 4: Invio alert risk management...")
    success = send_alert(
        title="Risk Alert: Drawdown Elevato",
        message="Il drawdown del portafoglio ha superato il 5%. Considera di ridurre le posizioni.",
        level=AlertLevel.WARNING,
        ticker="PORTFOLIO",
        data={
            "drawdown": -5.2,
            "max_position": "TSLA",
            "cash_available": 45000.00,
        }
    )
    print(f"   {'✅ Inviato!' if success else '❌ Errore'}")

    # Test 5: Report giornaliero
    print("\n📊 Test 5: Invio report giornaliero...")
    from ai_trading.alerts.notifier import send_daily_summary
    success = send_daily_summary(
        portfolio_value=125430.50,
        daily_pnl=+2340.75,
        positions={
            "SPY": {"quantity": 50},
            "QQQ": {"quantity": 30},
            "AAPL": {"quantity": 20},
            "_CASH": {"quantity": 0},
        },
        signals_today=3,
        trades_today=2
    )
    print(f"   {'✅ Inviato!' if success else '❌ Errore'}")

    print("\n" + "=" * 60)
    print("✅ Tutti i test completati!")
    print("📱 Controlla Telegram per i messaggi di test")
    print("=" * 60)


if __name__ == "__main__":
    test_trading_signals()
