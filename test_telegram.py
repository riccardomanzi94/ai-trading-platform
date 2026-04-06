#!/usr/bin/env python3
"""Test script per verificare la connessione Telegram."""

import os
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

from ai_trading.alerts.notifier import TelegramNotifier, Alert, AlertLevel


def test_telegram_connection():
    """Testa la connessione a Telegram."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print("=" * 50)
    print("🧪 Test Connessione Telegram")
    print("=" * 50)
    print(f"\n📋 Configurazione:")
    print(f"   Token: {'✅ Configurato' if token and len(token) > 10 else '❌ MANCANTE'}")
    print(f"   Chat ID: {'✅ Configurato' if chat_id else '❌ MANCANTE'}")

    if not token or not chat_id:
        print("\n❌ ERRORE: Credenziali mancanti!")
        print("\n✏️  Modifica il file .env con i tuoi dati:")
        print("   TELEGRAM_BOT_TOKEN=il_tuo_token")
        print("   TELEGRAM_CHAT_ID=il_tuo_chat_id")
        return False

    # Test invio messaggio
    print("\n📤 Invio messaggio di test...")

    notifier = TelegramNotifier()

    alert = Alert(
        title="🤖 AI Trading - Test Connessione",
        message="Ciao! Le notifiche Telegram sono configurate correttamente! 🎉",
        level=AlertLevel.INFO,
        ticker="SPY",
        data={
            "stato": "Connesso",
            "piattaforma": "AI Trading Platform",
        }
    )

    success = notifier.send(alert)

    if success:
        print("✅ Messaggio inviato con successo!")
        print("📱 Controlla Telegram per il messaggio di test")
    else:
        print("❌ Errore nell'invio del messaggio")
        print("\n🔍 Verifica:")
        print("   1. Hai copiato correttamente il token?")
        print("   2. Hai avviato una chat con il bot?")
        print("   3. Il token non è scaduto? (rigeneralo con @BotFather se necessario)")

    return success


if __name__ == "__main__":
    test_telegram_connection()
