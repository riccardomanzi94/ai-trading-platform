#!/usr/bin/env python3
"""
Script di esempio per scaricare dati macroeconomici da FRED.
"""

from __future__ import annotations

import logging
import os
import sys

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_api_key():
    """Check if FRED API key is configured."""
    if not os.getenv("FRED_API_KEY"):
        print("\n" + "=" * 70)
        print("ERRORE: FRED_API_KEY non configurata")
        print("=" * 70)
        print("\nPer scaricare dati macroeconomici:")
        print("1. Vai su https://research.stlouisfed.org/useraccount/apikey")
        print("2. Registrati e ottieni una API key gratuita")
        print("3. Aggiungi al file .env:")
        print("   FRED_API_KEY=tua_api_key")
        print("\n" + "=" * 70 + "\n")
        sys.exit(1)


def main():
    """Main entry point."""
    check_api_key()

    from ai_trading.data_ingestion.fred_client import (
        ingest_all_macro_series,
        get_economic_summary,
        get_yield_curve_status,
    )

    logger.info("=" * 70)
    logger.info("SCARICAMENTO DATI MACROECONOMICI DA FRED")
    logger.info("=" * 70)

    # Scarica tutte le serie macroeconomiche (ultimi 5 anni)
    results = ingest_all_macro_series()

    logger.info("\n" + "=" * 70)
    logger.info("RIEPILOGO DOWNLOAD")
    logger.info("=" * 70)

    for series_id, count in results.items():
        status = "✓" if count > 0 else "✗"
        logger.info(f"{status} {series_id}: {count} osservazioni")

    total = sum(results.values())
    logger.info(f"\nTotale: {total} osservazioni scaricate")

    # Mostra riepilogo economico attuale
    logger.info("\n" + "=" * 70)
    logger.info("SINTESI ECONOMICA ATTUALE")
    logger.info("=" * 70)

    try:
        summary = get_economic_summary()
        print("\n", summary.to_string(index=False))
    except Exception as e:
        logger.error(f"Errore nel recupero sintesi: {e}")

    # Mostra stato yield curve
    logger.info("\n" + "=" * 70)
    logger.info("CURVA DEI RENDIMENTI (Indicatore Recessione)")
    logger.info("=" * 70)

    try:
        yield_status = get_yield_curve_status()
        spread = yield_status.get("spread_10y_2y", "N/A")
        interpretation = yield_status.get("interpretation", "N/A")
        inverted = yield_status.get("inverted", False)

        print(f"\nSpread 10Y-2Y: {spread}%")
        print(f"Stato: {interpretation}")

        if inverted:
            print("\n⚠️  ATTENZIONE: Curva invertita! Storicamente precede recessioni.")
        else:
            print("\n✅ Curva normale (non invertita)")

    except Exception as e:
        logger.error(f"Errore nel recupero yield curve: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETATO")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
