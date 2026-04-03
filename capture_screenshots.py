#!/usr/bin/env python3
"""Cattura screenshot della dashboard per il README."""

import asyncio
import os
from playwright.async_api import async_playwright

# Directory per le immagini
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "docs", "images")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

DASHBOARD_URL = "http://localhost:8505"

# Tab da catturare con i loro selettori
TABS = [
    ("overview", "📊 Panoramica", 0),
    ("signals", "📡 Segnali", 1),
    ("ml_predictions", "🤖 Previsioni ML", 2),
    ("strategy_lab", "🔬 Lab Strategie", 3),
    ("risk_center", "⚠️ Centro Rischio", 4),
    ("backtest", "📉 Backtest", 5),
]

async def capture_screenshots():
    """Cattura screenshot di tutte le tab della dashboard."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1400, "height": 900},
            device_scale_factor=1.5
        )
        page = await context.new_page()
        
        print(f"🌐 Connessione a {DASHBOARD_URL}...")
        
        try:
            await page.goto(DASHBOARD_URL, wait_until="networkidle", timeout=60000)
            print("✓ Dashboard caricata")
            
            # Attendi che Streamlit sia completamente pronto
            await page.wait_for_timeout(5000)
            
            # Screenshot principale (prima tab - Panoramica)
            screenshot_path = os.path.join(SCREENSHOTS_DIR, "dashboard_overview.png")
            await page.screenshot(path=screenshot_path, full_page=False)
            print(f"✓ Salvato: dashboard_overview.png")
            
            # Prova diversi selettori per le tab di Streamlit
            tab_selectors = [
                'button[data-baseweb="tab"]',
                '[role="tab"]',
                '.stTabs button',
                'div[data-testid="stHorizontalBlock"] button'
            ]
            
            tabs = None
            for selector in tab_selectors:
                tabs = page.locator(selector)
                count = await tabs.count()
                if count > 0:
                    print(f"📋 Trovate {count} tab con selettore: {selector}")
                    break
            
            if tabs and await tabs.count() > 0:
                tab_count = await tabs.count()
                for i, (name, label, idx) in enumerate(TABS):
                    if idx < tab_count:
                        try:
                            # Clicca sulla tab
                            await tabs.nth(idx).click()
                            await page.wait_for_timeout(2500)  # Attendi rendering
                            
                            # Screenshot della tab
                            screenshot_path = os.path.join(SCREENSHOTS_DIR, f"dashboard_{name}.png")
                            await page.screenshot(path=screenshot_path, full_page=False)
                            print(f"✓ Salvato: dashboard_{name}.png")
                        except Exception as e:
                            print(f"⚠️ Errore tab {name}: {e}")
            else:
                print("⚠️ Tab non trovate, cattureremo solo l'overview")
                # Fai uno screenshot fullpage come alternativa
                full_screenshot = os.path.join(SCREENSHOTS_DIR, "dashboard_full.png")
                await page.screenshot(path=full_screenshot, full_page=True)
                print(f"✓ Salvato: dashboard_full.png")
            
            # Screenshot della sidebar (con controlli)
            sidebar_screenshot = os.path.join(SCREENSHOTS_DIR, "dashboard_sidebar.png")
            
            # Trova la sidebar
            sidebar = page.locator('[data-testid="stSidebar"]')
            if await sidebar.count() > 0:
                await sidebar.screenshot(path=sidebar_screenshot)
                print(f"✓ Salvato: dashboard_sidebar.png")
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            # Prova comunque a fare uno screenshot
            error_screenshot = os.path.join(SCREENSHOTS_DIR, "dashboard_error.png")
            await page.screenshot(path=error_screenshot)
            print(f"Screenshot di debug salvato: {error_screenshot}")
        
        await browser.close()
        
    print(f"\n✅ Screenshot salvati in: {SCREENSHOTS_DIR}")

if __name__ == "__main__":
    asyncio.run(capture_screenshots())
