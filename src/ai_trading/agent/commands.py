"""Command processor for AI Trader Agent CLI.

Handles parsing and execution of user commands.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, List, Callable
from datetime import datetime

from ai_trading.agent.trader_agent import TraderAgent, AgentMode, get_agent


@dataclass
class CommandResult:
    """Result of command execution."""

    success: bool
    message: str
    data: Optional[dict] = None


class CommandProcessor:
    """Process CLI commands for the AI Trader Agent."""

    def __init__(self, agent: Optional[TraderAgent] = None):
        """Initialize command processor.

        Args:
            agent: TraderAgent instance (creates default if None)
        """
        self.agent = agent or get_agent()
        self._setup_commands()

    def _setup_commands(self):
        """Setup command handlers."""
        self.commands = {
            "help": self._cmd_help,
            "status": self._cmd_status,
            "portfolio": self._cmd_portfolio,
            "analyze": self._cmd_analyze,
            "scan": self._cmd_scan,
            "run": self._cmd_scan,
            "mode": self._cmd_mode,
            "buy": self._cmd_buy,
            "sell": self._cmd_sell,
            "history": self._cmd_history,
            "config": self._cmd_config,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
        }

    def process(self, user_input: str) -> CommandResult:
        """Process a user command.

        Args:
            user_input: Command string from user

        Returns:
            CommandResult with outcome
        """
        user_input = user_input.strip()
        if not user_input:
            return CommandResult(True, "")

        # Parse command
        parts = user_input.split()
        cmd = parts[0].lower()
        args = parts[1:]

        # Find and execute command
        if cmd in self.commands:
            try:
                return self.commands[cmd](args)
            except Exception as e:
                return CommandResult(False, f"❌ Error: {e}")

        # Try natural language processing
        response = self.agent.chat(user_input)
        return CommandResult(True, response)

    def _cmd_help(self, args: List[str]) -> CommandResult:
        """Show help."""
        help_text = """
🤖 **AI Trader Agent - Comandi Disponibili**

**Informazioni:**
  help              Mostra questo aiuto
  status            Stato dell'agent e mercato
  portfolio         Dettaglio portafoglio
  config            Mostra configurazione

**Analisi:**
  analyze <TICKER>   Analizza un ticker (es: analyze AAPL)
  scan              Scansiona tutti i ticker configurati
  run               Alias per scan

**Trading:**
  buy <TICKER> [qty]  Compra azioni (richiede approvazione in semi)
  sell <TICKER> [qty] Vendi azioni

**Gestione:**
  mode <MODE>      Cambia modalità (manual/semi/auto)
  history           Mostra storico trade

**Uscita:**
  exit / quit       Esci dall'agent

**Esempi:**
  > analyze SPY
  > mode auto
  > scan
  > buy AAPL 10

Digita il comando e premi Invio.
"""
        return CommandResult(True, help_text)

    def _cmd_status(self, args: List[str]) -> CommandResult:
        """Show agent status."""
        status = f"""
📊 **Stato AI Trader Agent**

🎛️ **Modalità:** {self.agent.mode.value.upper()}
🎯 **Confidence minima:** {self.agent.min_confidence:.0%}
📈 **Max trade/giorno:** {self.agent.max_daily_trades}
🕐 **Trade oggi:** {self.agent.daily_trades_count}

💼 **Portafoglio:**
"""
        # Add portfolio status
        from ai_trading.execution import get_portfolio_summary
        try:
            df = get_portfolio_summary()
            if not df.empty:
                total = df["current_value"].sum()
                cash_row = df[df["ticker"] == "_CASH"]
                cash = cash_row["current_value"].iloc[0] if not cash_row.empty else 0
                invested = total - cash
                status += f"   Valore totale: ${total:,.2f}\n"
                status += f"   Cash: ${cash:,.2f}\n"
                status += f"   Investito: ${invested:,.2f}\n"
            else:
                status += "   Portafoglio non inizializzato\n"
        except:
            status += "   Portafoglio non inizializzato\n"

        status += f"\n✅ Pronto per operare"
        return CommandResult(True, status)

    def _cmd_portfolio(self, args: List[str]) -> CommandResult:
        """Show portfolio details."""
        response = self.agent._format_portfolio_status()
        return CommandResult(True, response)

    def _cmd_analyze(self, args: List[str]) -> CommandResult:
        """Analyze a ticker."""
        if not args:
            return CommandResult(False, "❌ Usage: analyze <TICKER> (es: analyze AAPL)")

        ticker = args[0].upper()
        response = self.agent._format_analysis(ticker)
        return CommandResult(True, response)

    def _cmd_scan(self, args: List[str]) -> CommandResult:
        """Run trading cycle."""
        from ai_trading.shared.config import config

        tickers = args if args else config.trading.tickers

        print(f"🔍 Scansionando {len(tickers)} ticker...")
        decisions = self.agent.run_trading_cycle(tickers)

        response = self.agent._format_cycle_results(decisions)

        # In semi-auto mode, ask for approval
        if self.agent.mode == AgentMode.SEMI_AUTO and decisions:
            return CommandResult(
                True,
                response + "\n\n⚠️ **Modalità SEMI-AUTO**: Approvare i trade? (y/n per ognuno)",
                data={"pending_decisions": decisions}
            )

        return CommandResult(True, response)

    def _cmd_mode(self, args: List[str]) -> CommandResult:
        """Change operating mode."""
        if not args:
            return CommandResult(
                True,
                f"Modalità attuale: {self.agent.mode.value}\n"
                f"Usage: mode <manual|semi|auto>"
            )

        mode_str = args[0].lower()
        mode_map = {
            "manual": AgentMode.MANUAL,
            "semi": AgentMode.SEMI_AUTO,
            "auto": AgentMode.AUTO,
        }

        if mode_str not in mode_map:
            return CommandResult(False, f"❌ Modalità non valida: {mode_str}. Usa: manual, semi, auto")

        response = self.agent.set_mode(mode_map[mode_str])
        return CommandResult(True, response)

    def _cmd_buy(self, args: List[str]) -> CommandResult:
        """Execute buy order."""
        if not args:
            return CommandResult(False, "❌ Usage: buy <TICKER> [quantity]")

        ticker = args[0].upper()
        quantity = int(args[1]) if len(args) > 1 else None

        if self.agent.mode == AgentMode.MANUAL:
            return CommandResult(
                True,
                f"🔍 Analisi BUY per {ticker}:\n"
                + self.agent._format_analysis(ticker) +
                f"\n\n⚠️ Modalità MANUAL: usa mode semi/auto per eseguire"
            )

        if self.agent.mode == AgentMode.SEMI_AUTO:
            return CommandResult(
                True,
                f"🤖 Suggerimento BUY per {ticker}\n"
                f"\nConfermare l'ordine? (y/n)",
                data={"pending_action": "buy", "ticker": ticker, "quantity": quantity}
            )

        # AUTO mode - execute
        from ai_trading.signals import Signal, SignalType
        from ai_trading.risk_engine import apply_risk_to_signal
        from ai_trading.execution import execute_order
        from datetime import datetime

        try:
            signal = Signal(
                time=datetime.now(),
                ticker=ticker,
                signal_type=SignalType.BUY,
                strength=0.8,
                price_at_signal=0,
            )
            order = apply_risk_to_signal(signal)
            if order.approved:
                execution = execute_order(order)
                if execution:
                    return CommandResult(
                        True,
                        f"✅ Eseguito BUY {ticker}: {execution.quantity} @ ${execution.price:.2f}"
                    )
            return CommandResult(False, f"❌ Ordine non approvato: {order.rejection_reason}")
        except Exception as e:
            return CommandResult(False, f"❌ Errore: {e}")

    def _cmd_sell(self, args: List[str]) -> CommandResult:
        """Execute sell order."""
        if not args:
            return CommandResult(False, "❌ Usage: sell <TICKER> [quantity]")

        ticker = args[0].upper()
        quantity = int(args[1]) if len(args) > 1 else None

        if self.agent.mode == AgentMode.MANUAL:
            return CommandResult(
                True,
                f"🔍 Analisi SELL per {ticker}:\n"
                + self.agent._format_analysis(ticker) +
                f"\n\n⚠️ Modalità MANUAL: usa mode semi/auto per eseguire"
            )

        if self.agent.mode == AgentMode.SEMI_AUTO:
            return CommandResult(
                True,
                f"🤖 Suggerimento SELL per {ticker}\n"
                f"\nConfermare l'ordine? (y/n)",
                data={"pending_action": "sell", "ticker": ticker, "quantity": quantity}
            )

        # AUTO mode - execute
        from ai_trading.signals import Signal, SignalType
        from ai_trading.risk_engine import apply_risk_to_signal
        from ai_trading.execution import execute_order
        from datetime import datetime

        try:
            signal = Signal(
                time=datetime.now(),
                ticker=ticker,
                signal_type=SignalType.SELL,
                strength=0.8,
                price_at_signal=0,
            )
            order = apply_risk_to_signal(signal)
            if order.approved:
                execution = execute_order(order)
                if execution:
                    return CommandResult(
                        True,
                        f"✅ Eseguito SELL {ticker}: {execution.quantity} @ ${execution.price:.2f}"
                    )
            return CommandResult(False, f"❌ Ordine non approvato: {order.rejection_reason}")
        except Exception as e:
            return CommandResult(False, f"❌ Errore: {e}")

    def _cmd_history(self, args: List[str]) -> CommandResult:
        """Show trade history."""
        from ai_trading.execution import get_executions

        try:
            df = get_executions(limit=20)
            if df.empty:
                return CommandResult(True, "📊 Nessun trade eseguito ancora.")

            response = "📊 **Storico Trade Recenti**\n\n"
            for _, row in df.iterrows():
                emoji = "🟢" if row["side"] == "BUY" else "🔴"
                response += f"{emoji} {row['side']} {row['quantity']} {row['ticker']} @ ${row['price']:.2f}\n"
                response += f"   ${row['total_value']:.2f} ({row['time']})\n\n"

            return CommandResult(True, response)
        except Exception as e:
            return CommandResult(False, f"❌ Errore: {e}")

    def _cmd_config(self, args: List[str]) -> CommandResult:
        """Show configuration."""
        from ai_trading.shared.config import config

        cfg_text = f"""
⚙️ **Configurazione AI Trading Platform**

**Trading:**
• Tickers: {', '.join(config.trading.tickers)}
• Capitale iniziale: ${config.trading.initial_capital:,.2f}
• Max posizione: {config.trading.max_position_size:.1%}
• Max rischio: {config.trading.max_portfolio_risk:.1%}
• Commissione: {config.trading.commission_rate:.2%}
• Slippage: {config.trading.slippage_rate:.2%}

**Feature Engineering:**
• EMA Short: {config.features.ema_short}
• EMA Long: {config.features.ema_long}
• RSI Period: {config.features.rsi_period}
• ATR Period: {config.features.atr_period}

**Database:**
• URL: {config.database.url}

**Agent:**
• Modalità: {self.agent.mode.value}
• Min confidence: {self.agent.min_confidence:.0%}
• Max trade/giorno: {self.agent.max_daily_trades}
"""
        return CommandResult(True, cfg_text)

    def _cmd_exit(self, args: List[str]) -> CommandResult:
        """Exit command."""
        return CommandResult(True, "👋 Arrivederci! Happy trading!", data={"exit": True})


def interactive_shell():
    """Run interactive shell."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║      🤖 AI Trader Agent - Interactive Shell              ║
║                                                          ║
║   Type 'help' for commands or start chatting!           ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    processor = CommandProcessor()
    running = True

    while running:
        try:
            user_input = input("\n> ").strip()
            result = processor.process(user_input)

            if result.message:
                print(result.message)

            if result.data and result.data.get("exit"):
                running = False

        except KeyboardInterrupt:
            print("\n\n👋 Arrivederci!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    interactive_shell()
