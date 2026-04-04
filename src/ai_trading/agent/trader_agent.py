"""AI Trader Agent - Main implementation.

A professional trading agent that can analyze markets, execute trades,
and manage portfolio with natural language interaction.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Callable
import pandas as pd

from ai_trading.shared.config import config
from ai_trading.signals import (
    Signal, SignalType, generate_combined_signal,
    MultiStrategyConfig, CombineMethod,
)
from ai_trading.risk_engine import apply_risk_to_signal, get_portfolio_state
from ai_trading.execution import execute_order, get_portfolio_summary, get_executions
from ai_trading.alerts import send_alert, AlertLevel

logger = logging.getLogger(__name__)


class AgentMode(str, Enum):
    """Operating modes for the AI Trader Agent."""

    MANUAL = "manual"       # Only provides analysis, no execution
    SEMI_AUTO = "semi_auto" # Suggests trades, waits for approval
    AUTO = "auto"           # Executes trades autonomously


class MarketSentiment(str, Enum):
    """Market sentiment classification."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class TradeDecision:
    """A trade decision with reasoning."""

    ticker: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reasoning: str
    risk_assessment: str
    suggested_size: Optional[int] = None
    execution_plan: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioInsight:
    """Portfolio analysis and insights."""

    total_value: float
    cash_balance: float
    invested_value: float
    positions_count: int
    top_positions: List[Dict[str, Any]]
    risk_exposure: str
    recommendations: List[str]
    health_score: float  # 0-100


@dataclass
class MarketAnalysis:
    """Market analysis for a ticker."""

    ticker: str
    current_price: float
    price_change_1d: Optional[float]
    price_change_7d: Optional[float]
    volatility: float
    trend: str  # uptrend, downtrend, sideways
    support_levels: List[float]
    resistance_levels: List[float]
    signals: List[Signal]
    sentiment: MarketSentiment
    recommendation: str


class TraderAgent:
    """AI Trader Agent that acts as a professional trader.

    Usage:
        agent = TraderAgent(mode=AgentMode.SEMI_AUTO)

        # Analyze a stock
        analysis = agent.analyze("AAPL")

        # Get portfolio insights
        insights = agent.get_portfolio_insights()

        # Execute trading cycle
        decisions = agent.run_trading_cycle()
    """

    def __init__(
        self,
        mode: AgentMode = AgentMode.SEMI_AUTO,
        strategy_config: Optional[MultiStrategyConfig] = None,
        min_confidence: float = 0.6,
        max_daily_trades: int = 10,
    ):
        """Initialize the AI Trader Agent.

        Args:
            mode: Operating mode (manual/semi-auto/auto)
            strategy_config: Multi-strategy configuration
            min_confidence: Minimum confidence to execute trades
            max_daily_trades: Maximum trades per day
        """
        self.mode = mode
        self.strategy_config = strategy_config or MultiStrategyConfig(
            combine_method=CombineMethod.WEIGHTED,
            min_strength=0.4,
        )
        self.min_confidence = min_confidence
        self.max_daily_trades = max_daily_trades
        self.daily_trades_count = 0
        self.last_reset = datetime.now()

        # Trade history for learning
        self.trade_history: List[Dict] = []

        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
        }

        logger.info(f"AI Trader Agent initialized in {mode.value} mode")

    def _reset_daily_counter(self):
        """Reset daily trade counter if it's a new day."""
        if (datetime.now() - self.last_reset).days >= 1:
            self.daily_trades_count = 0
            self.last_reset = datetime.now()

    def set_mode(self, mode: AgentMode):
        """Change operating mode."""
        old_mode = self.mode
        self.mode = mode
        logger.info(f"Mode changed from {old_mode.value} to {mode.value}")

        mode_descriptions = {
            AgentMode.MANUAL: "I will only provide analysis and recommendations.",
            AgentMode.SEMI_AUTO: "I will suggest trades and wait for your approval.",
            AgentMode.AUTO: "I will execute trades autonomously based on my analysis.",
        }

        return f"🔄 Mode switched to **{mode.value.upper()}**. {mode_descriptions[mode]}"

    def analyze(self, ticker: str) -> MarketAnalysis:
        """Analyze a specific ticker like a professional trader.

        Args:
            ticker: Stock/ETF symbol to analyze

        Returns:
            MarketAnalysis with comprehensive insights
        """
        logger.info(f"Analyzing {ticker}")

        # Get current price data
        from ai_trading.signals.generate_signals import load_features_from_db
        from ai_trading.data_ingestion import ingest_ticker

        try:
            features_df = load_features_from_db(ticker, limit=30)
            if features_df.empty:
                # Try to fetch fresh data
                ingest_ticker(ticker, start_date="2024-01-01")
                from ai_trading.feature_store import build_features_for_ticker
                build_features_for_ticker(ticker)
                features_df = load_features_from_db(ticker, limit=30)
        except Exception as e:
            logger.warning(f"Could not load features for {ticker}: {e}")
            features_df = pd.DataFrame()

        # Get current price
        current_price = features_df["adj_close"].iloc[-1] if not features_df.empty else 0

        # Calculate price changes
        price_change_1d = None
        price_change_7d = None
        if len(features_df) >= 2:
            price_change_1d = (current_price - features_df["adj_close"].iloc[-2]) / features_df["adj_close"].iloc[-2] * 100
        if len(features_df) >= 7:
            price_change_7d = (current_price - features_df["adj_close"].iloc[-7]) / features_df["adj_close"].iloc[-7] * 100

        # Get volatility
        volatility = features_df["volatility_20"].iloc[-1] if not features_df.empty and "volatility_20" in features_df.columns else 0.20

        # Determine trend
        if len(features_df) >= 20:
            ema_short = features_df["ema_12"].iloc[-1]
            ema_long = features_df["ema_26"].iloc[-1]
            if ema_short > ema_long * 1.02:
                trend = "uptrend"
            elif ema_short < ema_long * 0.98:
                trend = "downtrend"
            else:
                trend = "sideways"
        else:
            trend = "unknown"

        # Calculate support/resistance (simplified)
        if not features_df.empty and len(features_df) >= 20:
            recent_lows = features_df["adj_close"].rolling(window=5).min().dropna()
            recent_highs = features_df["adj_close"].rolling(window=5).max().dropna()
            support_levels = [recent_lows.iloc[-1]] if not recent_lows.empty else []
            resistance_levels = [recent_highs.iloc[-1]] if not recent_highs.empty else []
        else:
            support_levels = []
            resistance_levels = []

        # Generate signals
        signals = []
        try:
            signal = generate_combined_signal(ticker, self.strategy_config)
            if signal:
                signals.append(signal)
        except Exception as e:
            logger.warning(f"Could not generate signal for {ticker}: {e}")

        # Determine sentiment
        if trend == "uptrend":
            sentiment = MarketSentiment.BULLISH
        elif trend == "downtrend":
            sentiment = MarketSentiment.BEARISH
        elif signals and any(s.signal_type == SignalType.BUY for s in signals):
            sentiment = MarketSentiment.BULLISH
        elif signals and any(s.signal_type == SignalType.SELL for s in signals):
            sentiment = MarketSentiment.BEARISH
        else:
            sentiment = MarketSentiment.NEUTRAL

        # Generate recommendation
        if signals and signals[0].strength >= self.min_confidence:
            sig = signals[0]
            if sig.signal_type == SignalType.BUY:
                recommendation = f"STRONG BUY - Signal strength {sig.strength:.0%}"
            elif sig.signal_type == SignalType.SELL:
                recommendation = f"STRONG SELL - Signal strength {sig.strength:.0%}"
            else:
                recommendation = "HOLD"
        elif trend == "uptrend":
            recommendation = "BUY - Uptrend detected"
        elif trend == "downtrend":
            recommendation = "SELL - Downtrend detected"
        else:
            recommendation = "HOLD - No clear signal"

        return MarketAnalysis(
            ticker=ticker,
            current_price=current_price,
            price_change_1d=price_change_1d,
            price_change_7d=price_change_7d,
            volatility=volatility,
            trend=trend,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            signals=signals,
            sentiment=sentiment,
            recommendation=recommendation,
        )

    def get_portfolio_insights(self) -> PortfolioInsight:
        """Get comprehensive portfolio analysis.

        Returns:
            PortfolioInsight with recommendations
        """
        logger.info("Generating portfolio insights")

        try:
            portfolio = get_portfolio_state()
            portfolio_df = get_portfolio_summary()
        except Exception as e:
            logger.error(f"Could not get portfolio: {e}")
            return PortfolioInsight(
                total_value=config.trading.initial_capital,
                cash_balance=config.trading.initial_capital,
                invested_value=0,
                positions_count=0,
                top_positions=[],
                risk_exposure="low",
                recommendations=["Initialize portfolio to start trading"],
                health_score=0,
            )

        cash = portfolio["cash"]
        total_value = portfolio["total_value"]
        invested = portfolio["total_invested"]
        positions = portfolio["positions"]

        # Top positions
        top_positions = [
            {"ticker": k, "value": v["current_value"], "quantity": v["quantity"]}
            for k, v in sorted(positions.items(), key=lambda x: x[1]["current_value"], reverse=True)
            if k != "_CASH" and v.get("quantity", 0) > 0
        ][:5]

        # Calculate risk metrics
        position_count = len([p for p in positions.values() if p.get("quantity", 0) > 0])
        cash_pct = cash / total_value if total_value > 0 else 1.0
        invested_pct = 1 - cash_pct

        if invested_pct > 0.9:
            risk_exposure = "high"
        elif invested_pct > 0.7:
            risk_exposure = "medium"
        else:
            risk_exposure = "low"

        # Health score (0-100)
        health_score = 50
        if cash_pct > 0.1:  # Has cash reserves
            health_score += 20
        if position_count >= 3:  # Diversified
            health_score += 15
        if invested_pct < 0.8:  # Not over-leveraged
            health_score += 15
        health_score = min(100, health_score)

        # Generate recommendations
        recommendations = []
        if cash_pct > 0.5:
            recommendations.append(f"High cash position ({cash_pct:.0%}). Consider deploying capital.")
        if invested_pct > 0.9:
            recommendations.append("Portfolio fully invested. Consider taking profits.")
        if position_count == 0:
            recommendations.append("No open positions. Ready to trade.")
        if position_count > 10:
            recommendations.append("Over-diversified. Consider consolidating positions.")

        return PortfolioInsight(
            total_value=total_value,
            cash_balance=cash,
            invested_value=invested,
            positions_count=position_count,
            top_positions=top_positions,
            risk_exposure=risk_exposure,
            recommendations=recommendations,
            health_score=health_score,
        )

    def make_trade_decision(self, ticker: str) -> Optional[TradeDecision]:
        """Make a trade decision for a ticker.

        Args:
            ticker: Stock/ETF to analyze

        Returns:
            TradeDecision or None if no action
        """
        self._reset_daily_counter()

        if self.daily_trades_count >= self.max_daily_trades:
            logger.info(f"Daily trade limit reached ({self.max_daily_trades})")
            return None

        # Analyze the ticker
        analysis = self.analyze(ticker)

        if not analysis.signals:
            return None

        signal = analysis.signals[0]

        # Skip weak signals
        if signal.strength < self.min_confidence:
            return None

        # Skip HOLD signals
        if signal.signal_type == SignalType.HOLD:
            return None

        # Check if we already have a position
        try:
            portfolio = get_portfolio_state()
            has_position = ticker in portfolio["positions"] and portfolio["positions"][ticker]["quantity"] > 0
        except:
            has_position = False

        # Skip BUY if already have position
        if signal.signal_type == SignalType.BUY and has_position:
            return None

        # Skip SELL if no position
        if signal.signal_type == SignalType.SELL and not has_position:
            return None

        # Build reasoning
        reasoning_parts = [
            f"Signal: {signal.signal_type.value} with {signal.strength:.0%} confidence",
            f"Trend: {analysis.trend}",
            f"Sentiment: {analysis.sentiment.value}",
        ]
        if analysis.price_change_1d is not None:
            reasoning_parts.append(f"24h change: {analysis.price_change_1d:+.2f}%")

        reasoning = " | ".join(reasoning_parts)

        # Risk assessment
        risk_assessment = f"Volatility: {analysis.volatility:.1%} | "
        if analysis.volatility > 0.3:
            risk_assessment += "HIGH volatility - reduced position size recommended"
        elif analysis.volatility > 0.2:
            risk_assessment += "MEDIUM volatility - normal position sizing"
        else:
            risk_assessment += "LOW volatility - standard position sizing"

        # Calculate suggested size
        try:
            from ai_trading.risk_engine.apply_risk import calculate_quantity
            position_size = min(config.trading.max_position_size, 0.05)  # Max 5% per trade
            suggested_size = calculate_quantity(
                position_size,
                portfolio["total_value"],
                analysis.current_price,
            )
        except:
            suggested_size = None

        return TradeDecision(
            ticker=ticker,
            action=signal.signal_type.value,
            confidence=signal.strength,
            reasoning=reasoning,
            risk_assessment=risk_assessment,
            suggested_size=suggested_size,
            execution_plan=f"Execute {signal.signal_type.value} for {ticker} at market",
        )

    def execute_decision(self, decision: TradeDecision) -> bool:
        """Execute a trade decision.

        Args:
            decision: TradeDecision to execute

        Returns:
            True if executed successfully
        """
        if self.mode == AgentMode.MANUAL:
            logger.info("In MANUAL mode - not executing")
            return False

        # In SEMI_AUTO mode, we would ask for approval (handled by caller)

        try:
            # Create signal
            signal = Signal(
                time=datetime.now(),
                ticker=decision.ticker,
                signal_type=SignalType(decision.action),
                strength=decision.confidence,
                price_at_signal=0,  # Will be fetched
            )

            # Apply risk
            order = apply_risk_to_signal(signal, save_to_db=True)

            if not order.approved:
                logger.warning(f"Order rejected: {order.rejection_reason}")
                return False

            # Execute
            execution = execute_order(order, save_to_db=True)

            if execution:
                self.daily_trades_count += 1
                self.performance_metrics["total_trades"] += 1

                # Send alert
                send_alert(
                    title=f"AI Agent Executed: {decision.action} {decision.ticker}",
                    message=f"Executed {decision.action} order based on analysis",
                    level=AlertLevel.INFO,
                    ticker=decision.ticker,
                    data={
                        "price": execution.price,
                        "quantity": execution.quantity,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                    },
                )

                logger.info(f"Executed {decision.action} {decision.ticker}")
                return True

        except Exception as e:
            logger.error(f"Failed to execute decision: {e}")
            send_alert(
                title=f"Execution Failed: {decision.ticker}",
                message=str(e),
                level=AlertLevel.ERROR,
                ticker=decision.ticker,
            )

        return False

    def run_trading_cycle(self, tickers: Optional[List[str]] = None) -> List[TradeDecision]:
        """Run a complete trading cycle.

        Args:
            tickers: List of tickers to analyze (default: from config)

        Returns:
            List of TradeDecisions made
        """
        if tickers is None:
            tickers = config.trading.tickers

        logger.info(f"Running trading cycle for {len(tickers)} tickers")

        decisions = []

        for ticker in tickers:
            try:
                decision = self.make_trade_decision(ticker)
                if decision:
                    decisions.append(decision)

                    # Execute if in auto mode
                    if self.mode == AgentMode.AUTO:
                        self.execute_decision(decision)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")

        logger.info(f"Trading cycle complete: {len(decisions)} decisions")
        return decisions

    def chat(self, message: str) -> str:
        """Process a natural language message.

        Args:
            message: User's message

        Returns:
            Agent's response
        """
        message_lower = message.lower().strip()

        # Help
        if any(word in message_lower for word in ["help", "aiuto", "comandi", "commands"]):
            return self._get_help_text()

        # Status
        if any(word in message_lower for word in ["status", "stato", "portfolio", "portafoglio"]):
            return self._format_portfolio_status()

        # Analyze ticker
        if message_lower.startswith(("analyze ", "analizza ", "valuta ", "check ")):
            ticker = message.split()[-1].upper()
            return self._format_analysis(ticker)

        # Trade
        if any(word in message_lower for word in ["buy", "compra", "sell", "vendi", "trade"]):
            return "🤖 I'm in **{}** mode. Use specific commands or switch to AUTO mode for autonomous trading.".format(self.mode.value)

        # Mode switch
        if "mode manual" in message_lower or "modo manuale" in message_lower:
            return self.set_mode(AgentMode.MANUAL)
        if "mode semi" in message_lower or "modo semi" in message_lower:
            return self.set_mode(AgentMode.SEMI_AUTO)
        if "mode auto" in message_lower or "modo auto" in message_lower:
            return self.set_mode(AgentMode.AUTO)

        # Run cycle
        if any(word in message_lower for word in ["run", "scan", "ciclo", "avvia"]):
            decisions = self.run_trading_cycle()
            return self._format_cycle_results(decisions)

        # Default response
        return self._get_welcome_message()

    def _get_welcome_message(self) -> str:
        """Get welcome message."""
        return """🤖 **AI Trader Agent**

Sono il tuo assistente di trading AI. Posso:

📊 **Analizzare** azioni e mercati
💼 **Gestire** il tuo portafoglio
⚡ **Eseguire** trade (in base alla modalità)
📈 **Monitorare** performance e rischi

**Comandi principali:**
• `analyze <TICKER>` - Analizza un'azione
• `status` / `portfolio` - Mostra il portafoglio
• `run` / `scan` - Esegui ciclo di trading
• `mode manual/semi/auto` - Cambia modalità
• `help` - Mostra aiuto completo

**Modalità attuale:** `{}`
""".format(self.mode.value.upper())

    def _get_help_text(self) -> str:
        """Get detailed help text."""
        return """📚 **AI Trader Agent - Guida Completa**

**Modalità operative:**
• `mode manual` - Solo analisi e consigli
• `mode semi` - Suggerisce, attende approvazione
• `mode auto` - Esegue autonomamente

**Comandi analisi:**
• `analyze AAPL` - Analisi completa di Apple
• `analyze SPY` - Analisi ETF S&P 500
• `status` - Stato portafoglio completo
• `portfolio` - Dettaglio posizioni

**Comandi trading:**
• `run` - Scansiona tutti i ticker configurati
• `scan` - Alias per run
• `cycle` - Esegui ciclo trading singolo

**Parametri configurati:**
• Confidence minima: {:.0%}
• Max trade/giorno: {}
• Tickers: {}

**Rischio:**
• Max posizione: {:.0%} del portafoglio
• Max rischio: {:.0%} per trade
""".format(
            self.min_confidence,
            self.max_daily_trades,
            ", ".join(config.trading.tickers[:5]) + "...",
            config.trading.max_position_size,
            config.trading.max_portfolio_risk,
        )

    def _format_analysis(self, ticker: str) -> str:
        """Format analysis as text."""
        try:
            analysis = self.analyze(ticker)
        except Exception as e:
            return f"❌ Error analyzing {ticker}: {e}"

        change_1d = f"{analysis.price_change_1d:+.2f}%" if analysis.price_change_1d else "N/A"
        change_7d = f"{analysis.price_change_7d:+.2f}%" if analysis.price_change_7d else "N/A"

        sentiment_emoji = {
            MarketSentiment.BULLISH: "🟢",
            MarketSentiment.BEARISH: "🔴",
            MarketSentiment.NEUTRAL: "⚪",
            MarketSentiment.MIXED: "🟡",
        }

        signal_text = ""
        if analysis.signals:
            for sig in analysis.signals:
                signal_text += f"\n   • {sig.signal_type.value} (forza: {sig.strength:.0%})"
        else:
            signal_text = "\n   • Nessun segnale"

        return f"""📊 **Analisi {ticker}**

💰 **Prezzo:** ${analysis.current_price:.2f}
📈 **Variazione:** 1D: {change_1d} | 7D: {change_7d}
📉 **Trend:** {analysis.trend.upper()}
{sentiment_emoji[analysis.sentiment]} **Sentiment:** {analysis.sentiment.value.upper()}
📊 **Volatilità:** {analysis.volatility:.1%}

🔔 **Segnali:**{signal_text}

🎯 **Raccomandazione:** {analysis.recommendation}
"""

    def _format_portfolio_status(self) -> str:
        """Format portfolio status as text."""
        try:
            insights = self.get_portfolio_insights()
        except Exception as e:
            return f"❌ Error getting portfolio: {e}"

        # Health emoji
        if insights.health_score >= 80:
            health_emoji = "🟢"
        elif insights.health_score >= 60:
            health_emoji = "🟡"
        else:
            health_emoji = "🔴"

        positions_text = ""
        if insights.top_positions:
            for pos in insights.top_positions:
                positions_text += f"\n   • {pos['ticker']}: {pos['quantity']} azioni (${pos['value']:,.2f})"
        else:
            positions_text = "\n   • Nessuna posizione aperta"

        recs_text = ""
        if insights.recommendations:
            for rec in insights.recommendations:
                recs_text += f"\n   • {rec}"
        else:
            recs_text = "\n   • Nessuna raccomandazione specifica"

        return f"""💼 **Stato Portafoglio**

💰 **Valore Totale:** ${insights.total_value:,.2f}
💵 **Cash:** ${insights.cash_balance:,.2f}
📊 **Investito:** ${insights.invested_value:,.2f}
📈 **Posizioni:** {insights.positions_count}

{health_emoji} **Health Score:** {insights.health_score}/100
⚠️ **Esposizione:** {insights.risk_exposure.upper()}

📋 **Top Posizioni:**{positions_text}

💡 **Raccomandazioni:**{recs_text}
"""

    def _format_cycle_results(self, decisions: List[TradeDecision]) -> str:
        """Format trading cycle results."""
        if not decisions:
            return "🔍 **Ciclo Trading Completato**\n\nNessuna opportunità trovata."

        result = f"🔍 **Ciclo Trading Completato**\n\nTrovate {len(decisions)} opportunità:\n"

        for i, dec in enumerate(decisions, 1):
            action_emoji = "🟢" if dec.action == "BUY" else "🔴"
            result += f"\n{i}. {action_emoji} **{dec.action}** {dec.ticker}"
            result += f"\n   Confidenza: {dec.confidence:.0%}"
            result += f"\n   {dec.reasoning}"
            if self.mode == AgentMode.SEMI_AUTO:
                result += "\n   ⚠️ *Attendi approvazione*"

        return result


# Singleton instance
_agent_instance: Optional[TraderAgent] = None


def get_agent(
    mode: Optional[AgentMode] = None,
    min_confidence: float = 0.6,
) -> TraderAgent:
    """Get or create singleton agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TraderAgent(
            mode=mode or AgentMode.SEMI_AUTO,
            min_confidence=min_confidence,
        )
    elif mode is not None:
        _agent_instance.mode = mode
    return _agent_instance
