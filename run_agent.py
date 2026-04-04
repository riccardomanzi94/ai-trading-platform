#!/usr/bin/env python3
"""AI Trader Agent - Entry Point

Run the AI Trader Agent in different modes:
- Interactive shell (default)
- Single command
- Web API server
- Scheduled daemon

Usage:
    python run_agent.py                    # Interactive shell
    python run_agent.py --mode auto        # Auto mode
    python run_agent.py --cmd "analyze SPY"  # Single command
    python run_agent.py --server           # Start API server
    python run_agent.py --daemon         # Run as daemon
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from ai_trading.agent import TraderAgent, AgentMode
from ai_trading.agent.commands import CommandProcessor, interactive_shell


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def run_single_command(agent: TraderAgent, command: str):
    """Run a single command and exit."""
    processor = CommandProcessor(agent)
    result = processor.process(command)
    print(result.message)
    return 0 if result.success else 1


def run_api_server(agent: TraderAgent, host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        import uvicorn

        app = FastAPI(title="AI Trader Agent API", version="1.0.0")

        class AnalyzeRequest(BaseModel):
            ticker: str

        class TradeRequest(BaseModel):
            ticker: str
            action: str  # BUY or SELL
            quantity: int | None = None

        class ChatRequest(BaseModel):
            message: str

        @app.get("/")
        def root():
            return {
                "name": "AI Trader Agent API",
                "version": "1.0.0",
                "mode": agent.mode.value,
                "timestamp": datetime.now().isoformat(),
            }

        @app.get("/status")
        def status():
            """Get agent status."""
            from ai_trading.execution import get_portfolio_summary
            try:
                df = get_portfolio_summary()
                portfolio = {
                    "total_value": float(df["current_value"].sum()),
                    "positions": len(df) - 1,  # Exclude cash
                }
            except:
                portfolio = {"total_value": 0, "positions": 0}

            return {
                "mode": agent.mode.value,
                "daily_trades": agent.daily_trades_count,
                "max_daily_trades": agent.max_daily_trades,
                "portfolio": portfolio,
            }

        @app.post("/analyze")
        def analyze(req: AnalyzeRequest):
            """Analyze a ticker."""
            try:
                analysis = agent.analyze(req.ticker)
                return {
                    "ticker": analysis.ticker,
                    "price": analysis.current_price,
                    "trend": analysis.trend,
                    "sentiment": analysis.sentiment.value,
                    "recommendation": analysis.recommendation,
                    "signals": [
                        {
                            "type": s.signal_type.value,
                            "strength": s.strength,
                        }
                        for s in analysis.signals
                    ],
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/trade")
        def trade(req: TradeRequest):
            """Execute a trade."""
            if agent.mode == AgentMode.MANUAL:
                raise HTTPException(
                    status_code=403,
                    detail="Agent is in MANUAL mode. Switch to semi or auto."
                )

            from ai_trading.signals import Signal, SignalType
            from ai_trading.risk_engine import apply_risk_to_signal
            from ai_trading.execution import execute_order

            try:
                signal = Signal(
                    time=datetime.now(),
                    ticker=req.ticker,
                    signal_type=SignalType(req.action.upper()),
                    strength=0.8,
                    price_at_signal=0,
                )
                order = apply_risk_to_signal(signal)

                if not order.approved:
                    return {
                        "success": False,
                        "reason": order.rejection_reason,
                    }

                if agent.mode == AgentMode.SEMI_AUTO:
                    return {
                        "success": True,
                        "status": "pending_approval",
                        "order": {
                            "ticker": order.ticker,
                            "side": order.signal_type,
                            "quantity": order.adjusted_quantity,
                        },
                    }

                # AUTO mode - execute
                execution = execute_order(order)
                if execution:
                    return {
                        "success": True,
                        "execution": {
                            "ticker": execution.ticker,
                            "side": execution.side,
                            "quantity": execution.quantity,
                            "price": execution.price,
                            "total_value": execution.total_value,
                        },
                    }

                return {"success": False, "reason": "Execution failed"}

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/scan")
        def scan():
            """Run trading cycle."""
            decisions = agent.run_trading_cycle()
            return {
                "decisions_count": len(decisions),
                "decisions": [
                    {
                        "ticker": d.ticker,
                        "action": d.action,
                        "confidence": d.confidence,
                        "reasoning": d.reasoning,
                    }
                    for d in decisions
                ],
            }

        @app.post("/chat")
        def chat(req: ChatRequest):
            """Chat with the agent."""
            response = agent.chat(req.message)
            return {"response": response}

        @app.post("/mode/{mode}")
        def set_mode(mode: str):
            """Change agent mode."""
            mode_map = {
                "manual": AgentMode.MANUAL,
                "semi": AgentMode.SEMI_AUTO,
                "auto": AgentMode.AUTO,
            }
            if mode not in mode_map:
                raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

            agent.set_mode(mode_map[mode])
            return {"mode": agent.mode.value}

        print(f"🚀 Starting API server on http://{host}:{port}")
        print(f"📚 API docs: http://{host}:{port}/docs")
        uvicorn.run(app, host=host, port=port, log_level="info")

    except ImportError:
        print("❌ FastAPI not installed. Run: pip install fastapi uvicorn")
        return 1


def run_daemon(agent: TraderAgent, interval_minutes: int = 30):
    """Run as scheduled daemon."""
    import time
    import schedule

    print(f"🤖 AI Trader Agent Daemon Started")
    print(f"⏰ Trading cycle every {interval_minutes} minutes")
    print(f"🎛️ Mode: {agent.mode.value}")
    print("Press Ctrl+C to stop\n")

    def job():
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running trading cycle...")
        decisions = agent.run_trading_cycle()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle complete: {len(decisions)} decisions")

    # Schedule job
    schedule.every(interval_minutes).minutes.do(job)

    # Run immediately
    job()

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Daemon stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Trader Agent - Your intelligent trading assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py                          # Interactive shell
  python run_agent.py --mode auto              # Start in auto mode
  python run_agent.py --cmd "analyze AAPL"     # Single command
  python run_agent.py --server --port 8000     # Start API server
  python run_agent.py --daemon --interval 60   # Run every hour
        """
    )

    parser.add_argument(
        "--mode",
        choices=["manual", "semi", "auto"],
        default="semi",
        help="Operating mode (default: semi)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold (default: 0.6)"
    )
    parser.add_argument(
        "--cmd",
        type=str,
        help="Execute single command and exit"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start API server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as scheduled daemon"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Daemon interval in minutes (default: 30)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Create agent
    mode_map = {
        "manual": AgentMode.MANUAL,
        "semi": AgentMode.SEMI_AUTO,
        "auto": AgentMode.AUTO,
    }
    agent = TraderAgent(
        mode=mode_map[args.mode],
        min_confidence=args.confidence,
    )

    # Execute based on mode
    if args.server:
        return run_api_server(agent, args.host, args.port)
    elif args.daemon:
        return run_daemon(agent, args.interval)
    elif args.cmd:
        return run_single_command(agent, args.cmd)
    else:
        interactive_shell()
        return 0


if __name__ == "__main__":
    sys.exit(main())
