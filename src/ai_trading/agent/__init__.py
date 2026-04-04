"""AI Trader Agent - An intelligent trading assistant.

The AI Trader Agent acts as a professional trader that can:
- Analyze markets and generate insights
- Execute trades based on strategies and ML predictions
- Manage risk and portfolio
- Provide natural language interaction
- Operate in different modes (manual, semi-auto, auto)
"""

from ai_trading.agent.trader_agent import TraderAgent, AgentMode
from ai_trading.agent.commands import CommandProcessor

__all__ = ["TraderAgent", "AgentMode", "CommandProcessor"]
