"""Notification system for trading alerts.

Supports multiple channels:
- Telegram
- Slack
- Email
- Console (for testing)
"""

from __future__ import annotations

import logging
import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

from ai_trading.shared.config import config

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents an alert to be sent."""
    
    title: str
    message: str
    level: AlertLevel = AlertLevel.INFO
    ticker: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def format_text(self) -> str:
        """Format alert as plain text."""
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }
        
        emoji = level_emoji.get(self.level, "")
        ticker_str = f" [{self.ticker}]" if self.ticker else ""
        
        text = f"{emoji} {self.title}{ticker_str}\n\n{self.message}"
        
        if self.data:
            text += "\n\n📊 Details:\n"
            for key, value in self.data.items():
                if isinstance(value, float):
                    text += f"  • {key}: {value:.4f}\n"
                else:
                    text += f"  • {key}: {value}\n"
        
        text += f"\n🕐 {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        return text

    def format_html(self) -> str:
        """Format alert as HTML."""
        level_colors = {
            AlertLevel.INFO: "#17a2b8",
            AlertLevel.WARNING: "#ffc107",
            AlertLevel.ERROR: "#dc3545",
            AlertLevel.CRITICAL: "#dc3545",
        }
        
        color = level_colors.get(self.level, "#6c757d")
        ticker_str = f" [{self.ticker}]" if self.ticker else ""
        
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px;">
            <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px 5px 0 0;">
                <strong>{self.title}{ticker_str}</strong>
            </div>
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 0 0 5px 5px;">
                <p>{self.message}</p>
        """
        
        if self.data:
            html += "<h4>Details:</h4><ul>"
            for key, value in self.data.items():
                if isinstance(value, float):
                    html += f"<li><strong>{key}:</strong> {value:.4f}</li>"
                else:
                    html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        html += f"""
                <p style="color: #666; font-size: 0.9em;">
                    {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        </div>
        """
        return html


class Notifier(ABC):
    """Abstract base class for notifiers."""
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send an alert.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        pass

    def send_batch(self, alerts: List[Alert]) -> int:
        """Send multiple alerts.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Number of alerts sent successfully
        """
        return sum(1 for alert in alerts if self.send(alert))


class ConsoleNotifier(Notifier):
    """Console notifier for testing/debugging."""
    
    def send(self, alert: Alert) -> bool:
        """Print alert to console."""
        print("\n" + "=" * 50)
        print(alert.format_text())
        print("=" * 50 + "\n")
        return True


class TelegramNotifier(Notifier):
    """Telegram bot notifier."""
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
    
    def send(self, alert: Alert) -> bool:
        """Send alert via Telegram."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured, skipping")
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": alert.format_text(),
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            
            if response.status_code == 200:
                logger.info(f"Telegram alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Telegram error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False


class SlackNotifier(Notifier):
    """Slack webhook notifier."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        
        if not self.webhook_url:
            logger.warning("Slack webhook not configured")
    
    def send(self, alert: Alert) -> bool:
        """Send alert via Slack."""
        if not self.webhook_url:
            logger.warning("Slack not configured, skipping")
            return False
        
        level_colors = {
            AlertLevel.INFO: "#17a2b8",
            AlertLevel.WARNING: "#ffc107",
            AlertLevel.ERROR: "#dc3545",
            AlertLevel.CRITICAL: "#dc3545",
        }
        
        # Build Slack message with blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{alert.title}" + (f" [{alert.ticker}]" if alert.ticker else ""),
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message,
                }
            }
        ]
        
        if alert.data:
            fields = []
            for key, value in alert.data.items():
                if isinstance(value, float):
                    fields.append({"type": "mrkdwn", "text": f"*{key}:* {value:.4f}"})
                else:
                    fields.append({"type": "mrkdwn", "text": f"*{key}:* {value}"})
            
            blocks.append({
                "type": "section",
                "fields": fields[:10],  # Slack limit
            })
        
        try:
            response = requests.post(
                self.webhook_url,
                json={
                    "attachments": [{
                        "color": level_colors.get(alert.level),
                        "blocks": blocks,
                    }]
                },
                timeout=10,
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Slack error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False


class EmailNotifier(Notifier):
    """Email notifier via SMTP."""
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_email: Optional[str] = None,
        to_emails: Optional[List[str]] = None,
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.username = username or os.getenv("SMTP_USERNAME")
        self.password = password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("ALERT_FROM_EMAIL")
        
        to_emails_env = os.getenv("ALERT_TO_EMAILS", "")
        self.to_emails = to_emails or [e.strip() for e in to_emails_env.split(",") if e.strip()]
        
        if not all([self.username, self.password, self.from_email, self.to_emails]):
            logger.warning("Email credentials not fully configured")
    
    def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not all([self.username, self.password, self.from_email, self.to_emails]):
            logger.warning("Email not configured, skipping")
            return False
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[AI Trading] {alert.level.value.upper()}: {alert.title}"
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            
            # Attach both plain text and HTML
            msg.attach(MIMEText(alert.format_text(), "plain"))
            msg.attach(MIMEText(alert.format_html(), "html"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, self.to_emails, msg.as_string())
            
            logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False


class MultiNotifier(Notifier):
    """Sends alerts to multiple channels."""
    
    def __init__(self, notifiers: Optional[List[Notifier]] = None):
        if notifiers is None:
            # Default: all configured notifiers
            self.notifiers = []
            
            # Always add console
            self.notifiers.append(ConsoleNotifier())
            
            # Add others if configured
            if os.getenv("TELEGRAM_BOT_TOKEN"):
                self.notifiers.append(TelegramNotifier())
            if os.getenv("SLACK_WEBHOOK_URL"):
                self.notifiers.append(SlackNotifier())
            if os.getenv("SMTP_USERNAME"):
                self.notifiers.append(EmailNotifier())
        else:
            self.notifiers = notifiers
    
    def send(self, alert: Alert) -> bool:
        """Send alert to all configured notifiers."""
        results = [notifier.send(alert) for notifier in self.notifiers]
        return any(results)


# Convenience functions

_default_notifier: Optional[MultiNotifier] = None


def get_default_notifier() -> MultiNotifier:
    """Get or create default notifier."""
    global _default_notifier
    if _default_notifier is None:
        _default_notifier = MultiNotifier()
    return _default_notifier


def send_alert(
    title: str,
    message: str,
    level: AlertLevel = AlertLevel.INFO,
    ticker: Optional[str] = None,
    data: Optional[Dict] = None,
) -> bool:
    """Send an alert using the default notifier.
    
    Args:
        title: Alert title
        message: Alert message
        level: Severity level
        ticker: Related ticker symbol
        data: Additional data
        
    Returns:
        True if sent to at least one channel
    """
    alert = Alert(
        title=title,
        message=message,
        level=level,
        ticker=ticker,
        data=data or {},
    )
    return get_default_notifier().send(alert)


def send_signal_alert(signal) -> bool:
    """Send alert for a trading signal.
    
    Args:
        signal: Signal object
        
    Returns:
        True if sent
    """
    from ai_trading.signals import SignalType
    
    return send_alert(
        title=f"Trading Signal: {signal.signal_type.value}",
        message=f"New {signal.signal_type.value} signal generated for {signal.ticker}",
        level=AlertLevel.INFO,
        ticker=signal.ticker,
        data={
            "price": signal.price_at_signal,
            "strength": signal.strength,
            "time": str(signal.time),
        },
    )


def send_execution_alert(execution) -> bool:
    """Send alert for a trade execution.
    
    Args:
        execution: Execution object
        
    Returns:
        True if sent
    """
    return send_alert(
        title=f"Trade Executed: {execution.side} {execution.ticker}",
        message=f"Executed {execution.side} order for {execution.quantity} shares of {execution.ticker}",
        level=AlertLevel.INFO,
        ticker=execution.ticker,
        data={
            "quantity": execution.quantity,
            "price": execution.price,
            "commission": execution.commission,
            "total_value": execution.total_value,
        },
    )


def send_daily_summary(
    portfolio_value: float,
    daily_pnl: float,
    positions: Dict[str, Any],
    signals_today: int = 0,
    trades_today: int = 0,
) -> bool:
    """Send daily portfolio summary.
    
    Args:
        portfolio_value: Total portfolio value
        daily_pnl: Daily P&L
        positions: Current positions dict
        signals_today: Number of signals generated today
        trades_today: Number of trades executed today
        
    Returns:
        True if sent
    """
    pnl_pct = daily_pnl / (portfolio_value - daily_pnl) * 100 if portfolio_value != daily_pnl else 0
    level = AlertLevel.INFO if daily_pnl >= 0 else AlertLevel.WARNING
    
    positions_str = "\n".join(
        f"  • {ticker}: {info.get('quantity', 0)} shares"
        for ticker, info in positions.items()
        if ticker != "_CASH" and info.get("quantity", 0) > 0
    )
    
    return send_alert(
        title="📊 Daily Portfolio Summary",
        message=f"""
Portfolio Value: ${portfolio_value:,.2f}
Daily P&L: ${daily_pnl:+,.2f} ({pnl_pct:+.2f}%)

Signals Today: {signals_today}
Trades Today: {trades_today}

Positions:
{positions_str if positions_str else "  No open positions"}
        """.strip(),
        level=level,
        data={
            "portfolio_value": portfolio_value,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": pnl_pct,
        },
    )


def send_error_alert(error: Exception, context: str = "") -> bool:
    """Send alert for system errors.
    
    Args:
        error: Exception that occurred
        context: Additional context about where error occurred
        
    Returns:
        True if sent
    """
    return send_alert(
        title="System Error",
        message=f"An error occurred{' in ' + context if context else ''}: {str(error)}",
        level=AlertLevel.ERROR,
        data={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test console notifier
    alert = Alert(
        title="Test Alert",
        message="This is a test alert message.",
        level=AlertLevel.INFO,
        ticker="SPY",
        data={"price": 450.25, "volume": 1000000},
    )
    
    notifier = ConsoleNotifier()
    notifier.send(alert)
    
    # Test convenience function
    send_alert(
        title="Test Signal",
        message="Buy signal detected",
        ticker="AAPL",
        data={"strength": 0.85},
    )
