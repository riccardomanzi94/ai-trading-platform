"""Alpaca Broker Integration.

Provides integration with Alpaca Markets API for:
- Paper trading
- Live trading
- Account management
- Order execution
- Position tracking
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

from ai_trading.shared.config import config as app_config
from ai_trading.alerts import send_alert, send_execution_alert, AlertLevel

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""
    NEW = "new"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class TimeInForce(str, Enum):
    """Time in force."""
    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    
    api_key: str = field(
        default_factory=lambda: os.getenv("ALPACA_API_KEY", "")
    )
    api_secret: str = field(
        default_factory=lambda: os.getenv("ALPACA_API_SECRET", "")
    )
    paper: bool = field(
        default_factory=lambda: os.getenv("ALPACA_PAPER", "true").lower() == "true"
    )
    
    @property
    def base_url(self) -> str:
        if self.paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"
    
    @property
    def data_url(self) -> str:
        return "https://data.alpaca.markets"


@dataclass
class Position:
    """Represents a position in the account."""
    
    ticker: str
    quantity: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str  # "long" or "short"


@dataclass
class Order:
    """Represents an order."""
    
    id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: int
    filled_avg_price: Optional[float]
    submitted_at: datetime
    filled_at: Optional[datetime]
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.PENDING_NEW,
            OrderStatus.PARTIALLY_FILLED,
        ]


class AlpacaBroker:
    """Alpaca API broker client."""
    
    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.config = config or AlpacaConfig()
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.api_secret,
        })
        
        if not self.config.api_key or not self.config.api_secret:
            logger.warning("Alpaca API credentials not configured")
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Dict:
        """Make API request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json: JSON body
            
        Returns:
            Response JSON
        """
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Alpaca API error: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Alpaca request failed: {e}")
            raise
    
    # Account methods
    
    def get_account(self) -> Dict:
        """Get account information.
        
        Returns:
            Account info dict
        """
        return self._request("GET", "/v2/account")
    
    def get_buying_power(self) -> float:
        """Get available buying power.
        
        Returns:
            Buying power amount
        """
        account = self.get_account()
        return float(account.get("buying_power", 0))
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value.
        
        Returns:
            Portfolio value
        """
        account = self.get_account()
        return float(account.get("portfolio_value", 0))
    
    def get_cash(self) -> float:
        """Get cash balance.
        
        Returns:
            Cash balance
        """
        account = self.get_account()
        return float(account.get("cash", 0))
    
    # Position methods
    
    def get_positions(self) -> List[Position]:
        """Get all open positions.
        
        Returns:
            List of Position objects
        """
        response = self._request("GET", "/v2/positions")
        
        positions = []
        for pos in response:
            positions.append(Position(
                ticker=pos["symbol"],
                quantity=int(pos["qty"]),
                avg_entry_price=float(pos["avg_entry_price"]),
                current_price=float(pos["current_price"]),
                market_value=float(pos["market_value"]),
                unrealized_pnl=float(pos["unrealized_pl"]),
                unrealized_pnl_pct=float(pos["unrealized_plpc"]) * 100,
                side=pos["side"],
            ))
        
        return positions
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Position or None if not held
        """
        try:
            pos = self._request("GET", f"/v2/positions/{ticker}")
            return Position(
                ticker=pos["symbol"],
                quantity=int(pos["qty"]),
                avg_entry_price=float(pos["avg_entry_price"]),
                current_price=float(pos["current_price"]),
                market_value=float(pos["market_value"]),
                unrealized_pnl=float(pos["unrealized_pl"]),
                unrealized_pnl_pct=float(pos["unrealized_plpc"]) * 100,
                side=pos["side"],
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def close_position(self, ticker: str) -> Order:
        """Close entire position in a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Order object for the closing trade
        """
        response = self._request("DELETE", f"/v2/positions/{ticker}")
        return self._parse_order(response)
    
    def close_all_positions(self) -> List[Order]:
        """Close all open positions.
        
        Returns:
            List of closing orders
        """
        response = self._request("DELETE", "/v2/positions")
        return [self._parse_order(o) for o in response]
    
    # Order methods
    
    def submit_order(
        self,
        ticker: str,
        quantity: int,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> Order:
        """Submit an order.
        
        Args:
            ticker: Stock symbol
            quantity: Number of shares
            side: Buy or sell
            order_type: Market, limit, etc.
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Order duration
            
        Returns:
            Order object
        """
        order_data = {
            "symbol": ticker,
            "qty": str(quantity),
            "side": side.value,
            "type": order_type.value,
            "time_in_force": time_in_force.value,
        }
        
        if limit_price is not None:
            order_data["limit_price"] = str(limit_price)
        if stop_price is not None:
            order_data["stop_price"] = str(stop_price)
        
        response = self._request("POST", "/v2/orders", json=order_data)
        order = self._parse_order(response)
        
        logger.info(
            f"Order submitted: {side.value} {quantity} {ticker} "
            f"({order_type.value}) - ID: {order.id}"
        )
        
        return order
    
    def buy(
        self,
        ticker: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Order:
        """Submit a buy order.
        
        Args:
            ticker: Stock symbol
            quantity: Number of shares
            order_type: Order type
            limit_price: Limit price
            
        Returns:
            Order object
        """
        return self.submit_order(
            ticker=ticker,
            quantity=quantity,
            side=OrderSide.BUY,
            order_type=order_type,
            limit_price=limit_price,
        )
    
    def sell(
        self,
        ticker: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Order:
        """Submit a sell order.
        
        Args:
            ticker: Stock symbol
            quantity: Number of shares
            order_type: Order type
            limit_price: Limit price
            
        Returns:
            Order object
        """
        return self.submit_order(
            ticker=ticker,
            quantity=quantity,
            side=OrderSide.SELL,
            order_type=order_type,
            limit_price=limit_price,
        )
    
    def get_order(self, order_id: str) -> Order:
        """Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object
        """
        response = self._request("GET", f"/v2/orders/{order_id}")
        return self._parse_order(response)
    
    def get_orders(
        self,
        status: Optional[str] = "open",
        limit: int = 50,
    ) -> List[Order]:
        """Get orders.
        
        Args:
            status: Filter by status ("open", "closed", "all")
            limit: Maximum number of orders
            
        Returns:
            List of Order objects
        """
        params = {"limit": limit}
        if status:
            params["status"] = status
            
        response = self._request("GET", "/v2/orders", params=params)
        return [self._parse_order(o) for o in response]
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if cancelled
        """
        try:
            self._request("DELETE", f"/v2/orders/{order_id}")
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        response = self._request("DELETE", "/v2/orders")
        count = len(response) if isinstance(response, list) else 0
        logger.info(f"Cancelled {count} orders")
        return count
    
    def _parse_order(self, data: Dict) -> Order:
        """Parse order from API response."""
        filled_at = None
        if data.get("filled_at"):
            filled_at = datetime.fromisoformat(data["filled_at"].replace("Z", "+00:00"))
        
        return Order(
            id=data["id"],
            ticker=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["type"]),
            quantity=int(data["qty"]),
            limit_price=float(data["limit_price"]) if data.get("limit_price") else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") else None,
            status=OrderStatus(data["status"]),
            filled_quantity=int(data.get("filled_qty", 0)),
            filled_avg_price=float(data["filled_avg_price"]) if data.get("filled_avg_price") else None,
            submitted_at=datetime.fromisoformat(data["submitted_at"].replace("Z", "+00:00")),
            filled_at=filled_at,
        )
    
    # Market data methods
    
    def get_latest_quote(self, ticker: str) -> Dict:
        """Get latest quote for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Quote data
        """
        url = f"{self.config.data_url}/v2/stocks/{ticker}/quotes/latest"
        response = self._session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()["quote"]
    
    def get_latest_trade(self, ticker: str) -> Dict:
        """Get latest trade for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Trade data
        """
        url = f"{self.config.data_url}/v2/stocks/{ticker}/trades/latest"
        response = self._session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()["trade"]
    
    def get_bars(
        self,
        ticker: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get historical bars.
        
        Args:
            ticker: Stock symbol
            timeframe: Bar timeframe (e.g., "1Min", "1Hour", "1Day")
            start: Start datetime (ISO format)
            end: End datetime (ISO format)
            limit: Maximum number of bars
            
        Returns:
            List of bar data
        """
        url = f"{self.config.data_url}/v2/stocks/{ticker}/bars"
        params = {"timeframe": timeframe, "limit": limit}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("bars", [])
    
    # Utility methods
    
    def is_market_open(self) -> bool:
        """Check if market is currently open.
        
        Returns:
            True if market is open
        """
        response = self._request("GET", "/v2/clock")
        return response.get("is_open", False)
    
    def get_clock(self) -> Dict:
        """Get market clock info.
        
        Returns:
            Clock data including next open/close times
        """
        return self._request("GET", "/v2/clock")
    
    def execute_signal(
        self,
        signal,
        quantity: Optional[int] = None,
    ) -> Optional[Order]:
        """Execute a trading signal through the broker.
        
        Args:
            signal: Signal object from ai_trading.signals
            quantity: Override quantity (uses risk-adjusted if None)
            
        Returns:
            Order if executed, None otherwise
        """
        from ai_trading.signals import SignalType
        
        if quantity is None:
            # Calculate quantity based on position sizing
            buying_power = self.get_buying_power()
            max_position = buying_power * app_config.trading.max_position_size
            
            quote = self.get_latest_quote(signal.ticker)
            price = float(quote.get("ap", signal.price_at_signal))  # Ask price
            
            quantity = int(max_position / price)
        
        if quantity <= 0:
            logger.warning(f"Cannot execute signal: quantity is {quantity}")
            return None
        
        try:
            if signal.signal_type == SignalType.BUY:
                order = self.buy(signal.ticker, quantity)
            elif signal.signal_type == SignalType.SELL:
                # Check if we have position to sell
                position = self.get_position(signal.ticker)
                if not position:
                    logger.warning(f"No position to sell for {signal.ticker}")
                    return None
                quantity = min(quantity, position.quantity)
                order = self.sell(signal.ticker, quantity)
            else:
                return None
            
            # Send alert
            send_alert(
                title=f"Order Submitted: {signal.signal_type.value} {signal.ticker}",
                message=f"Order ID: {order.id}\nQuantity: {quantity}",
                level=AlertLevel.INFO,
                ticker=signal.ticker,
                data={"order_id": order.id, "quantity": quantity},
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            send_alert(
                title="Order Failed",
                message=str(e),
                level=AlertLevel.ERROR,
                ticker=signal.ticker,
            )
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test connection (requires API keys)
    broker = AlpacaBroker()
    
    if broker.config.api_key:
        try:
            account = broker.get_account()
            print(f"Account Status: {account.get('status')}")
            print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
            print(f"Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
            
            clock = broker.get_clock()
            print(f"Market Open: {clock.get('is_open')}")
            
        except Exception as e:
            print(f"Connection failed: {e}")
    else:
        print("Alpaca API keys not configured")
        print("Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
