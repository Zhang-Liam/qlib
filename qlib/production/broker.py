# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Broker integration for live trading.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
try:
    import pandas as pd
except ImportError:
    pd = None
from datetime import datetime, timedelta
import random

from .config import BrokerConfig


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order representation."""
    
    symbol: str
    quantity: int
    side: OrderSide
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: Optional[float] = None
    commission: float = 0.0
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


@dataclass
class Position:
    """Position representation."""
    
    symbol: str
    quantity: int
    average_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime


@dataclass
class Account:
    """Account information."""
    
    account_id: str
    cash: float
    buying_power: float
    equity: float
    margin_balance: float
    positions: List[Position]
    timestamp: datetime


class BrokerConnector(ABC):
    """Abstract base class for broker connectors."""
    
    def __init__(self, config: BrokerConfig):
        """
        Initialize broker connector.
        
        Parameters
        ----------
        config : BrokerConfig
            Broker configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.connected = False
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.account: Optional[Account] = None
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Account:
        """Get account information."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Place an order and return order ID."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        pass
    
    @abstractmethod
    def get_market_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, interval: str = "1d"):
        """Get historical price data."""
        pass


class MockBrokerConnector(BrokerConnector):
    """Mock broker connector for testing and development."""
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self._order_id_counter = 1
        self._mock_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'SPY': 450.0,
            'QQQ': 380.0,
            'TSLA': 250.0,
            'AMZN': 3300.0,
            'NVDA': 500.0
        }
        self._mock_account = Account(
            account_id="MOCK_ACCOUNT_001",
            cash=100000.0,
            buying_power=100000.0,
            equity=100000.0,
            margin_balance=100000.0,
            positions=[],
            timestamp=datetime.now()
        )
    
    def connect(self) -> bool:
        """Connect to mock broker (always succeeds)."""
        self.connected = True
        self.logger.info("Connected to Mock Broker")
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from mock broker."""
        self.connected = False
        self.logger.info("Disconnected from Mock Broker")
        return True
    
    def is_connected(self) -> bool:
        """Check if connected to mock broker."""
        return self.connected
    
    def get_account_info(self) -> Account:
        """Get mock account information."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        # Update account with current positions
        total_positions_value = sum(pos.market_value for pos in self._mock_account.positions)
        self._mock_account.equity = self._mock_account.cash + total_positions_value
        self._mock_account.buying_power = self._mock_account.cash
        self._mock_account.timestamp = datetime.now()
        
        return self._mock_account
    
    def get_positions(self) -> Dict[str, Position]:
        """Get mock positions."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        positions_dict = {}
        for position in self._mock_account.positions:
            # Update market value and PnL
            current_price = self.get_market_price(position.symbol)
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = position.market_value - (position.quantity * position.average_price)
            position.timestamp = datetime.now()
            positions_dict[position.symbol] = position
        
        return positions_dict
    
    def place_order(self, order: Order) -> str:
        """Place a mock order."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        order_id = f"MOCK_ORDER_{self._order_id_counter:06d}"
        self._order_id_counter += 1
        
        order.order_id = order_id
        order.created_at = datetime.now()
        order.status = OrderStatus.SUBMITTED
        
        # Simulate order execution
        current_price = self.get_market_price(order.symbol)
        order.average_price = current_price
        order.filled_quantity = order.quantity
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        
        # Update account and positions
        self._update_account_with_order(order)
        
        self.orders[order_id] = order
        self.logger.info(f"Mock order placed: {order_id} - {order.symbol} {order.side.value} {order.quantity}")
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a mock order."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            self.logger.info(f"Mock order cancelled: {order_id}")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get mock order status."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.REJECTED
    
    def get_market_price(self, symbol: str) -> float:
        """Get mock market price with some randomness."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        base_price = self._mock_prices.get(symbol, 100.0)
        # Add some random price movement (±2%)
        price_change = random.uniform(-0.02, 0.02)
        new_price = base_price * (1 + price_change)
        self._mock_prices[symbol] = new_price
        
        return round(new_price, 2)
    
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, interval: str = "1d"):
        """Get mock historical data."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        if pd is None:
            raise ImportError("pandas is required for historical data")
        
        # Generate mock historical data
        base_price = self._mock_prices.get(symbol, 100.0)
        dates = pd.date_range(start=start_date, end=end_date, freq=interval)
        
        data = []
        current_price = base_price
        
        for date in dates:
            # Simulate price movement
            price_change = random.uniform(-0.05, 0.05)
            current_price *= (1 + price_change)
            
            data.append({
                'date': date,
                'open': current_price * random.uniform(0.98, 1.02),
                'high': current_price * random.uniform(1.0, 1.05),
                'low': current_price * random.uniform(0.95, 1.0),
                'close': current_price,
                'volume': random.randint(1000000, 10000000)
            })
        
        return pd.DataFrame(data)
    
    def _update_account_with_order(self, order: Order):
        """Update account and positions with executed order."""
        if order.average_price is None:
            raise ValueError("Order average price cannot be None")
            
        order_value = order.quantity * order.average_price
        commission = order_value * 0.001  # 0.1% commission
        
        if order.side == OrderSide.BUY:
            # Deduct cash for purchase
            self._mock_account.cash -= (order_value + commission)
            
            # Update or create position
            existing_position = None
            for pos in self._mock_account.positions:
                if pos.symbol == order.symbol:
                    existing_position = pos
                    break
            
            if existing_position:
                # Update existing position
                total_quantity = existing_position.quantity + order.quantity
                total_cost = (existing_position.quantity * existing_position.average_price) + order_value
                existing_position.average_price = total_cost / total_quantity
                existing_position.quantity = total_quantity
                existing_position.market_value = total_quantity * order.average_price
            else:
                # Create new position
                new_position = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    average_price=order.average_price,
                    market_value=order_value,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=datetime.now()
                )
                self._mock_account.positions.append(new_position)
        else:
            # SELL order
            # Add cash from sale
            self._mock_account.cash += (order_value - commission)
            
            # Update position
            for pos in self._mock_account.positions:
                if pos.symbol == order.symbol:
                    if pos.quantity >= order.quantity:
                        # Calculate realized PnL
                        realized_pnl = (order.average_price - pos.average_price) * order.quantity
                        pos.realized_pnl += realized_pnl
                        pos.quantity -= order.quantity
                        
                        if pos.quantity == 0:
                            # Remove position if fully sold
                            self._mock_account.positions.remove(pos)
                        else:
                            pos.market_value = pos.quantity * order.average_price
                        break


class InteractiveBrokersConnector(BrokerConnector):
    """Interactive Brokers connector implementation."""
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self.ib = None
        self._order_id_counter = 1
    
    def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway."""
        try:
            # Import ib_insync here to avoid dependency issues
            from ib_insync import IB, Ticker, Contract, Order as IBOrder
            
            self.ib = IB()
            self.ib.connect(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout
            )
            
            if self.ib.isConnected():
                self.connected = True
                self.logger.info("Connected to Interactive Brokers")
                
                # Set up event handlers
                self.ib.orderStatusEvent += self._on_order_status
                self.ib.positionEvent += self._on_position_update
                self.ib.accountValueEvent += self._on_account_update
                
                return True
            else:
                self.logger.error("Failed to connect to Interactive Brokers")
                return False
                
        except ImportError:
            self.logger.error("ib_insync not installed. Install with: pip install ib_insync")
            return False
        except Exception as e:
            self.logger.error(f"Error connecting to Interactive Brokers: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Interactive Brokers."""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
                self.connected = False
                self.logger.info("Disconnected from Interactive Brokers")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error disconnecting: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Interactive Brokers."""
        return self.connected and self.ib and self.ib.isConnected()
    
    def get_account_info(self) -> Account:
        """Get account information."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            # Get account summary
            account_values = self.ib.accountSummary()
            
            # Extract key values
            cash = 0.0
            buying_power = 0.0
            equity = 0.0
            margin_balance = 0.0
            
            for value in account_values:
                if value.tag == "AvailableFunds" and value.currency == "USD":
                    cash = float(value.value)
                elif value.tag == "BuyingPower" and value.currency == "USD":
                    buying_power = float(value.value)
                elif value.tag == "NetLiquidation" and value.currency == "USD":
                    equity = float(value.value)
                elif value.tag == "TotalCashValue" and value.currency == "USD":
                    margin_balance = float(value.value)
            
            # Get positions
            positions = self.get_positions()
            
            self.account = Account(
                account_id=self.config.ib_account or "default",
                cash=cash,
                buying_power=buying_power,
                equity=equity,
                margin_balance=margin_balance,
                positions=list(positions.values()),
                timestamp=datetime.now()
            )
            
            return self.account
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            ib_positions = self.ib.positions()
            positions = {}
            
            for pos in ib_positions:
                if pos.contract.secType == "STK":  # Only handle stocks for now
                    symbol = pos.contract.symbol
                    
                    # Calculate market value and PnL
                    market_price = self.get_market_price(symbol)
                    market_value = abs(pos.position) * market_price
                    
                    # For now, assume unrealized PnL is 0 (would need position cost basis)
                    unrealized_pnl = 0.0
                    realized_pnl = 0.0
                    
                    position = Position(
                        symbol=symbol,
                        quantity=pos.position,
                        average_price=pos.avgCost,
                        market_value=market_value,
                        unrealized_pnl=unrealized_pnl,
                        realized_pnl=realized_pnl,
                        timestamp=datetime.now()
                    )
                    
                    positions[symbol] = position
            
            self.positions = positions
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            raise
    
    def place_order(self, order: Order) -> str:
        """Place an order and return order ID."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            from ib_insync import Contract, Order as IBOrder
            
            # Create contract
            contract = Contract(
                symbol=order.symbol,
                secType="STK",
                exchange="SMART",
                currency="USD"
            )
            
            # Create IB order
            ib_order = IBOrder()
            ib_order.action = order.side.value.upper()
            ib_order.totalQuantity = order.quantity
            ib_order.orderType = order.order_type.value.upper()
            
            if order.price:
                ib_order.lmtPrice = order.price
            
            if order.stop_price:
                ib_order.auxPrice = order.stop_price
            
            ib_order.tif = order.time_in_force
            
            # Place order
            trade = self.ib.placeOrder(contract, ib_order)
            
            # Generate order ID
            order_id = f"IB_{self._order_id_counter}"
            self._order_id_counter += 1
            
            # Store order
            order.order_id = order_id
            order.created_at = datetime.now()
            self.orders[order_id] = order
            
            self.logger.info(f"Placed order {order_id}: {order.symbol} {order.side.value} {order.quantity}")
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            if order_id in self.orders:
                # For now, we'll just mark it as cancelled
                # In a real implementation, you'd need to track the IB trade object
                self.orders[order_id].status = OrderStatus.CANCELLED
                self.logger.info(f"Cancelled order {order_id}")
                return True
            else:
                self.logger.warning(f"Order {order_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if order_id in self.orders:
            return self.orders[order_id].status
        else:
            raise ValueError(f"Order {order_id} not found")
    
    def get_market_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            from ib_insync import Contract, Ticker
            
            # Create contract
            contract = Contract(
                symbol=symbol,
                secType="STK",
                exchange="SMART",
                currency="USD"
            )
            
            # Request market data
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data
            
            if ticker.marketPrice():
                return ticker.marketPrice()
            elif ticker.close:
                return ticker.close
            else:
                raise ValueError(f"No market price available for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error getting market price for {symbol}: {e}")
            raise
    
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """Get historical price data."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            from ib_insync import Contract, util
            
            # Create contract
            contract = Contract(
                symbol=symbol,
                secType="STK",
                exchange="SMART",
                currency="USD"
            )
            
            # Convert interval to IB format
            ib_interval = "1 day" if interval == "1d" else interval
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                end_date.strftime("%Y%m%d %H:%M:%S"),
                ib_interval,
                "1 D",
                "TRADES",
                1,
                1,
                False,
                []
            )
            
            # Convert to DataFrame
            if bars:
                df = util.df(bars)
                df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            raise
    
    def _on_order_status(self, trade):
        """Handle order status updates."""
        # This would need to be implemented to track order updates
        pass
    
    def _on_position_update(self, position):
        """Handle position updates."""
        # This would need to be implemented to track position changes
        pass
    
    def _on_account_update(self, account_value):
        """Handle account updates."""
        # This would need to be implemented to track account changes
        pass


class AlpacaConnector(BrokerConnector):
    """Alpaca connector implementation."""
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self.api = None
    
    def connect(self) -> bool:
        """Connect to Alpaca."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            
            # Create trading client
            self.trading_client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.api_secret,
                paper=True if self.config.paper_trading else False
            )
            
            # Create data client
            self.data_client = StockHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.api_secret
            )
            
            # Test connection
            account = self.trading_client.get_account()
            if account.status == "ACTIVE":
                self.connected = True
                self.logger.info("Connected to Alpaca")
                return True
            else:
                self.logger.error(f"Alpaca account not active: {account.status}")
                return False
                
        except ImportError:
            self.logger.error("alpaca-py not installed. Install with: pip install alpaca-py")
            return False
        except Exception as e:
            self.logger.error(f"Error connecting to Alpaca: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Alpaca."""
        self.connected = False
        self.logger.info("Disconnected from Alpaca")
        return True
    
    def is_connected(self) -> bool:
        """Check if connected to Alpaca."""
        return self.connected and self.trading_client is not None
    
    def get_account_info(self) -> Account:
        """Get account information."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            alpaca_account = self.trading_client.get_account()
            
            # Get positions
            positions = self.get_positions()
            
            self.account = Account(
                account_id=alpaca_account.id,
                cash=float(alpaca_account.cash),
                buying_power=float(alpaca_account.buying_power),
                equity=float(alpaca_account.equity),
                margin_balance=0.0,  # Alpaca paper/cash accounts do not have margin_balance
                positions=list(positions.values()),
                timestamp=datetime.now()
            )
            
            return self.account
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            positions = {}
            
            for pos in alpaca_positions:
                symbol = pos.symbol
                
                position = Position(
                    symbol=symbol,
                    quantity=int(float(pos.qty)),
                    average_price=float(pos.avg_entry_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    realized_pnl=0.0,  # Would need to track separately
                    timestamp=datetime.now()
                )
                
                positions[symbol] = position
            
            self.positions = positions
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            raise
    
    def place_order(self, order: Order) -> str:
        """Place an order and return order ID."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
            
            # Convert order to Alpaca format
            side = "buy" if order.side == OrderSide.BUY else "sell"
            
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=order.time_in_force.lower()
                )
            elif order.order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=order.time_in_force.lower(),
                    limit_price=order.price
                )
            elif order.order_type == OrderType.STOP:
                request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=order.time_in_force.lower(),
                    stop_price=order.stop_price
                )
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")
            
            alpaca_order = self.trading_client.submit_order(request)
            
            order_id = alpaca_order.id
            order.order_id = order_id
            order.created_at = datetime.now()
            self.orders[order_id] = order
            
            self.logger.info(f"Placed order {order_id}: {order.symbol} {order.side.value} {order.quantity}")
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            self.trading_client.cancel_order_by_id(order_id)
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
            self.logger.info(f"Cancelled order {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            alpaca_order = self.trading_client.get_order_by_id(order_id)
            
            # Map Alpaca status to our enum
            status_map = {
                "new": OrderStatus.PENDING,
                "accepted": OrderStatus.SUBMITTED,
                "partially_filled": OrderStatus.PARTIALLY_FILLED,
                "filled": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
                "expired": OrderStatus.EXPIRED
            }
            
            return status_map.get(alpaca_order.status, OrderStatus.PENDING)
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            raise
    
    def get_market_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quote:
                latest_quote = quote[symbol]
                return (latest_quote.ask_price + latest_quote.bid_price) / 2
            else:
                raise ValueError(f"No quote available for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error getting market price for {symbol}: {e}")
            raise
    
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, interval: str = "1d"):
        """Get historical price data."""
        if not self.is_connected():
            raise ConnectionError("Not connected to broker")
        
        if pd is None:
            raise ImportError("pandas is required for historical data")
        
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            # Convert interval to Alpaca TimeFrame
            interval_map = {
                "1d": TimeFrame.Day,
                "1h": TimeFrame.Hour,
                "5min": TimeFrame.Minute(5),
                "1min": TimeFrame.Minute(1)
            }
            
            alpaca_timeframe = interval_map.get(interval, TimeFrame.Day)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_timeframe,
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            # Convert to DataFrame
            if symbol in bars:
                df = bars[symbol].df
                df.reset_index(inplace=True)
                df.rename(columns={'timestamp': 'date'}, inplace=True)
                return df
            else:
                raise ValueError(f"No historical data available for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            raise


def create_broker_connector(config: BrokerConfig) -> BrokerConnector:
    """Factory function to create broker connector."""
    
    if config.broker_type.lower() == "interactive_brokers":
        return InteractiveBrokersConnector(config)
    elif config.broker_type.lower() == "alpaca":
        return AlpacaConnector(config)
    elif config.broker_type.lower() == "mock":
        return MockBrokerConnector(config)
    else:
        raise ValueError(f"Unsupported broker type: {config.broker_type}") 