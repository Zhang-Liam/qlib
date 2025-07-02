"""
Live data provider for real-time market data.
Supports multiple data sources with a unified interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import time


class LiveDataProvider(ABC):
    """
    Abstract base class for live data providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the live data provider.
        
        Args:
            config: Configuration dictionary for the data provider
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the data source.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the data source."""
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest price data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'SPY')
            
        Returns:
            Dictionary containing price data or None if not available
        """
        pass
    
    @abstractmethod
    def subscribe_symbols(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time updates for symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            
        Returns:
            True if subscription successful
        """
        pass
    
    @abstractmethod
    def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time updates for symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            True if unsubscription successful
        """
        pass
    
    def is_connected(self) -> bool:
        """Check if the provider is connected."""
        return self._connected
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status information.
        
        Returns:
            Dictionary with connection status details
        """
        return {
            'connected': self._connected,
            'provider': self.__class__.__name__,
            'config': self.config
        }


class IBKRLiveDataProvider(LiveDataProvider):
    """
    Interactive Brokers live data provider using ib_insync.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ib = None
        self.subscribed_symbols = set()
        
    def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway."""
        try:
            from ib_insync import IB
            
            self.ib = IB()
            host = self.config.get('host', '127.0.0.1')
            port = self.config.get('port', 7497)
            client_id = self.config.get('client_id', 1)
            
            self.ib.connect(host, port, clientId=client_id)
            self._connected = True
            self.logger.info(f"Connected to IB at {host}:{port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib and self._connected:
            try:
                self.ib.disconnect()
                self._connected = False
                self.logger.info("Disconnected from IB")
            except Exception as e:
                self.logger.error(f"Error disconnecting from IB: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest price data for a symbol from IB.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with price data or None
        """
        if not self._connected or not self.ib:
            return None
        
        try:
            from ib_insync import Stock, Contract
            
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request market data
            self.ib.reqMktData(contract)
            
            # Wait for data (with timeout)
            timeout = self.config.get('timeout', 5)
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if contract.marketPrice():
                    return {
                        'symbol': symbol,
                        'price': contract.marketPrice(),
                        'bid': contract.bid,
                        'ask': contract.ask,
                        'volume': contract.volume,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'IBKR'
                    }
                time.sleep(0.1)
            
            # Cancel market data request
            self.ib.cancelMktData(contract)
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def subscribe_symbols(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time updates for symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            
        Returns:
            True if subscription successful
        """
        if not self._connected:
            return False
        
        try:
            for symbol in symbols:
                if symbol not in self.subscribed_symbols:
                    # For IB, we'll handle real-time updates in a separate method
                    # This is a simplified implementation
                    self.subscribed_symbols.add(symbol)
            
            self.logger.info(f"Subscribed to {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to symbols: {e}")
            return False
    
    def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time updates for symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            True if unsubscription successful
        """
        try:
            for symbol in symbols:
                self.subscribed_symbols.discard(symbol)
            
            self.logger.info(f"Unsubscribed from {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing from symbols: {e}")
            return False


class AlpacaLiveDataProvider(LiveDataProvider):
    """
    Alpaca live data provider for US market data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api = None
        
    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            import alpaca_trade_api as tradeapi
            
            api_key = self.config.get('api_key')
            secret_key = self.config.get('secret_key')
            base_url = self.config.get('base_url', 'https://paper-api.alpaca.markets')
            
            self.api = tradeapi.REST(api_key, secret_key, base_url)
            
            # Test connection
            account = self.api.get_account()
            self._connected = True
            self.logger.info(f"Connected to Alpaca API")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Alpaca API."""
        self._connected = False
        self.logger.info("Disconnected from Alpaca API")
    
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest price data for a symbol from Alpaca.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with price data or None
        """
        if not self._connected or not self.api:
            return None
        
        try:
            # Get latest trade
            trade = self.api.get_latest_trade(symbol)
            
            if trade:
                return {
                    'symbol': symbol,
                    'price': float(trade.price),
                    'volume': int(trade.size),
                    'timestamp': trade.timestamp.isoformat(),
                    'source': 'Alpaca'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def subscribe_symbols(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time updates for symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            
        Returns:
            True if subscription successful
        """
        # Alpaca REST API doesn't support real-time streaming in this implementation
        # Would need to use WebSocket API for real-time data
        self.logger.info(f"Subscribed to {len(symbols)} symbols (REST API)")
        return True
    
    def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time updates for symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            True if unsubscription successful
        """
        self.logger.info(f"Unsubscribed from {len(symbols)} symbols")
        return True


class MockLiveDataProvider(LiveDataProvider):
    """
    Mock live data provider for testing and development.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mock_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'SPY': 450.0,
            'QQQ': 380.0
        }
        self.price_volatility = config.get('volatility', 0.01)
    
    def connect(self) -> bool:
        """Mock connection - always succeeds."""
        self._connected = True
        self.logger.info("Mock data provider connected")
        return True
    
    def disconnect(self):
        """Mock disconnection."""
        self._connected = False
        self.logger.info("Mock data provider disconnected")
    
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get mock price data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with mock price data
        """
        if symbol not in self.mock_prices:
            return None
        
        import random
        
        # Add some random price movement
        base_price = self.mock_prices[symbol]
        change = random.uniform(-self.price_volatility, self.price_volatility)
        current_price = base_price * (1 + change)
        
        # Update the mock price
        self.mock_prices[symbol] = current_price
        
        return {
            'symbol': symbol,
            'price': round(current_price, 2),
            'bid': round(current_price * 0.999, 2),
            'ask': round(current_price * 1.001, 2),
            'volume': random.randint(1000, 10000),
            'timestamp': datetime.now().isoformat(),
            'source': 'Mock'
        }
    
    def subscribe_symbols(self, symbols: List[str]) -> bool:
        """Mock subscription."""
        self.logger.info(f"Mock subscribed to {len(symbols)} symbols")
        return True
    
    def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """Mock unsubscription."""
        self.logger.info(f"Mock unsubscribed from {len(symbols)} symbols")
        return True


def create_live_data_provider(config: Dict[str, Any]) -> LiveDataProvider:
    """
    Factory function to create a live data provider based on configuration.
    
    Args:
        config: Configuration dictionary with 'provider' field
        
    Returns:
        LiveDataProvider instance
    """
    provider_type = config.get('provider', 'mock').lower()
    
    if provider_type == 'ibkr':
        return IBKRLiveDataProvider(config)
    elif provider_type == 'alpaca':
        return AlpacaLiveDataProvider(config)
    elif provider_type == 'mock':
        return MockLiveDataProvider(config)
    else:
        raise ValueError(f"Unknown data provider: {provider_type}") 