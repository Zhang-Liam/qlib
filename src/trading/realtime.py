#!/usr/bin/env python3
"""
Real-time Data Streaming System
WebSocket-based real-time market data streaming with signal generation.
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time

class DataType(Enum):
    """Types of real-time data."""
    PRICE = "price"
    TRADE = "trade"
    ORDERBOOK = "orderbook"
    VOLUME = "volume"

@dataclass
class RealTimeData:
    """Real-time market data point."""
    timestamp: datetime
    symbol: str
    data_type: DataType
    price: Optional[float] = None
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    trade_size: Optional[int] = None

class RealTimeDataStream:
    """Real-time market data streaming system."""
    
    def __init__(self, config: dict):
        """Initialize real-time data stream."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # WebSocket connections
        self.websocket = None
        self.connected = False
        
        # Data storage
        self.latest_data: Dict[str, RealTimeData] = {}
        self.data_buffer: Dict[str, List[RealTimeData]] = {}
        self.buffer_size = config.get('buffer_size', 1000)
        
        # Subscriptions
        self.subscribed_symbols = set()
        self.callbacks: List[Callable] = []
        
        # Threading
        self.data_queue = queue.Queue()
        self.running = False
        self.processing_thread = None
        
        # Alpaca configuration
        self.api_key = config['broker']['api_key']
        self.api_secret = config['broker']['secret_key']
        self.base_url = "wss://stream.data.alpaca.markets/v2/iex"  # Alpaca v2 WebSocket
        
    async def connect(self):
        """Connect to Alpaca WebSocket stream."""
        try:
            self.websocket = await websockets.connect(self.base_url)
            self.connected = True
            self.logger.info("Connected to Alpaca WebSocket stream")
            
            # Send authentication
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            await self.websocket.send(json.dumps(auth_message))
            
            # Start processing thread
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_data_loop)
            self.processing_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket stream."""
        self.running = False
        if self.websocket:
            await self.websocket.close()
        self.connected = False
        self.logger.info("Disconnected from WebSocket stream")
    
    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to real-time data for symbols."""
        if not self.connected or self.websocket is None:
            self.logger.error("Not connected to WebSocket")
            return False
        
        try:
            # Subscribe to trades and quotes
            subscribe_message = {
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols
            }
            await self.websocket.send(json.dumps(subscribe_message))
            
            self.subscribed_symbols.update(symbols)
            self.logger.info(f"Subscribed to symbols: {symbols}")
            
            # Initialize data buffers
            for symbol in symbols:
                if symbol not in self.data_buffer:
                    self.data_buffer[symbol] = []
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to symbols: {e}")
            return False
    
    async def unsubscribe_symbols(self, symbols: List[str]):
        """Unsubscribe from real-time data for symbols."""
        if not self.connected or self.websocket is None:
            return False
        
        try:
            unsubscribe_message = {
                "action": "unsubscribe",
                "trades": symbols,
                "quotes": symbols
            }
            await self.websocket.send(json.dumps(unsubscribe_message))
            
            self.subscribed_symbols.difference_update(symbols)
            self.logger.info(f"Unsubscribed from symbols: {symbols}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from symbols: {e}")
            return False
    
    async def listen_for_data(self):
        """Listen for incoming WebSocket data."""
        if not self.connected or self.websocket is None:
            return
        
        try:
            async for message in self.websocket:
                if not self.running:
                    break
                
                data = json.loads(message)
                await self._handle_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            self.logger.error(f"Error in WebSocket listener: {e}")
    
    async def _handle_message(self, message: dict):
        """Handle incoming WebSocket message."""
        try:
            if 'T' in message:  # Trade data
                await self._handle_trade_data(message)
            elif 'Q' in message:  # Quote data
                await self._handle_quote_data(message)
            elif 'success' in message:  # Authentication response
                self.logger.info(f"Authentication: {message['success']}")
            else:
                self.logger.debug(f"Unknown message type: {message}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def _handle_trade_data(self, message: dict):
        """Handle trade data message."""
        try:
            trade_data = message['T']
            
            real_time_data = RealTimeData(
                timestamp=datetime.fromtimestamp(trade_data['t'] / 1000000000),
                symbol=trade_data['S'],
                data_type=DataType.TRADE,
                price=float(trade_data['p']),
                volume=int(trade_data['s']),
                trade_size=int(trade_data['s'])
            )
            
            # Update latest data
            self.latest_data[trade_data['S']] = real_time_data
            
            # Add to buffer
            symbol = trade_data['S']
            if symbol in self.data_buffer:
                self.data_buffer[symbol].append(real_time_data)
                
                # Maintain buffer size
                if len(self.data_buffer[symbol]) > self.buffer_size:
                    self.data_buffer[symbol].pop(0)
            
            # Add to processing queue
            self.data_queue.put(real_time_data)
            
        except Exception as e:
            self.logger.error(f"Error handling trade data: {e}")
    
    async def _handle_quote_data(self, message: dict):
        """Handle quote data message."""
        try:
            quote_data = message['Q']
            
            real_time_data = RealTimeData(
                timestamp=datetime.fromtimestamp(quote_data['t'] / 1000000000),
                symbol=quote_data['S'],
                data_type=DataType.ORDERBOOK,
                bid=float(quote_data['b']),
                ask=float(quote_data['a']),
                bid_size=int(quote_data['B']),
                ask_size=int(quote_data['A'])
            )
            
            # Update latest data
            self.latest_data[quote_data['S']] = real_time_data
            
            # Add to buffer
            symbol = quote_data['S']
            if symbol in self.data_buffer:
                self.data_buffer[symbol].append(real_time_data)
                
                # Maintain buffer size
                if len(self.data_buffer[symbol]) > self.buffer_size:
                    self.data_buffer[symbol].pop(0)
            
            # Add to processing queue
            self.data_queue.put(real_time_data)
            
        except Exception as e:
            self.logger.error(f"Error handling quote data: {e}")
    
    def _process_data_loop(self):
        """Process data from queue in separate thread."""
        while self.running:
            try:
                # Get data from queue with timeout
                data = self.data_queue.get(timeout=1)
                
                # Process data
                self._process_data(data)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in callback: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in data processing loop: {e}")
    
    def _process_data(self, data: RealTimeData):
        """Process real-time data."""
        # This is where you can add custom processing logic
        # For example, updating technical indicators, generating signals, etc.
        pass
    
    def add_callback(self, callback: Callable):
        """Add callback function for data updates."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        if symbol in self.latest_data:
            return self.latest_data[symbol].price
        return None
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest quote for a symbol."""
        if symbol in self.latest_data:
            data = self.latest_data[symbol]
            if data.bid and data.ask:
                return {
                    'bid': float(data.bid),
                    'ask': float(data.ask),
                    'bid_size': float(data.bid_size) if data.bid_size else 0.0,
                    'ask_size': float(data.ask_size) if data.ask_size else 0.0,
                    'spread': float(data.ask - data.bid)
                }
        return None
    
    def get_data_buffer(self, symbol: str, minutes: int = 5) -> List[RealTimeData]:
        """Get recent data buffer for a symbol."""
        if symbol not in self.data_buffer:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_data = [
            data for data in self.data_buffer[symbol]
            if data.timestamp >= cutoff_time
        ]
        
        return recent_data
    
    def get_data_as_dataframe(self, symbol: str, minutes: int = 5) -> pd.DataFrame:
        """Get recent data as pandas DataFrame."""
        data_list = self.get_data_buffer(symbol, minutes)
        
        if not data_list:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_data = []
        for data in data_list:
            row = {
                'timestamp': data.timestamp,
                'symbol': data.symbol,
                'data_type': data.data_type.value
            }
            
            if data.price:
                row['price'] = data.price
            if data.volume:
                row['volume'] = data.volume
            if data.bid:
                row['bid'] = data.bid
            if data.ask:
                row['ask'] = data.ask
            if data.bid_size:
                row['bid_size'] = data.bid_size
            if data.ask_size:
                row['ask_size'] = data.ask_size
            
            df_data.append(row)
        
        return pd.DataFrame(df_data)

class RealTimeSignalGenerator:
    """Real-time signal generator using streaming data."""
    
    def __init__(self, config: dict):
        """Initialize real-time signal generator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Technical indicators for real-time calculation
        self.price_windows = {}  # Rolling price windows
        self.volume_windows = {}  # Rolling volume windows
        
        # Signal thresholds
        self.price_change_threshold = config.get('price_change_threshold', 0.01)
        self.volume_spike_threshold = config.get('volume_spike_threshold', 2.0)
        
    def update_price_data(self, symbol: str, price: float, timestamp: datetime):
        """Update price data for real-time analysis."""
        if symbol not in self.price_windows:
            self.price_windows[symbol] = []
        
        self.price_windows[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Keep only recent data (last 100 points)
        if len(self.price_windows[symbol]) > 100:
            self.price_windows[symbol].pop(0)
    
    def update_volume_data(self, symbol: str, volume: int, timestamp: datetime):
        """Update volume data for real-time analysis."""
        if symbol not in self.volume_windows:
            self.volume_windows[symbol] = []
        
        self.volume_windows[symbol].append({
            'volume': volume,
            'timestamp': timestamp
        })
        
        # Keep only recent data (last 100 points)
        if len(self.volume_windows[symbol]) > 100:
            self.volume_windows[symbol].pop(0)
    
    def generate_real_time_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate real-time trading signal."""
        if symbol not in self.price_windows or len(self.price_windows[symbol]) < 10:
            return None
        
        price_data = self.price_windows[symbol]
        recent_prices = [p['price'] for p in price_data[-10:]]
        
        # Calculate price momentum
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Calculate volume spike
        volume_spike = False
        if symbol in self.volume_windows and len(self.volume_windows[symbol]) >= 10:
            recent_volumes = [v['volume'] for v in self.volume_windows[symbol][-10:]]
            avg_volume = np.mean(recent_volumes[:-1])
            current_volume = recent_volumes[-1]
            volume_spike = current_volume > (avg_volume * self.volume_spike_threshold)
        
        # Generate signal
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price_change': price_change,
            'volume_spike': volume_spike,
            'signal_strength': 0.0,
            'signal_type': 'hold'
        }
        
        # Determine signal type
        if price_change > self.price_change_threshold and volume_spike:
            signal['signal_type'] = 'buy'
            signal['signal_strength'] = min(abs(price_change) * 10, 1.0)
        elif price_change < -self.price_change_threshold and volume_spike:
            signal['signal_type'] = 'sell'
            signal['signal_strength'] = min(abs(price_change) * 10, 1.0)
        
        return signal

class RealTimeTradingEngine:
    """Real-time trading engine using streaming data."""
    
    def __init__(self, config: dict):
        """Initialize real-time trading engine."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.data_stream = RealTimeDataStream(config)
        self.signal_generator = RealTimeSignalGenerator(config)
        
        # Trading state
        self.active_signals = {}
        self.last_trade_time = {}
        
        # Callbacks
        self.data_stream.add_callback(self._on_data_update)
        
    async def start(self, symbols: List[str]):
        """Start real-time trading."""
        # Connect to data stream
        if not await self.data_stream.connect():
            self.logger.error("Failed to connect to data stream")
            return False
        
        # Subscribe to symbols
        if not await self.data_stream.subscribe_symbols(symbols):
            self.logger.error("Failed to subscribe to symbols")
            return False
        
        # Start listening for data
        await self.data_stream.listen_for_data()
        
        return True
    
    async def stop(self):
        """Stop real-time trading."""
        await self.data_stream.disconnect()
    
    def _on_data_update(self, data: RealTimeData):
        """Handle real-time data updates."""
        try:
            # Update signal generator
            if data.price:
                self.signal_generator.update_price_data(data.symbol, data.price, data.timestamp)
            
            if data.volume:
                self.signal_generator.update_volume_data(data.symbol, data.volume, data.timestamp)
            
            # Generate real-time signal
            signal = self.signal_generator.generate_real_time_signal(data.symbol)
            
            if signal and signal['signal_strength'] > 0.5:
                self._handle_real_time_signal(signal)
                
        except Exception as e:
            self.logger.error(f"Error handling data update: {e}")
    
    def _handle_real_time_signal(self, signal: Dict[str, Any]):
        """Handle real-time trading signal."""
        symbol = signal['symbol']
        
        # Check if we should trade (avoid too frequent trading)
        if symbol in self.last_trade_time:
            time_since_last_trade = datetime.now() - self.last_trade_time[symbol]
            if time_since_last_trade.total_seconds() < 60:  # Minimum 1 minute between trades
                return
        
        # Log signal
        self.logger.info(f"Real-time signal: {signal}")
        
        # Store active signal
        self.active_signals[symbol] = signal
        
        # Here you would integrate with your main trading engine
        # For now, just log the signal
        self.logger.info(f"Would execute {signal['signal_type']} order for {symbol} "
                        f"with strength {signal['signal_strength']:.2f}")
        
        # Update last trade time
        self.last_trade_time[symbol] = datetime.now() 