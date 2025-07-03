#!/usr/bin/env python3
"""
Optimized Automated Trading Engine
Performance-optimized version with caching, parallel processing, and reduced API calls.
"""

import os
import sys
import time
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
from collections import defaultdict

# Add qlib to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'qlib'))

from qlib.production.broker import AlpacaConnector, Order, OrderSide, OrderType
from qlib.production.config import BrokerConfig

# Import our custom modules
from .signals import OptimizedSignalGenerator, SignalType, SignalResult
from .risk_manager import RiskManager, RiskLevel

class DataCache:
    """Efficient data caching system."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if not expired."""
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl_seconds:
                    return self.cache[key]
                else:
                    # Expired, remove
                    del self.cache[key]
                    del self.timestamps[key]
            return None
    
    def set(self, key: str, value: Any):
        """Set cached data with timestamp."""
        with self.lock:
            # Implement LRU eviction
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cached data."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

class AutomatedTradingEngine:
    """Optimized automated trading engine with performance improvements."""
    
    def __init__(self, config_path: str = "trading_config.yaml"):
        """Initialize the optimized trading engine."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.broker = self._setup_broker()
        self.signal_generator = OptimizedSignalGenerator(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Performance optimizations
        self.data_cache = DataCache(max_size=500, ttl_seconds=300)  # 5 minutes TTL
        self.account_cache = {'data': None, 'timestamp': 0, 'ttl': 30}  # 30 seconds TTL
        self.positions_cache = {'data': None, 'timestamp': 0, 'ttl': 30}  # 30 seconds TTL
        
        # Trading state
        self.positions = {}
        self.orders = {}
        self.daily_pnl = 0.0
        self.last_trading_date = None
        self.market_data = {}
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("Optimized Trading Engine initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variables
        if 'broker' in config:
            if 'api_key' in config['broker']:
                config['broker']['api_key'] = os.getenv('ALPACA_API_KEY')
            if 'secret_key' in config['broker']:
                config['broker']['secret_key'] = os.getenv('ALPACA_SECRET_KEY')
        
        return config
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config.get('file', 'trading.log')),
                logging.StreamHandler()
            ]
        )
    
    def _setup_broker(self) -> AlpacaConnector:
        """Setup broker connection."""
        broker_config = self.config['broker']
        
        config = BrokerConfig(
            broker_type=broker_config['type'],
            api_key=broker_config['api_key'],
            api_secret=broker_config['secret_key'],
            paper_trading=broker_config['paper_trading'],
            host="",  # Not used for Alpaca
            port=0,   # Not used for Alpaca
            client_id=0  # Not used for Alpaca
        )
        
        broker = AlpacaConnector(config)
        if not broker.connect():
            raise ConnectionError("Failed to connect to broker")
        
        return broker
    
    @lru_cache(maxsize=128)
    def _parse_trading_hours(self, start_time: str, end_time: str) -> Tuple[Any, Any]:
        """Cache trading hours parsing."""
        return (
            datetime.strptime(start_time, "%H:%M").time(),
            datetime.strptime(end_time, "%H:%M").time()
        )
    
    def is_market_open(self) -> bool:
        """Check if market is currently open (optimized with caching)."""
        et_tz = pytz.timezone('US/Eastern')
        now = datetime.now(et_tz)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check trading hours (cached parsing)
        trading_hours = self.config['trading']['trading_hours']
        start_time, end_time = self._parse_trading_hours(
            trading_hours['start'], trading_hours['end']
        )
        
        return start_time <= now.time() <= end_time
    
    def _get_cached_account_info(self):
        """Get cached account info to reduce API calls."""
        current_time = time.time()
        
        # Check if cache is valid
        if (self.account_cache['data'] is not None and 
            current_time - self.account_cache['timestamp'] < self.account_cache['ttl']):
            return self.account_cache['data']
        
        # Fetch fresh data
        account_info = self.broker.get_account_info()
        self.account_cache['data'] = account_info
        self.account_cache['timestamp'] = current_time
        
        return account_info
    
    def _get_cached_positions(self):
        """Get cached positions to reduce API calls."""
        current_time = time.time()
        
        # Check if cache is valid
        if (self.positions_cache['data'] is not None and 
            current_time - self.positions_cache['timestamp'] < self.positions_cache['ttl']):
            return self.positions_cache['data']
        
        # Fetch fresh data
        positions = self.broker.get_positions()
        self.positions_cache['data'] = positions
        self.positions_cache['timestamp'] = current_time
        
        return positions
    
    def _fetch_symbol_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol (for parallel processing)."""
        try:
            cache_key = f"hist_data_{symbol}_{start_date.date()}_{end_date.date()}"
            cached_data = self.data_cache.get(cache_key)
            
            if cached_data is not None:
                return cached_data
            
            hist_data = self.broker.get_historical_data(symbol, start_date, end_date, "1d")
            
            if len(hist_data) > 0:
                self.data_cache.set(cache_key, hist_data)
                return hist_data
            
        except Exception as e:
            self.logger.warning(f"Error fetching data for {symbol}: {e}")
        
        return None
    
    def get_market_data_parallel(self) -> Dict[str, pd.DataFrame]:
        """Get historical market data for all symbols using parallel processing."""
        symbols = self.config['trading']['symbols']
        lookback_days = self.config['trading']['lookback_days']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Submit all fetch tasks to thread pool
        future_to_symbol = {
            self.executor.submit(self._fetch_symbol_data, symbol, start_date, end_date): symbol
            for symbol in symbols
        }
        
        market_data = {}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                hist_data = future.result()
                if hist_data is not None:
                    market_data[symbol] = hist_data
                    self.logger.debug(f"Got {len(hist_data)} days of data for {symbol}")
            except Exception as e:
                self.logger.warning(f"Error processing data for {symbol}: {e}")
        
        return market_data
    
    def get_market_data(self) -> Dict[str, pd.DataFrame]:
        """Get market data (optimized version)."""
        return self.get_market_data_parallel()
    
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for a symbol."""
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            return self._fetch_symbol_data(symbol, start_dt, end_dt)
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def get_account_info(self):
        """Get account information."""
        return self._get_cached_account_info()
    
    def get_positions(self):
        """Get current positions."""
        return self._get_cached_positions()
    
    def run_trading_cycle(self, symbols: List[str], continuous: bool = True):
        """Run trading cycle for specific symbols."""
        if continuous:
            self.run_continuous()
        else:
            self.run_enhanced_trading_cycle()
    
    def _generate_signal_parallel(self, symbol: str, hist_data: pd.DataFrame) -> Tuple[str, SignalResult]:
        """Generate signal for a single symbol (for parallel processing)."""
        try:
            signal = self.signal_generator.generate_signal(symbol, hist_data)
            return symbol, signal
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return symbol, SignalResult(
                signal_type=SignalType.HOLD,
                confidence=0.5,
                reasoning=[f"Error: {str(e)}"]
            )
    
    def generate_enhanced_signals_parallel(self) -> Dict[str, SignalResult]:
        """Generate enhanced trading signals using parallel processing."""
        # Submit all signal generation tasks to thread pool
        future_to_symbol = {
            self.executor.submit(self._generate_signal_parallel, symbol, self.market_data[symbol]): symbol
            for symbol in self.config['trading']['symbols']
            if symbol in self.market_data
        }
        
        signals = {}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                symbol_result, signal = future.result()
                signals[symbol_result] = signal
                
                self.logger.info(f"Signal for {symbol_result}: {signal.signal_type.value} "
                               f"(confidence: {signal.confidence:.2f})")
                
            except Exception as e:
                self.logger.error(f"Error processing signal for {symbol}: {e}")
        
        return signals
    
    def generate_enhanced_signals(self) -> Dict[str, SignalResult]:
        """Generate enhanced trading signals (optimized version)."""
        return self.generate_enhanced_signals_parallel()
    
    def execute_enhanced_trades(self, signals: Dict[str, SignalResult]):
        """Execute trades based on enhanced signals and risk management (optimized)."""
        # Use cached account and position data
        account = self._get_cached_account_info()
        positions = self._get_cached_positions()
        
        # Update risk manager with current market data
        self.risk_manager.update_market_risk(self.market_data)
        
        # Get risk report (cached for this cycle)
        risk_report = self.risk_manager.get_risk_report(
            account.equity, positions, self.market_data
        )
        
        # Log risk report
        self.logger.info("Risk Report:")
        self.logger.info(f"  Portfolio PnL: {risk_report['portfolio_risk'].total_pnl_pct:.2%}")
        self.logger.info(f"  Max Drawdown: {risk_report['portfolio_risk'].max_drawdown:.2%}")
        self.logger.info(f"  Market Volatility: {risk_report['market_volatility']:.2%}")
        
        # Check for risk alerts
        if risk_report['alerts']:
            for alert in risk_report['alerts']:
                self.logger.warning(f"Risk Alert: {alert}")
        
        # Check portfolio limits
        limit_checks = risk_report['limit_checks']
        if not all(limit_checks.values()):
            failed_checks = [k for k, v in limit_checks.items() if not v]
            self.logger.warning(f"Portfolio limits exceeded: {failed_checks}")
            return  # Skip trading if limits exceeded
        
        # Process signals
        for symbol, signal in signals.items():
            current_position = positions.get(symbol)
            current_price = self.broker.get_market_price(symbol)
            
            # Check if we should exit existing position
            if current_position:
                should_exit, reason = self.risk_manager.should_exit_position(
                    symbol, current_position, current_price
                )
                if should_exit:
                    self.logger.info(f"Exiting position in {symbol}: {reason}")
                    self._place_sell_order(symbol, current_position.quantity)
                    continue
            
            # Calculate position size using risk manager
            position_size = self.risk_manager.calculate_position_size(
                symbol, signal.confidence, current_price, account.equity
            )
            
            # Execute trade based on signal
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                if current_position is None:
                    # New position
                    self._place_buy_order(symbol, position_size, signal)
                else:
                    # Add to existing position
                    additional_shares = max(1, position_size - current_position.quantity)
                    if additional_shares > 0:
                        self._place_buy_order(symbol, additional_shares, signal)
            
            elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                if current_position:
                    # Reduce or exit position
                    shares_to_sell = min(current_position.quantity, position_size)
                    self._place_sell_order(symbol, shares_to_sell, signal)
    
    def _place_buy_order(self, symbol: str, shares: int, signal: SignalResult):
        """Place a buy order with enhanced risk management."""
        try:
            # Determine order type based on signal strength
            if signal.confidence > 0.8:
                order_type = OrderType.MARKET  # Strong signal, use market order
            else:
                order_type = OrderType.LIMIT  # Weaker signal, use limit order
            
            order = Order(
                symbol=symbol,
                quantity=shares,
                side=OrderSide.BUY,
                order_type=order_type,
                time_in_force=self.config['trading']['time_in_force']
            )
            
            # Add stop loss and take profit if enabled
            if self.config['risk']['enable_stop_loss'] and signal.stop_loss:
                order.stop_price = signal.stop_loss
            
            order_id = self.broker.place_order(order)
            
            self.logger.info(f"Placed BUY order: {symbol} {shares} shares "
                           f"(ID: {order_id}, Signal: {signal.signal_type.value}, "
                           f"Confidence: {signal.confidence:.2f})")
            
            # Record trade
            self._record_trade(symbol, "BUY", shares, signal)
            
            # Invalidate position cache
            self.positions_cache['data'] = None
            
        except Exception as e:
            self.logger.error(f"Error placing buy order for {symbol}: {e}")
    
    def _place_sell_order(self, symbol: str, shares: int, signal: Optional[SignalResult] = None):
        """Place a sell order with enhanced risk management."""
        try:
            order = Order(
                symbol=symbol,
                quantity=shares,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                time_in_force=self.config['trading']['time_in_force']
            )
            
            order_id = self.broker.place_order(order)
            
            reason = signal.signal_type.value if signal else "Risk Management"
            confidence = signal.confidence if signal else 1.0
            
            self.logger.info(f"Placed SELL order: {symbol} {shares} shares "
                           f"(ID: {order_id}, Reason: {reason}, "
                           f"Confidence: {confidence:.2f})")
            
            # Record trade
            self._record_trade(symbol, "SELL", shares, signal)
            
            # Invalidate position cache
            self.positions_cache['data'] = None
            
        except Exception as e:
            self.logger.error(f"Error placing sell order for {symbol}: {e}")
    
    def _record_trade(self, symbol: str, side: str, shares: int, signal: Optional[SignalResult]):
        """Record trade for performance tracking."""
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'shares': shares,
            'signal_type': signal.signal_type.value if signal else 'MANUAL',
            'confidence': signal.confidence if signal else 1.0,
            'reasoning': signal.reasoning if signal else []
        }
        
        self.trade_history.append(trade)
        self.performance_metrics['total_trades'] += 1
    
    def update_performance_metrics(self):
        """Update performance metrics."""
        if not self.trade_history:
            return
        
        # Calculate basic metrics
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trade_history if t.get('pnl', 0) < 0])
        
        self.performance_metrics.update({
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0
        })
    
    def run_enhanced_trading_cycle(self):
        """Run one complete enhanced trading cycle (optimized)."""
        try:
            # Check if market is open
            if not self.is_market_open():
                self.logger.info("Market is closed, skipping trading cycle")
                return
            
            # Get account summary (cached)
            account = self._get_cached_account_info()
            self.logger.info(f"Account Summary: Cash: ${account.cash:,.2f}, "
                           f"Equity: ${account.equity:,.2f}")
            
            # Get market data (parallel)
            self.market_data = self.get_market_data()
            self.logger.info(f"Retrieved market data for {len(self.market_data)} symbols")
            
            # Generate enhanced signals (parallel)
            signals = self.generate_enhanced_signals()
            self.logger.info(f"Generated signals for {len(signals)} symbols")
            
            # Execute enhanced trades
            self.execute_enhanced_trades(signals)
            
            # Update performance metrics
            self.update_performance_metrics()
            
            # Log performance summary
            self.logger.info(f"Performance: {self.performance_metrics['win_rate']:.1%} win rate, "
                           f"{self.performance_metrics['total_trades']} total trades")
            
        except Exception as e:
            self.logger.error(f"Error in enhanced trading cycle: {e}")
            import traceback
            traceback.print_exc()
    
    def run_continuous(self, interval_minutes: int = 5):
        """Run continuous enhanced trading with specified interval."""
        self.logger.info(f"Starting optimized continuous trading with {interval_minutes}-minute intervals")
        
        try:
            while True:
                self.run_enhanced_trading_cycle()
                
                # Wait for next cycle
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            self.logger.info("Optimized trading stopped by user")
        except Exception as e:
            self.logger.error(f"Optimized trading stopped due to error: {e}")
        finally:
            self.executor.shutdown(wait=True)
            self.broker.disconnect()


def main():
    """Main function to run the optimized automated trading engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Automated Trading Engine')
    parser.add_argument('--config', default='trading_config.yaml', help='Configuration file path')
    parser.add_argument('--interval', type=int, default=5, help='Trading interval in minutes')
    parser.add_argument('--once', action='store_true', help='Run once instead of continuously')
    
    args = parser.parse_args()
    
    # Initialize optimized trading engine
    engine = AutomatedTradingEngine(args.config)
    
    if args.once:
        # Run once
        engine.run_enhanced_trading_cycle()
    else:
        # Run continuously
        engine.run_continuous(args.interval)


if __name__ == "__main__":
    main() 