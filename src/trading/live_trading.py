#!/usr/bin/env python3
"""
Live Trading System
Real money trading with comprehensive safety controls and position management.
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import yaml
import os
from pathlib import Path

# Import existing components
from .engine import AutomatedTradingEngine
from .risk_manager import RiskManager, RiskLevel
from .signals import OptimizedSignalGenerator, SignalType, SignalResult

class TradingMode(Enum):
    """Trading modes for safety."""
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class LiveTrade:
    """Live trade record."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    order_id: str
    status: str
    signal_confidence: float
    risk_level: str

@dataclass
class LivePosition:
    """Live position tracking."""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: datetime
    last_update: datetime

class EmergencyStop:
    """Emergency stop system for live trading."""
    
    def __init__(self):
        self.emergency_stop_active = False
        self.emergency_stop_time = None
        self.emergency_stop_reason = ""
        self.lock = threading.RLock()
    
    def activate_emergency_stop(self, reason: str = "Manual activation"):
        """Activate emergency stop."""
        with self.lock:
            self.emergency_stop_active = True
            self.emergency_stop_time = datetime.now()
            self.emergency_stop_reason = reason
            logging.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
    
    def deactivate_emergency_stop(self):
        """Deactivate emergency stop."""
        with self.lock:
            self.emergency_stop_active = False
            self.emergency_stop_time = None
            self.emergency_stop_reason = ""
            logging.info("Emergency stop deactivated")
    
    def is_active(self) -> bool:
        """Check if emergency stop is active."""
        with self.lock:
            return self.emergency_stop_active
    
    def get_status(self) -> Dict[str, Any]:
        """Get emergency stop status."""
        with self.lock:
            return {
                'active': self.emergency_stop_active,
                'time': self.emergency_stop_time,
                'reason': self.emergency_stop_reason
            }

class LiveTradingSystem:
    """Live trading system with comprehensive safety controls."""
    
    def __init__(self, config_path: str = "config/alpaca_config.yaml"):
        """Initialize live trading system."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Trading mode and safety
        self.trading_mode = TradingMode.PAPER_TRADING
        self.emergency_stop = EmergencyStop()
        
        # Initialize components
        self.trading_engine = AutomatedTradingEngine(config_path)
        self.risk_manager = RiskManager(self.config)
        
        # Live trading state
        self.positions: Dict[str, LivePosition] = {}
        self.trade_history: List[LiveTrade] = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_trading_date = None
        
        # Safety limits
        self.max_daily_loss = self.config['trading']['max_daily_loss']
        self.max_position_size = self.config['trading']['max_position_size']
        self.min_cash_reserve = self.config['trading']['min_cash_reserve']
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Threading
        self.running = False
        self.trading_thread = None
        self.monitoring_thread = None
        
        self.logger.info("Live Trading System initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
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
        """Setup logging for live trading."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'live_trading.log'),
                logging.StreamHandler()
            ]
        )
    
    def switch_to_live_trading(self) -> bool:
        """Switch from paper trading to live trading."""
        if self.emergency_stop.is_active():
            self.logger.error("Cannot switch to live trading - emergency stop is active")
            return False
        
        # Verify account setup
        try:
            account = self.trading_engine.get_account_info()
            if account.cash < 1000:  # Minimum $1000 for live trading
                self.logger.error(f"Insufficient funds for live trading: ${account.cash:.2f}")
                return False
            
            self.logger.warning("SWITCHING TO LIVE TRADING - REAL MONEY WILL BE USED")
            self.trading_mode = TradingMode.LIVE_TRADING
            
            # Update broker to live trading
            self.trading_engine.broker.config.paper_trading = False
            
            self.logger.info(f"Live trading activated. Account value: ${account.equity:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch to live trading: {e}")
            return False
    
    def switch_to_paper_trading(self):
        """Switch back to paper trading."""
        self.logger.info("Switching to paper trading")
        self.trading_mode = TradingMode.PAPER_TRADING
        self.trading_engine.broker.config.paper_trading = True
    
    def activate_emergency_stop(self, reason: str = "Manual activation"):
        """Activate emergency stop and close all positions."""
        self.emergency_stop.activate_emergency_stop(reason)
        self.trading_mode = TradingMode.EMERGENCY_STOP
        
        # Close all positions immediately
        self._close_all_positions_emergency()
        
        self.logger.critical("EMERGENCY STOP: All positions closed")
    
    def _close_all_positions_emergency(self):
        """Emergency close all positions."""
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                try:
                    # Place market sell order
                    order = self.trading_engine.broker.place_order(
                        symbol=symbol,
                        quantity=abs(position.quantity),
                        side="sell",
                        order_type="market"
                    )
                    self.logger.critical(f"Emergency sell order placed for {symbol}: {order}")
                except Exception as e:
                    self.logger.error(f"Failed to close position {symbol}: {e}")
    
    def check_safety_limits(self) -> Dict[str, bool]:
        """Check all safety limits before trading."""
        checks = {}
        
        # Check emergency stop
        checks['emergency_stop'] = not self.emergency_stop.is_active()
        
        # Check daily loss limit
        account = self.trading_engine.get_account_info()
        daily_loss_pct = abs(self.daily_pnl) / account.equity
        checks['daily_loss'] = daily_loss_pct <= self.max_daily_loss
        
        # Check cash reserve
        cash_ratio = account.cash / account.equity
        checks['cash_reserve'] = cash_ratio >= self.min_cash_reserve
        
        # Check position limits
        checks['position_count'] = len(self.positions) <= self.config['trading']['max_positions']
        
        # Check market hours
        checks['market_hours'] = self.trading_engine.is_market_open()
        
        return checks
    
    def execute_live_trade(self, signal: SignalResult, symbol: str, 
                          current_price: float) -> Optional[LiveTrade]:
        """Execute a live trade with safety checks."""
        
        # Check safety limits
        safety_checks = self.check_safety_limits()
        if not all(safety_checks.values()):
            failed_checks = [k for k, v in safety_checks.items() if not v]
            self.logger.warning(f"Safety check failed: {failed_checks}")
            return None
        
        # Calculate position size
        account = self.trading_engine.get_account_info()
        position_size = self.risk_manager.calculate_position_size(
            symbol, signal.confidence, current_price, account.equity
        )
        
        if position_size <= 0:
            self.logger.warning(f"Position size too small for {symbol}")
            return None
        
        try:
            # Place order
            if signal.signal_type == SignalType.BUY:
                order = self.trading_engine.broker.place_order(
                    symbol=symbol,
                    quantity=position_size,
                    side="buy",
                    order_type="market"
                )
                side = "buy"
            elif signal.signal_type == SignalType.SELL:
                order = self.trading_engine.broker.place_order(
                    symbol=symbol,
                    quantity=position_size,
                    side="sell",
                    order_type="market"
                )
                side = "sell"
            else:
                return None
            
            # Record trade
            live_trade = LiveTrade(
                timestamp=datetime.now(),
                symbol=symbol,
                side=side,
                quantity=position_size,
                price=current_price,
                order_id=order,
                status="submitted",
                signal_confidence=signal.confidence,
                risk_level=self.risk_manager.risk_level.value
            )
            
            self.trade_history.append(live_trade)
            self.total_trades += 1
            
            self.logger.info(f"Live trade executed: {side} {position_size} {symbol} @ ${current_price:.2f}")
            
            return live_trade
            
        except Exception as e:
            self.logger.error(f"Failed to execute live trade for {symbol}: {e}")
            return None
    
    def update_positions(self):
        """Update current positions from broker."""
        try:
            broker_positions = self.trading_engine.get_positions()
            
            for symbol, position in broker_positions.items():
                if position.quantity != 0:
                    live_position = LivePosition(
                        symbol=symbol,
                        quantity=position.quantity,
                        average_price=position.average_price,
                        current_price=position.market_value / abs(position.quantity),
                        unrealized_pnl=position.unrealized_pnl,
                        unrealized_pnl_pct=position.unrealized_pnl / (position.quantity * position.average_price),
                        entry_time=datetime.now(),  # Would need to track actual entry time
                        last_update=datetime.now()
                    )
                    self.positions[symbol] = live_position
                else:
                    # Position closed
                    if symbol in self.positions:
                        del self.positions[symbol]
            
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}")
    
    def run_live_trading_cycle(self, symbols: List[str]):
        """Run one live trading cycle."""
        
        # Check emergency stop
        if self.emergency_stop.is_active():
            self.logger.warning("Trading cycle skipped - emergency stop active")
            return
        
        # Update positions
        self.update_positions()
        
        # Get market data
        market_data = self.trading_engine.get_market_data()
        
        # Generate signals
        signals = self.trading_engine.generate_enhanced_signals()
        
        # Execute trades
        for symbol, signal in signals.items():
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]['close'].iloc[-1]
            
            # Execute live trade
            trade = self.execute_live_trade(signal, symbol, current_price)
            
            if trade:
                self.logger.info(f"Live trade executed: {trade.symbol} {trade.side} {trade.quantity}")
    
    def start_live_trading(self, symbols: List[str], interval_minutes: int = 5):
        """Start live trading with continuous monitoring."""
        
        if self.trading_mode == TradingMode.LIVE_TRADING:
            self.logger.warning("STARTING LIVE TRADING - REAL MONEY WILL BE USED")
        else:
            self.logger.info("Starting paper trading")
        
        self.running = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()
        
        # Start trading thread
        self.trading_thread = threading.Thread(
            target=self._trading_loop, 
            args=(symbols, interval_minutes)
        )
        self.trading_thread.start()
        
        self.logger.info(f"Live trading started for symbols: {symbols}")
    
    def stop_live_trading(self):
        """Stop live trading."""
        self.running = False
        
        if self.trading_thread:
            self.trading_thread.join()
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        self.logger.info("Live trading stopped")
    
    def _trading_loop(self, symbols: List[str], interval_minutes: int):
        """Main trading loop."""
        while self.running:
            try:
                if self.trading_engine.is_market_open():
                    self.run_live_trading_cycle(symbols)
                
                # Wait for next cycle
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.running:
            try:
                # Check safety limits
                safety_checks = self.check_safety_limits()
                
                # Check for emergency conditions
                account = self.trading_engine.get_account_info()
                daily_loss_pct = abs(self.daily_pnl) / account.equity
                
                if daily_loss_pct > self.max_daily_loss * 0.8:  # 80% of limit
                    self.logger.warning(f"Approaching daily loss limit: {daily_loss_pct:.2%}")
                
                if daily_loss_pct > self.max_daily_loss:
                    self.activate_emergency_stop(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
                
                # Update performance metrics
                self._update_performance_metrics()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        # Calculate total PnL
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Update daily PnL
        current_date = datetime.now().date()
        if self.last_trading_date != current_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_trading_date = current_date
        
        self.daily_pnl = total_pnl
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get comprehensive trading status."""
        account = self.trading_engine.get_account_info()
        
        return {
            'trading_mode': self.trading_mode.value,
            'emergency_stop': self.emergency_stop.get_status(),
            'account_value': account.equity,
            'cash': account.cash,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'positions': len(self.positions),
            'safety_checks': self.check_safety_limits(),
            'uptime': (datetime.now() - self.start_time).total_seconds() / 3600,  # hours
            'market_open': self.trading_engine.is_market_open()
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        account = self.trading_engine.get_account_info()
        
        return {
            'summary': {
                'total_return': (account.equity - 100000) / 100000,  # Assuming $100k starting
                'total_trades': self.total_trades,
                'win_rate': self.winning_trades / max(self.total_trades, 1),
                'avg_trade_pnl': self.total_pnl / max(self.total_trades, 1),
                'max_drawdown': self._calculate_max_drawdown(),
                'sharpe_ratio': self._calculate_sharpe_ratio()
            },
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.average_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                }
                for symbol, pos in self.positions.items()
            },
            'recent_trades': [
                {
                    'timestamp': trade.timestamp.isoformat(),
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'confidence': trade.signal_confidence
                }
                for trade in self.trade_history[-10:]  # Last 10 trades
            ]
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        # Simplified calculation - would need historical equity curve
        return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        # Simplified calculation - would need historical returns
        return 0.0

def main():
    """Main function for live trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Trading System")
    parser.add_argument("--config", default="config/alpaca_config.yaml", help="Configuration file")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOGL"], help="Trading symbols")
    parser.add_argument("--interval", type=int, default=5, help="Trading interval in minutes")
    parser.add_argument("--live", action="store_true", help="Enable live trading (real money)")
    parser.add_argument("--emergency-stop", action="store_true", help="Activate emergency stop")
    
    args = parser.parse_args()
    
    # Initialize live trading system
    live_system = LiveTradingSystem(args.config)
    
    if args.emergency_stop:
        live_system.activate_emergency_stop("Command line activation")
        return
    
    if args.live:
        if not live_system.switch_to_live_trading():
            print("Failed to switch to live trading")
            return
    
    # Start trading
    try:
        live_system.start_live_trading(args.symbols, args.interval)
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping live trading...")
        live_system.stop_live_trading()
        
        # Print final status
        status = live_system.get_trading_status()
        print(f"\nFinal Status:")
        print(f"Trading Mode: {status['trading_mode']}")
        print(f"Account Value: ${status['account_value']:.2f}")
        print(f"Total PnL: ${status['total_pnl']:.2f}")
        print(f"Total Trades: {status['total_trades']}")

if __name__ == "__main__":
    main() 