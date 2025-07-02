"""
Production workflow for Qlib trading system.
Connects broker, risk manager, and live data components.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import time

from .config import ProductionConfig
from .broker import BrokerConnector, create_broker_connector
from .risk_manager import RiskManager
from .live_data import LiveDataProvider, create_live_data_provider


class ProductionWorkflow:
    """
    Main workflow class that orchestrates the production trading system.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the production workflow.
        
        Args:
            config_path: Path to the production configuration file
        """
        self.config = ProductionConfig(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.broker: Optional[BrokerConnector] = None
        self.risk_manager: Optional[RiskManager] = None
        self.live_data: Optional[LiveDataProvider] = None
        
        self.logger.info("Production workflow initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the workflow."""
        logger = logging.getLogger("qlib.production.workflow")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize_components(self) -> bool:
        """
        Initialize all production components.
        
        Returns:
            True if all components initialized successfully
        """
        try:
            # Initialize broker
            self.logger.info("Initializing broker connection...")
            self.broker = create_broker_connector(self.config.broker)
            if not self.broker.connect():
                self.logger.error("Failed to connect to broker")
                return False
            
            # Initialize risk manager
            self.logger.info("Initializing risk manager...")
            self.risk_manager = RiskManager(self.config.risk)
            self.risk_manager.update_account(self.broker.get_account_info())
            
            # Initialize live data provider
            self.logger.info("Initializing live data provider...")
            self.live_data = create_live_data_provider(self.config.data.__dict__)
            if not self.live_data.connect():
                self.logger.error("Failed to connect to live data provider")
                return False
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    def run_single_cycle(self) -> bool:
        """
        Run a single trading cycle.
        
        Returns:
            True if cycle completed successfully
        """
        try:
            # Update account information
            account_info = self.broker.get_account_info()
            self.risk_manager.update_account(account_info)
            
            # Get live market data
            symbols = getattr(self.config.trading, 'symbols', [])
            market_data = {}
            
            for symbol in symbols:
                try:
                    price_data = self.live_data.get_latest_price(symbol)
                    if price_data:
                        market_data[symbol] = price_data
                except Exception as e:
                    self.logger.warning(f"Failed to get data for {symbol}: {e}")
            
            if not market_data:
                self.logger.warning("No market data available")
                return False
            
            # Execute trading logic
            self._execute_trading_logic(market_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            return False
    
    def _execute_trading_logic(self, market_data: Dict[str, Any]):
        """
        Execute the trading logic based on market data.
        
        Args:
            market_data: Dictionary of symbol -> price data
        """
        # This is where you would implement your trading strategy
        # For now, we'll just log the data
        self.logger.info(f"Processing market data for {len(market_data)} symbols")
        
        for symbol, data in market_data.items():
            self.logger.info(f"{symbol}: {data}")
            
            # Example: Simple buy order if price is below threshold
            # In practice, you would implement your actual trading strategy here
            if getattr(self.config.trading, 'enable_trading', False):
                self._process_trading_signal(symbol, data)
    
    def _process_trading_signal(self, symbol: str, price_data: Dict[str, Any]):
        """
        Process a trading signal for a symbol.
        
        Args:
            symbol: Trading symbol
            price_data: Price data for the symbol
        """
        try:
            # Example trading logic - replace with your actual strategy
            current_price = price_data.get('price', 0)
            
            # Create a sample order (this is just an example)
            from qlib.production.broker import Order, OrderSide, OrderType
            
            order = Order(
                symbol=symbol,
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                price=current_price
            )
            
            # Check risk limits before placing order
            if self.risk_manager.validate_order(order):
                # Place the order
                order_id = self.broker.place_order(order)
                if order_id:
                    self.logger.info(f"Order placed successfully: {symbol}, ID: {order_id}")
                else:
                    self.logger.error(f"Order failed for {symbol}")
            else:
                self.logger.warning(f"Order rejected by risk manager: {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error processing trading signal for {symbol}: {e}")
    
    def run_continuous(self, cycle_interval: int = 60):
        """
        Run the workflow continuously.
        
        Args:
            cycle_interval: Interval between cycles in seconds
        """
        self.logger.info(f"Starting continuous trading with {cycle_interval}s intervals")
        
        try:
            while True:
                start_time = time.time()
                
                # Run single cycle
                success = self.run_single_cycle()
                if not success:
                    self.logger.warning("Trading cycle failed, continuing...")
                
                # Wait for next cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, cycle_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Unexpected error in continuous mode: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown all components gracefully."""
        self.logger.info("Shutting down production workflow...")
        
        try:
            if self.live_data:
                self.live_data.disconnect()
            
            if self.broker:
                self.broker.disconnect()
                
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        
        self.logger.info("Production workflow shutdown complete")


def main():
    """Main entry point for the production workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qlib Production Trading Workflow")
    parser.add_argument("--config", required=True, help="Path to production config file")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    parser.add_argument("--interval", type=int, default=60, help="Cycle interval in seconds")
    
    args = parser.parse_args()
    
    # Create and run workflow
    workflow = ProductionWorkflow(args.config)
    
    if workflow.initialize_components():
        if args.continuous:
            workflow.run_continuous(args.interval)
        else:
            workflow.run_single_cycle()
    else:
        print("Failed to initialize components")
        return 1
    
    workflow.shutdown()
    return 0


if __name__ == "__main__":
    exit(main()) 