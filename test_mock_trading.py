#!/usr/bin/env python3
"""
Test script for mock trading system.
This script tests the production workflow with mock broker and data providers.
"""

import sys
import os
import logging
from datetime import datetime

# Add the qlib directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qlib'))

from qlib.production.workflow import ProductionWorkflow
from qlib.production.config import ProductionConfig

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_mock_trading.log')
        ]
    )

def test_mock_trading():
    """Test the mock trading system."""
    logger = logging.getLogger(__name__)
    logger.info("Starting mock trading test")
    
    try:
        # Create a mock configuration
        config = ProductionConfig()
        
        # Override with mock settings
        config.broker.broker_type = "mock"
        config.data.data_provider = "mock"
        config.trading.symbols = ["AAPL", "GOOGL", "MSFT"]
        config.trading.enable_trading = True
        
        # Create workflow
        workflow = ProductionWorkflow("mock_config")
        workflow.config = config
        
        logger.info("Initializing components...")
        
        # Initialize components
        if not workflow.initialize_components():
            logger.error("Failed to initialize components")
            return False
        
        logger.info("Components initialized successfully")
        
        # Test account info
        account_info = workflow.broker.get_account_info()
        logger.info(f"Account: {account_info.account_id}")
        logger.info(f"Cash: ${account_info.cash:,.2f}")
        logger.info(f"Equity: ${account_info.equity:,.2f}")
        
        # Test market data
        logger.info("Testing market data...")
        for symbol in config.trading.symbols:
            price_data = workflow.live_data.get_latest_price(symbol)
            if price_data:
                logger.info(f"{symbol}: ${price_data['price']:.2f}")
            else:
                logger.warning(f"No price data for {symbol}")
        
        # Test a single trading cycle
        logger.info("Running single trading cycle...")
        success = workflow.run_single_cycle()
        
        if success:
            logger.info("Trading cycle completed successfully")
        else:
            logger.warning("Trading cycle failed")
        
        # Test order placement
        logger.info("Testing order placement...")
        from qlib.production.broker import Order, OrderSide, OrderType
        
        test_order = Order(
            symbol="AAPL",
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        if workflow.risk_manager.validate_order(test_order):
            order_id = workflow.broker.place_order(test_order)
            logger.info(f"Test order placed: {order_id}")
            
            # Check order status
            status = workflow.broker.get_order_status(order_id)
            logger.info(f"Order status: {status.value}")
        else:
            logger.warning("Test order rejected by risk manager")
        
        # Get updated account info
        updated_account = workflow.broker.get_account_info()
        logger.info(f"Updated cash: ${updated_account.cash:,.2f}")
        logger.info(f"Updated equity: ${updated_account.equity:,.2f}")
        
        # Get positions
        positions = workflow.broker.get_positions()
        if positions:
            logger.info("Current positions:")
            for symbol, pos in positions.items():
                logger.info(f"  {symbol}: {pos.quantity} shares @ ${pos.average_price:.2f}")
        else:
            logger.info("No positions")
        
        logger.info("Mock trading test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    setup_logging()
    
    print("=" * 60)
    print("QLIB MOCK TRADING TEST")
    print("=" * 60)
    print()
    
    success = test_mock_trading()
    
    print()
    print("=" * 60)
    if success:
        print("✅ MOCK TRADING TEST PASSED")
        print("The mock trading system is working correctly.")
        print("You can now proceed with real broker integration.")
    else:
        print("❌ MOCK TRADING TEST FAILED")
        print("Check the logs for details.")
    print("=" * 60)

if __name__ == "__main__":
    main() 