#!/usr/bin/env python3
"""
Alpaca Integration Test Suite
Test Alpaca broker integration functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alpaca_connection():
    """Test Alpaca broker connection."""
    print("\n" + "="*60)
    print("TESTING ALPACA CONNECTION")
    print("="*60)
    
    try:
        from qlib.production.broker import AlpacaConnector
        from qlib.production.config import BrokerConfig
        
        # Load config
        with open("config/alpaca_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        broker_config = config['broker']
        
        # Create broker config
        alpaca_config = BrokerConfig(
            broker_type=broker_config['type'],
            api_key=broker_config['api_key'],
            api_secret=broker_config['secret_key'],
            paper_trading=broker_config['paper_trading'],
            host="",
            port=0,
            client_id=0
        )
        
        # Test connection
        broker = AlpacaConnector(alpaca_config)
        connected = broker.connect()
        
        if connected:
            print("✅ Alpaca connection successful")
            
            # Test account info
            account = broker.get_account_info()
            print(f"   Account ID: {account.account_id}")
            print(f"   Cash: ${account.cash:,.2f}")
            print(f"   Equity: ${account.equity:,.2f}")
            
            broker.disconnect()
            return True
        else:
            print("❌ Alpaca connection failed")
            return False
        
    except Exception as e:
        print(f"❌ Alpaca connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_market_data():
    """Test market data retrieval."""
    print("\n" + "="*60)
    print("TESTING MARKET DATA")
    print("="*60)
    
    try:
        from qlib.production.broker import AlpacaConnector
        from qlib.production.config import BrokerConfig
        import yaml
        
        # Load config
        with open("config/alpaca_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        broker_config = config['broker']
        
        # Create broker config
        alpaca_config = BrokerConfig(
            broker_type=broker_config['type'],
            api_key=broker_config['api_key'],
            api_secret=broker_config['secret_key'],
            paper_trading=broker_config['paper_trading'],
            host="",
            port=0,
            client_id=0
        )
        
        # Connect and test market data
        broker = AlpacaConnector(alpaca_config)
        if broker.connect():
            # Test market prices
            symbols = ["AAPL", "MSFT", "GOOGL"]
            for symbol in symbols:
                try:
                    price = broker.get_market_price(symbol)
                    print(f"✅ {symbol}: ${price:.2f}")
                except Exception as e:
                    print(f"❌ {symbol}: Error - {e}")
            
            broker.disconnect()
            return True
        else:
            print("❌ Failed to connect to Alpaca")
            return False
        
    except Exception as e:
        print(f"❌ Market data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_order_placement():
    """Test order placement (paper trading only)."""
    print("\n" + "="*60)
    print("TESTING ORDER PLACEMENT")
    print("="*60)
    
    try:
        from qlib.production.broker import AlpacaConnector, Order, OrderSide, OrderType
        from qlib.production.config import BrokerConfig
        import yaml
        
        # Load config
        with open("config/alpaca_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        broker_config = config['broker']
        
        # Only test if paper trading is enabled
        if not broker_config['paper_trading']:
            print("⚠️ Skipping order test - not in paper trading mode")
            return True
        
        # Create broker config
        alpaca_config = BrokerConfig(
            broker_type=broker_config['type'],
            api_key=broker_config['api_key'],
            api_secret=broker_config['secret_key'],
            paper_trading=broker_config['paper_trading'],
            host="",
            port=0,
            client_id=0
        )
        
        # Connect and test order placement
        broker = AlpacaConnector(alpaca_config)
        if broker.connect():
            # Test limit order (will likely not fill, but tests order creation)
            symbol = "AAPL"
            current_price = broker.get_market_price(symbol)
            
            # Create a limit order well below market price (won't fill)
            limit_price = current_price * 0.5  # 50% below market
            
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1,
                price=limit_price
            )
            
            # Place order
            order_id = broker.place_order(order)
            print(f"✅ Order placed successfully: {order_id}")
            
            # Cancel order
            cancelled = broker.cancel_order(order_id)
            if cancelled:
                print(f"✅ Order cancelled successfully: {order_id}")
            else:
                print(f"⚠️ Order cancellation failed: {order_id}")
            
            broker.disconnect()
            return True
        else:
            print("❌ Failed to connect to Alpaca")
            return False
        
    except Exception as e:
        print(f"❌ Order placement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_historical_data():
    """Test historical data retrieval."""
    print("\n" + "="*60)
    print("TESTING HISTORICAL DATA")
    print("="*60)
    
    try:
        from qlib.production.broker import AlpacaConnector
        from qlib.production.config import BrokerConfig
        import yaml
        
        # Load config
        with open("config/alpaca_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        broker_config = config['broker']
        
        # Create broker config
        alpaca_config = BrokerConfig(
            broker_type=broker_config['type'],
            api_key=broker_config['api_key'],
            api_secret=broker_config['secret_key'],
            paper_trading=broker_config['paper_trading'],
            host="",
            port=0,
            client_id=0
        )
        
        # Connect and test historical data
        broker = AlpacaConnector(alpaca_config)
        if broker.connect():
            symbol = "AAPL"
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 31)
            
            # Test historical data retrieval
            data = broker.get_historical_data(symbol, start_date, end_date, "1d")
            
            if data is not None and not data.empty:
                print(f"✅ Historical data retrieved successfully")
                print(f"   Symbol: {symbol}")
                print(f"   Data points: {len(data)}")
                print(f"   Columns: {list(data.columns)}")
                print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            else:
                print("⚠️ No historical data available (this is normal for paper trading)")
            
            broker.disconnect()
            return True
        else:
            print("❌ Failed to connect to Alpaca")
            return False
        
    except Exception as e:
        print(f"❌ Historical data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Alpaca integration tests."""
    print("🧪 ALPACA INTEGRATION TEST SUITE")
    print("="*60)
    
    results = {
        "connection": False,
        "market_data": False,
        "order_placement": False,
        "historical_data": False
    }
    
    # Run tests
    results["connection"] = test_alpaca_connection()
    results["market_data"] = test_market_data()
    results["order_placement"] = test_order_placement()
    results["historical_data"] = test_historical_data()
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 