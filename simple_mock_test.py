#!/usr/bin/env python3
"""
Simple test script for mock trading components.
This script tests the mock broker and data providers directly.
"""

import sys
import os
import logging
from datetime import datetime

# Add the qlib/production directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qlib', 'production'))

def test_mock_components():
    """Test the mock components directly."""
    print("Testing Mock Trading Components")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from broker import MockBrokerConnector, Order, OrderSide, OrderType
        from live_data import MockLiveDataProvider
        from config import BrokerConfig, DataConfig
        print("✅ Imports successful")
        
        # Test broker configuration
        print("\n2. Testing broker configuration...")
        broker_config = BrokerConfig(
            broker_type="mock",
            host="127.0.0.1",
            port=7497,
            client_id=1
        )
        print("✅ Broker config created")
        
        # Test mock broker
        print("\n3. Testing mock broker...")
        broker = MockBrokerConnector(broker_config)
        
        # Test connection
        if broker.connect():
            print("✅ Mock broker connected")
        else:
            print("❌ Mock broker connection failed")
            return False
        
        # Test account info
        account = broker.get_account_info()
        print(f"✅ Account: {account.account_id}")
        print(f"   Cash: ${account.cash:,.2f}")
        print(f"   Equity: ${account.equity:,.2f}")
        
        # Test market prices
        print("\n4. Testing market prices...")
        symbols = ["AAPL", "GOOGL", "MSFT", "SPY"]
        for symbol in symbols:
            price = broker.get_market_price(symbol)
            print(f"   {symbol}: ${price:.2f}")
        
        # Test order placement
        print("\n5. Testing order placement...")
        order = Order(
            symbol="AAPL",
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        order_id = broker.place_order(order)
        print(f"✅ Order placed: {order_id}")
        
        # Check order status
        status = broker.get_order_status(order_id)
        print(f"   Order status: {status.value}")
        
        # Get updated account
        updated_account = broker.get_account_info()
        print(f"   Updated cash: ${updated_account.cash:,.2f}")
        
        # Get positions
        positions = broker.get_positions()
        if positions:
            print("   Positions:")
            for symbol, pos in positions.items():
                print(f"     {symbol}: {pos.quantity} shares @ ${pos.average_price:.2f}")
        
        # Test live data provider
        print("\n6. Testing live data provider...")
        data_config = DataConfig(data_provider="mock")
        live_data = MockLiveDataProvider(data_config.__dict__)
        
        if live_data.connect():
            print("✅ Live data provider connected")
        else:
            print("❌ Live data provider connection failed")
            return False
        
        # Test price data
        for symbol in symbols:
            price_data = live_data.get_latest_price(symbol)
            if price_data:
                print(f"   {symbol}: ${price_data['price']:.2f} (bid: ${price_data['bid']:.2f}, ask: ${price_data['ask']:.2f})")
            else:
                print(f"   {symbol}: No data")
        
        # Test subscription
        if live_data.subscribe_symbols(symbols):
            print("✅ Symbol subscription successful")
        else:
            print("❌ Symbol subscription failed")
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("The mock trading system is working correctly.")
        print("You can now use this for testing without real broker connections.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mock_components()
    if not success:
        sys.exit(1) 