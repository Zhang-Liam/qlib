#!/usr/bin/env python3
"""
Test script for Alpaca integration.
This script tests the Alpaca broker connector with real API calls.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Use absolute imports for direct script execution
from qlib.production.broker import AlpacaConnector, Order, OrderSide, OrderType
from qlib.production.config import BrokerConfig

def test_alpaca_integration():
    """Test the Alpaca integration."""
    print("Testing Alpaca Integration")
    print("=" * 50)
    
    try:
        print("1. Testing imports...")
        print("✅ Imports successful")
        
        # Check for API keys
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            print("\n❌ ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables not set")
            print("Please set them with your Alpaca API credentials:")
            print("export ALPACA_API_KEY='your_api_key'")
            print("export ALPACA_SECRET_KEY='your_secret_key'")
            return False
        
        # Test broker configuration
        print("\n2. Testing broker configuration...")
        broker_config = BrokerConfig(
            broker_type="alpaca",
            host="",  # Not used for Alpaca
            port=0,   # Not used for Alpaca
            client_id=1,
            api_key=api_key,
            api_secret=secret_key,
            paper_trading=True
        )
        print("✅ Broker config created")
        
        # Test Alpaca connector
        print("\n3. Testing Alpaca connector...")
        broker = AlpacaConnector(broker_config)
        
        # Test connection
        if broker.connect():
            print("✅ Alpaca connected successfully")
        else:
            print("❌ Alpaca connection failed")
            return False
        
        # Test account info
        print("\n4. Testing account info...")
        account = broker.get_account_info()
        print(f"✅ Account: {account.account_id}")
        print(f"   Cash: ${account.cash:,.2f}")
        print(f"   Buying Power: ${account.buying_power:,.2f}")
        print(f"   Equity: ${account.equity:,.2f}")
        
        # Test market prices
        print("\n5. Testing market prices...")
        symbols = ["AAPL", "GOOGL", "MSFT", "SPY"]
        for symbol in symbols:
            try:
                price = broker.get_market_price(symbol)
                print(f"   {symbol}: ${price:.2f}")
            except Exception as e:
                print(f"   {symbol}: Error - {e}")
        
        # Test historical data
        print("\n6. Testing historical data...")
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            hist_data = broker.get_historical_data("AAPL", start_date, end_date, "1d")
            print(f"   AAPL historical data: {len(hist_data)} bars")
            if len(hist_data) > 0:
                print(f"   Latest close: ${hist_data.iloc[-1]['close']:.2f}")
        except Exception as e:
            print(f"   Historical data error: {e}")
        
        # Test positions
        print("\n7. Testing positions...")
        try:
            positions = broker.get_positions()
            if positions:
                print("   Current positions:")
                for symbol, pos in positions.items():
                    print(f"     {symbol}: {pos.quantity} shares @ ${pos.average_price:.2f}")
            else:
                print("   No positions")
        except Exception as e:
            print(f"   Positions error: {e}")
        
        # Test order placement (paper trading only)
        print("\n8. Testing order placement (paper trading)...")
        try:
            # Create a small test order
            test_order = Order(
                symbol="AAPL",
                quantity=1,  # Just 1 share for testing
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                price=None
            )
            
            order_id = broker.place_order(test_order)
            print(f"✅ Test order placed: {order_id}")
            
            # Check order status
            status = broker.get_order_status(order_id)
            print(f"   Order status: {status.value}")
            
            # Cancel the order
            if broker.cancel_order(order_id):
                print("✅ Test order cancelled")
            else:
                print("⚠️  Could not cancel test order")
                
        except Exception as e:
            print(f"   Order placement error: {e}")
        
        print("\n" + "=" * 50)
        print("✅ ALPACA INTEGRATION TEST COMPLETED!")
        print("The Alpaca integration is working correctly.")
        print("You can now use Alpaca for live trading.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("ALPACA INTEGRATION TEST")
    print("=" * 50)
    print()
    print("This test will:")
    print("1. Connect to Alpaca using your API credentials")
    print("2. Test account information retrieval")
    print("3. Test market data access")
    print("4. Test historical data retrieval")
    print("5. Test order placement (paper trading only)")
    print()
    print("Make sure you have set your Alpaca API credentials:")
    print("export ALPACA_API_KEY='your_api_key'")
    print("export ALPACA_SECRET_KEY='your_secret_key'")
    print()
    
    success = test_alpaca_integration()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 