#!/usr/bin/env python3
"""
Quick test for Alpaca integration.
This script directly tests Alpaca API connectivity.
"""

import os
import sys
from datetime import datetime, timedelta

def test_alpaca_connection():
    """Test basic Alpaca connectivity."""
    print("Testing Alpaca Connection")
    print("=" * 40)
    
    try:
        # Check if alpaca-py is installed
        print("1. Checking alpaca-py installation...")
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        print("✅ alpaca-py is installed")
        
        # Check for API keys
        print("\n2. Checking API credentials...")
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            print("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY not set")
            print("Please set them:")
            print("export ALPACA_API_KEY='your_api_key'")
            print("export ALPACA_SECRET_KEY='your_secret_key'")
            return False
        
        print("✅ API credentials found")
        
        # Test connection
        print("\n3. Testing Alpaca connection...")
        trading_client = TradingClient(api_key, secret_key, paper=True)
        
        # Get account info
        account = trading_client.get_account()
        print(f"✅ Connected to Alpaca")
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Equity: ${float(account.equity):,.2f}")
        
        # Test market data
        print("\n4. Testing market data...")
        data_client = StockHistoricalDataClient(api_key, secret_key)
        
        from alpaca.data.requests import StockLatestQuoteRequest
        request = StockLatestQuoteRequest(symbol_or_symbols="AAPL")
        quote = data_client.get_stock_latest_quote(request)
        
        if "AAPL" in quote:
            aapl_quote = quote["AAPL"]
            print(f"✅ AAPL Quote:")
            print(f"   Bid: ${aapl_quote.bid_price:.2f}")
            print(f"   Ask: ${aapl_quote.ask_price:.2f}")
            print(f"   Mid: ${(aapl_quote.bid_price + aapl_quote.ask_price) / 2:.2f}")
        else:
            print("❌ No AAPL quote available")
        
        # Test historical data
        print("\n5. Testing historical data...")
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        bars_request = StockBarsRequest(
            symbol_or_symbols="AAPL",
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = data_client.get_stock_bars(bars_request)
        
        if "AAPL" in bars:
            aapl_bars = bars["AAPL"]
            print(f"✅ AAPL Historical Data:")
            print(f"   Bars: {len(aapl_bars)}")
            if len(aapl_bars) > 0:
                latest_bar = aapl_bars[-1]
                print(f"   Latest Close: ${latest_bar.close:.2f}")
                print(f"   Latest Volume: {latest_bar.volume:,}")
        else:
            print("❌ No AAPL historical data available")
        
        print("\n" + "=" * 40)
        print("✅ ALPACA CONNECTION TEST PASSED!")
        print("Your Alpaca integration is working correctly.")
        print("You can now use Alpaca for trading.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install alpaca-py:")
        print("pip3 install alpaca-py")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("ALPACA CONNECTION TEST")
    print("=" * 40)
    print()
    print("This test will verify:")
    print("1. alpaca-py installation")
    print("2. API credentials")
    print("3. Account connection")
    print("4. Market data access")
    print("5. Historical data access")
    print()
    
    success = test_alpaca_connection()
    
    if not success:
        print("\n❌ Test failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n🎉 Ready to proceed with Alpaca integration!")

if __name__ == "__main__":
    main() 