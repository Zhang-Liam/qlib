#!/usr/bin/env python3
"""
Debug Backtest
Test backtesting with actual Alpaca data to identify the indexer error.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_backtest_with_alpaca_data():
    """Test backtesting with actual Alpaca data."""
    print("Testing backtesting with Alpaca data...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        from src.trading.engine import AutomatedTradingEngine
        from src.trading.backtesting import BacktestingEngine
        
        # Initialize components
        trading_engine = AutomatedTradingEngine("config/alpaca_config.yaml")
        backtest_engine = BacktestingEngine(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Test symbols
        symbols = ["AAPL", "MSFT"]
        start_date = "2024-01-01"
        end_date = "2024-06-30"
        
        print(f"Fetching data for {symbols} from {start_date} to {end_date}")
        
        # Fetch historical data
        market_data = {}
        for symbol in symbols:
            print(f"\nFetching data for {symbol}...")
            data = trading_engine.fetch_historical_data(symbol, start_date, end_date)
            
            if data is not None and not data.empty:
                print(f"✅ {symbol}: {len(data)} rows")
                print(f"   Columns: {list(data.columns)}")
                print(f"   Index: {data.index[0]} to {data.index[-1]}")
                print(f"   Sample data:")
                print(f"   {data.head(3)}")
                market_data[symbol] = data
            else:
                print(f"❌ {symbol}: No data returned")
        
        if not market_data:
            print("❌ No market data available")
            return False
        
        print(f"\nRunning backtest with {len(market_data)} symbols...")
        
        # Test signal generation first
        print("\nTesting signal generation with actual data...")
        from src.trading.signals import OptimizedSignalGenerator
        
        config = {
            'trading': {
                'max_position_size': 0.1,
                'max_positions': 10,
                'min_cash_reserve': 0.1,
                'max_daily_loss': 0.05,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.1
            }
        }
        
        signal_gen = OptimizedSignalGenerator(config)
        
        # Test signals for each symbol
        for symbol, data in market_data.items():
            print(f"\nTesting signals for {symbol}...")
            try:
                signals = signal_gen.generate_signals({symbol: data})
                print(f"✅ Generated {len(signals)} signals for {symbol}")
                if symbol in signals:
                    signal = signals[symbol]
                    print(f"   Signal: {signal.signal_type.value} (confidence: {signal.confidence})")
            except Exception as e:
                print(f"❌ Error generating signals for {symbol}: {e}")
                import traceback
                traceback.print_exc()
        
        # Run backtest
        print(f"\nRunning full backtest...")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        results = backtest_engine.run_backtest(
            signal_gen, 
            market_data, 
            start_dt, 
            end_dt
        )
        
        print(f"✅ Backtest completed!")
        print(f"   Total Return: {results.total_return:.2%}")
        print(f"   Total Trades: {results.total_trades}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_backtest_with_alpaca_data()
    if success:
        print("\n🎉 Backtest debug completed successfully!")
    else:
        print("\n❌ Backtest debug failed!") 