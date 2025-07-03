#!/usr/bin/env python3
"""
Debug Signal Test
Simple test to isolate the signal generation issue.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create minimal test data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data

def test_signal_generation():
    """Test signal generation with minimal data."""
    print("Testing signal generation...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

        from src.trading.signals import OptimizedSignalGenerator
        
        # Create test config
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
        
        # Create signal generator
        signal_gen = OptimizedSignalGenerator(config)
        print("✅ Signal generator created")
        
        # Create test data
        test_data = create_test_data()
        print(f"✅ Test data created: {len(test_data)} rows")
        print(f"   Columns: {list(test_data.columns)}")
        print(f"   Index: {test_data.index[0]} to {test_data.index[-1]}")
        
        # Test single signal generation
        print("\nTesting single signal generation...")
        signal = signal_gen.generate_signal("TEST", test_data)
        print(f"✅ Signal generated: {signal.signal_type.value} (confidence: {signal.confidence})")
        
        # Test multiple signals generation
        print("\nTesting multiple signals generation...")
        market_data = {"AAPL": test_data, "MSFT": test_data}
        signals = signal_gen.generate_signals(market_data)
        print(f"✅ Generated {len(signals)} signals")
        
        for symbol, signal in signals.items():
            print(f"   {symbol}: {signal.signal_type.value} (confidence: {signal.confidence})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_signal_generation()
    if success:
        print("\n🎉 Signal generation test passed!")
    else:
        print("\n❌ Signal generation test failed!") 