#!/usr/bin/env python3
"""
Performance Test Suite
Test performance optimizations and compare original vs optimized versions.
"""

import sys
import os
import time
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

def test_signal_generation_performance():
    """Test signal generation performance."""
    print("\n" + "="*60)
    print("TESTING SIGNAL GENERATION PERFORMANCE")
    print("="*60)
    
    try:
        from src.trading.signals import OptimizedSignalGenerator
        
        # Create signal generator
        config = {
            'trading': {
                'max_position_size': 0.1,
                'max_positions': 10
            }
        }
        
        signal_gen = OptimizedSignalGenerator(config)
        
        # Create large test dataset
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        market_data = {}
        
        for symbol in symbols:
            dates = pd.date_range('2024-01-01', periods=500, freq='D')  # Large dataset
            base_price = 100 + np.random.randint(-20, 20)
            returns = np.random.normal(0, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            
            data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            
            market_data[symbol] = data
        
        # Test performance
        start_time = time.time()
        signals = signal_gen.generate_signals(market_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"✅ Signal generation performance test completed")
        print(f"   Symbols: {len(symbols)}")
        print(f"   Data points per symbol: {len(list(market_data.values())[0])}")
        print(f"   Total data points: {len(symbols) * len(list(market_data.values())[0])}")
        print(f"   Execution time: {execution_time:.3f} seconds")
        print(f"   Signals generated: {len(signals)}")
        print(f"   Performance: {len(symbols) * len(list(market_data.values())[0]) / execution_time:.0f} data points/second")
        
        # Performance threshold
        if execution_time < 5.0:  # Should complete in under 5 seconds
            print("✅ Performance threshold met")
            return True
        else:
            print("⚠️ Performance threshold exceeded")
            return False
        
    except Exception as e:
        print(f"❌ Signal generation performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtesting_performance():
    """Test backtesting performance."""
    print("\n" + "="*60)
    print("TESTING BACKTESTING PERFORMANCE")
    print("="*60)
    
    try:
        from src.trading.backtesting import BacktestingEngine
        from src.trading.signals import OptimizedSignalGenerator
        
        # Create components
        backtest_engine = BacktestingEngine(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        signal_generator = OptimizedSignalGenerator({
            'trading': {
                'max_position_size': 0.1,
                'max_positions': 10
            }
        })
        
        # Create test data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        market_data = {}
        
        for symbol in symbols:
            dates = pd.date_range('2024-01-01', periods=200, freq='D')
            base_price = 100 + np.random.randint(-20, 20)
            returns = np.random.normal(0, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            
            data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            
            market_data[symbol] = data
        
        # Test performance
        start_time = time.time()
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 7, 1)
        
        results = backtest_engine.run_backtest(
            signal_generator, 
            market_data, 
            start_date, 
            end_date
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"✅ Backtesting performance test completed")
        print(f"   Symbols: {len(symbols)}")
        print(f"   Date range: {start_date.date()} to {end_date.date()}")
        print(f"   Execution time: {execution_time:.3f} seconds")
        print(f"   Total return: {results.total_return:.2%}")
        print(f"   Total trades: {results.total_trades}")
        print(f"   Performance: {len(symbols) * len(list(market_data.values())[0]) / execution_time:.0f} data points/second")
        
        # Performance threshold
        if execution_time < 10.0:  # Should complete in under 10 seconds
            print("✅ Performance threshold met")
            return True
        else:
            print("⚠️ Performance threshold exceeded")
            return False
        
    except Exception as e:
        print(f"❌ Backtesting performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage optimization."""
    print("\n" + "="*60)
    print("TESTING MEMORY USAGE")
    print("="*60)
    
    try:
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        market_data = {}
        
        for symbol in symbols:
            dates = pd.date_range('2024-01-01', periods=1000, freq='D')  # Very large dataset
            base_price = 100 + np.random.randint(-20, 20)
            returns = np.random.normal(0, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            
            data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            
            market_data[symbol] = data
        
        # Get memory after data creation
        data_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test signal generation
        from src.trading.signals import OptimizedSignalGenerator
        
        config = {
            'trading': {
                'max_position_size': 0.1,
                'max_positions': 10
            }
        }
        
        signal_gen = OptimizedSignalGenerator(config)
        signals = signal_gen.generate_signals(market_data)
        
        # Get memory after signal generation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate memory usage
        data_memory_usage = data_memory - initial_memory
        total_memory_usage = final_memory - initial_memory
        signal_memory_usage = final_memory - data_memory
        
        print(f"✅ Memory usage test completed")
        print(f"   Initial memory: {initial_memory:.1f} MB")
        print(f"   Data memory usage: {data_memory_usage:.1f} MB")
        print(f"   Signal generation memory: {signal_memory_usage:.1f} MB")
        print(f"   Total memory usage: {total_memory_usage:.1f} MB")
        print(f"   Data points: {len(symbols) * len(list(market_data.values())[0])}")
        print(f"   Memory efficiency: {len(symbols) * len(list(market_data.values())[0]) / total_memory_usage:.0f} data points/MB")
        
        # Memory threshold (should be reasonable)
        if total_memory_usage < 1000:  # Less than 1GB for large dataset
            print("✅ Memory usage threshold met")
            return True
        else:
            print("⚠️ Memory usage threshold exceeded")
            return False
        
    except ImportError:
        print("⚠️ psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_caching_efficiency():
    """Test caching efficiency."""
    print("\n" + "="*60)
    print("TESTING CACHING EFFICIENCY")
    print("="*60)
    
    try:
        from src.trading.signals import OptimizedSignalGenerator
        
        # Create signal generator
        config = {
            'trading': {
                'max_position_size': 0.1,
                'max_positions': 10
            }
        }
        
        signal_gen = OptimizedSignalGenerator(config)
        
        # Create test data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        base_price = 100
        returns = np.random.normal(0, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        test_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        market_data = {"AAPL": test_data}
        
        # First run (no cache)
        start_time = time.time()
        signals1 = signal_gen.generate_signals(market_data)
        first_run_time = time.time() - start_time
        
        # Second run (with cache)
        start_time = time.time()
        signals2 = signal_gen.generate_signals(market_data)
        second_run_time = time.time() - start_time
        
        # Calculate speedup
        speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
        
        print(f"✅ Caching efficiency test completed")
        print(f"   First run time: {first_run_time:.3f} seconds")
        print(f"   Second run time: {second_run_time:.3f} seconds")
        print(f"   Speedup: {speedup:.1f}x")
        
        # Caching threshold
        if speedup > 1.5:  # Should be at least 1.5x faster
            print("✅ Caching efficiency threshold met")
            return True
        else:
            print("⚠️ Caching efficiency threshold not met")
            return False
        
    except Exception as e:
        print(f"❌ Caching efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all performance tests."""
    print("🧪 PERFORMANCE TEST SUITE")
    print("="*60)
    
    results = {
        "signal_generation": False,
        "backtesting": False,
        "memory_usage": False,
        "caching_efficiency": False
    }
    
    # Run tests
    results["signal_generation"] = test_signal_generation_performance()
    results["backtesting"] = test_backtesting_performance()
    results["memory_usage"] = test_memory_usage()
    results["caching_efficiency"] = test_caching_efficiency()
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE TEST RESULTS SUMMARY")
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
        print("🎉 All performance tests passed!")
        return True
    else:
        print("⚠️ Some performance tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 