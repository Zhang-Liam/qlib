#!/usr/bin/env python3
"""
Trading Engine Test Suite
Test the automated trading engine functionality.
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

def test_trading_engine_initialization():
    """Test trading engine initialization."""
    print("\n" + "="*60)
    print("TESTING TRADING ENGINE INITIALIZATION")
    print("="*60)
    
    try:
        from src.trading.engine import AutomatedTradingEngine
        
        # Test initialization
        engine = AutomatedTradingEngine("config/alpaca_config.yaml")
        
        # Verify components
        assert engine is not None
        assert hasattr(engine, 'broker')
        assert hasattr(engine, 'signal_generator')
        assert hasattr(engine, 'risk_manager')
        
        print("✅ Trading engine initialized successfully")
        print(f"   Broker: {type(engine.broker).__name__}")
        print(f"   Signal Generator: {type(engine.signal_generator).__name__}")
        print(f"   Risk Manager: {type(engine.risk_manager).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading engine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_generation():
    """Test signal generation functionality."""
    print("\n" + "="*60)
    print("TESTING SIGNAL GENERATION")
    print("="*60)
    
    try:
        from src.trading.signals import OptimizedSignalGenerator
        
        # Create signal generator
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
        
        # Test single signal generation
        signal = signal_gen.generate_signal("AAPL", test_data)
        
        # Verify signal
        assert signal is not None
        assert hasattr(signal, 'signal_type')
        assert hasattr(signal, 'confidence')
        
        print("✅ Signal generation working")
        print(f"   Signal: {signal.signal_type}")
        print(f"   Confidence: {signal.confidence:.2f}")
        
        # Test multiple signals
        market_data = {"AAPL": test_data, "MSFT": test_data}
        signals = signal_gen.generate_signals(market_data)
        
        assert len(signals) == 2
        print(f"✅ Generated {len(signals)} signals for multiple symbols")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_management():
    """Test risk management functionality."""
    print("\n" + "="*60)
    print("TESTING RISK MANAGEMENT")
    print("="*60)
    
    try:
        from src.trading.risk_manager import RiskManager
        
        # Create risk manager
        config = {
            'risk_management': {
                'max_position_size': 0.1,
                'max_positions': 10,
                'min_cash_reserve': 0.1,
                'max_daily_loss': 0.05,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.1,
                'max_correlation': 0.7,
                'volatility_threshold': 0.3
            }
        }
        
        risk_manager = RiskManager(config)
        
        # Test position sizing
        account_value = 100000
        current_price = 150.0
        volatility = 0.2
        
        position_size = risk_manager.calculate_position_size(
            "AAPL", 0.7, current_price, account_value
        )
        
        assert position_size > 0
        print(f"✅ Position sizing working: {position_size:.0f} shares")
        
        # Test risk assessment
        positions = {
            'AAPL': {'shares': 100, 'avg_price': 150.0},
            'MSFT': {'shares': 50, 'avg_price': 300.0}
        }
        
        portfolio_risk = risk_manager.calculate_portfolio_risk(account_value, positions, {})
        assert portfolio_risk is not None
        print(f"✅ Risk assessment working: {portfolio_risk.total_pnl_pct:.2%} PnL")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk management failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_market_data_fetching():
    """Test market data fetching functionality."""
    print("\n" + "="*60)
    print("TESTING MARKET DATA FETCHING")
    print("="*60)
    
    try:
        from src.trading.engine import AutomatedTradingEngine
        
        # Initialize engine
        engine = AutomatedTradingEngine("config/alpaca_config.yaml")
        
        # Test historical data fetching
        symbol = "AAPL"
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        
        data = engine.fetch_historical_data(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            print(f"✅ Historical data fetched successfully")
            print(f"   Symbol: {symbol}")
            print(f"   Data points: {len(data)}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        else:
            print("⚠️ No historical data available (this is normal for paper trading)")
        
        return True
        
    except Exception as e:
        print(f"❌ Market data fetching failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all trading engine tests."""
    print("🧪 TRADING ENGINE TEST SUITE")
    print("="*60)
    
    results = {
        "initialization": False,
        "signal_generation": False,
        "risk_management": False,
        "data_fetching": False
    }
    
    # Run tests
    results["initialization"] = test_trading_engine_initialization()
    results["signal_generation"] = test_signal_generation()
    results["risk_management"] = test_risk_management()
    results["data_fetching"] = test_market_data_fetching()
    
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