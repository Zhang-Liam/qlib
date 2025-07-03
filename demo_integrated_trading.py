#!/usr/bin/env python3
"""
Demo Script: Using All Three Features
This script demonstrates how to use backtesting, real-time streaming, and ML signals.
"""

import asyncio
from integrated_trading_system import IntegratedTradingSystem

async def main():
    # Initialize the integrated trading system
    system = IntegratedTradingSystem("config/alpaca_config.yaml")
    
    # Define trading parameters
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2024-01-01"
    end_date = "2024-06-30"
    
    print("🚀 INTEGRATED TRADING SYSTEM DEMO")
    print("="*50)
    
    # 1. Train ML Models
    print("
1️⃣ Training ML Models...")
    success = system.train_ml_models(symbols, start_date, end_date)
    if success:
        print("✅ ML models trained successfully!")
    
    # 2. Run Backtest with ML Signals
    print("
2️⃣ Running Backtest with ML Signals...")
    results = system.run_backtest(symbols, start_date, end_date, use_ml_signals=True)
    if results:
        print(f"✅ Backtest completed! Total return: {results.total_return:.2%}")
    
    # 3. Compare Strategies
    print("
3️⃣ Comparing Different Strategies...")
    system.compare_strategies(symbols, start_date, end_date)
    
    # 4. Start Paper Trading with ML
    print("
4️⃣ Starting Paper Trading with ML Signals...")
    print("Press Ctrl+C to stop")
    try:
        system.start_paper_trading(symbols, use_ml_signals=True, continuous=True)
    except KeyboardInterrupt:
        print("
⏹️ Paper trading stopped")
    
    # 5. Real-time Trading (optional)
    print("
5️⃣ Real-time Trading Demo...")
    print("This will connect to live market data")
    print("Press Ctrl+C to stop after a few seconds")
    
    try:
        await system.start_realtime_trading(symbols, use_ml_signals=True)
    except KeyboardInterrupt:
        await system.stop_realtime_trading()
        print("
⏹️ Real-time trading stopped")

if __name__ == "__main__":
    asyncio.run(main())
