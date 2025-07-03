#!/usr/bin/env python3
"""
Automated Stock Selection Script
Selects the best stocks for trading based on multiple criteria.
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading.stock_selector import AutomatedStockSelector

async def main():
    """Main function for stock selection."""
    parser = argparse.ArgumentParser(description="Automated Stock Selection")
    parser.add_argument('--max-stocks', type=int, default=10, help='Maximum number of stocks to select')
    parser.add_argument('--save', action='store_true', help='Save selected stocks to file')
    parser.add_argument('--output', default='selected_stocks.txt', help='Output file for selected stocks')
    
    args = parser.parse_args()
    
    print("🔍 Automated Stock Selection System")
    print("=" * 50)
    
    # Initialize stock selector
    selector = AutomatedStockSelector()
    
    print(f"Analyzing {len(selector.get_stock_universe())} stocks...")
    print("Criteria: Momentum, Volatility, Volume, Technical Indicators, Fundamentals")
    print()
    
    # Select best stocks
    selected_stocks = await selector.select_best_stocks(args.max_stocks)
    
    print(f"📊 Top {len(selected_stocks)} Stocks Selected:")
    print("=" * 80)
    
    # Display results
    for i, stock in enumerate(selected_stocks, 1):
        print(f"{i:2d}. {stock.symbol:6s} | Total Score: {stock.total_score:.3f}")
        print(f"     Momentum: {stock.momentum_score:.2f} | Volatility: {stock.volatility_score:.2f} | Volume: {stock.volume_score:.2f}")
        print(f"     Technical: {stock.technical_score:.2f} | Fundamental: {stock.fundamental_score:.2f}")
        print(f"     Sector: {stock.sector_score:.2f} | Market Cap: {stock.market_cap_score:.2f}")
        print(f"     Top Reasons: {', '.join(stock.reasons[:3])}")
        print()
    
    # Save to file if requested
    if args.save:
        symbols = [stock.symbol for stock in selected_stocks]
        with open(args.output, 'w') as f:
            f.write(f"# Automatically Selected Stocks - {len(symbols)} stocks\n")
            f.write(f"# Generated on: {asyncio.get_event_loop().time()}\n\n")
            f.write(" ".join(symbols))
            f.write("\n")
        
        print(f"💾 Selected stocks saved to: {args.output}")
        print(f"   Use with: python3 start_trading.py --symbols {' '.join(symbols)}")
    
    # Show usage examples
    print("\n🚀 Usage Examples:")
    print("=" * 50)
    symbols_str = " ".join([stock.symbol for stock in selected_stocks])
    print(f"# Paper trading with selected stocks:")
    print(f"python3 start_trading.py --paper --symbols {symbols_str}")
    print()
    print(f"# Live trading with selected stocks:")
    print(f"python3 start_trading.py --live --symbols {symbols_str}")
    print()
    print(f"# Auto-select stocks (recommended):")
    print(f"python3 start_trading.py --paper --auto-select --max-stocks {args.max_stocks}")

if __name__ == "__main__":
    asyncio.run(main()) 