#!/usr/bin/env python3
"""
Startup Script for Live Trading System
Simple script to start trading with minimal configuration.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading.cli import TradingCLI

def setup_environment():
    """Setup environment variables for trading."""
    # Check if Alpaca API keys are set
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
        print("⚠️  Alpaca API keys not found in environment variables")
        print("Please set the following environment variables:")
        print("  export ALPACA_API_KEY='your_api_key'")
        print("  export ALPACA_SECRET_KEY='your_secret_key'")
        
        # Try to load from .env file
        env_file = Path(".env")
        if env_file.exists():
            print("\nLoading from .env file...")
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            print("Environment variables loaded from .env")
        else:
            print("\nNo .env file found. Please set up your API keys.")
            return False
    
    return True

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(
        description="Start Live Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Start Examples:
  # Paper trading with default symbols
  python3 start_trading.py --paper
  
  # Live trading with specific symbols
  python3 start_trading.py --live --symbols AAPL MSFT GOOGL
  
  # Paper trading with custom interval
  python3 start_trading.py --paper --interval 10 --symbols TSLA NVDA
  
  # Check status only
  python3 start_trading.py --status
  
  # View performance
  python3 start_trading.py --performance
  
  # Emergency stop
  python3 start_trading.py --emergency-stop
        """
    )
    
    parser.add_argument('--live', action='store_true', 
                       help='Enable live trading (real money)')
    parser.add_argument('--paper', action='store_true', 
                       help='Enable paper trading (default)')
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
                       help='Trading symbols (default: AAPL MSFT GOOGL TSLA NVDA)')
    parser.add_argument('--interval', type=int, default=5,
                       help='Trading interval in minutes (default: 5)')
    parser.add_argument('--config', default='config/alpaca_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--status', action='store_true',
                       help='Show trading status only')
    parser.add_argument('--performance', action='store_true',
                       help='Show performance report only')
    parser.add_argument('--emergency-stop', action='store_true',
                       help='Activate emergency stop')
    parser.add_argument('--reason', help='Reason for emergency stop')
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        return 1
    
    # Initialize CLI
    cli = TradingCLI()
    cli.config_path = args.config
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        print("Please create the configuration file or specify a different path with --config")
        return 1
    
    # Handle different modes
    if args.emergency_stop:
        print("🚨 ACTIVATING EMERGENCY STOP")
        reason = args.reason or "Startup script emergency stop"
        return cli.emergency_stop(type('Args', (), {'reason': reason})())
    
    elif args.status:
        print("📊 Checking trading status...")
        return cli.status(type('Args', (), {})())
    
    elif args.performance:
        print("📈 Generating performance report...")
        return cli.performance(type('Args', (), {})())
    
    else:
        # Start trading
        print("🚀 Starting Live Trading System")
        print(f"Symbols: {', '.join(args.symbols)}")
        print(f"Interval: {args.interval} minutes")
        print(f"Config: {args.config}")
        
        if args.live:
            print("\n⚠️  LIVE TRADING MODE - REAL MONEY WILL BE USED ⚠️")
            confirm = input("Type 'YES' to confirm live trading: ")
            if confirm != "YES":
                print("❌ Live trading cancelled")
                return 0
            print("✅ Live trading confirmed")
        else:
            print("📝 PAPER TRADING MODE - No real money will be used")
        
        # Create args object for CLI
        cli_args = type('Args', (), {
            'symbols': args.symbols,
            'interval': args.interval,
            'live': args.live,
            'config': args.config
        })()
        
        return cli.start_trading(cli_args)

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Trading stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1) 