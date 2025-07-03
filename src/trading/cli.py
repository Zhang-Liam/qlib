#!/usr/bin/env python3
"""
Command Line Interface for Live Trading System
Provides easy-to-use commands for managing live trading operations.
"""

import argparse
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any
import yaml
from pathlib import Path

from .live_trading import LiveTradingSystem, TradingMode

class TradingCLI:
    """Command line interface for live trading."""
    
    def __init__(self):
        self.live_system = None
        self.config_path = "config/alpaca_config.yaml"
    
    def init_system(self):
        """Initialize the live trading system."""
        try:
            self.live_system = LiveTradingSystem(self.config_path)
            return True
        except Exception as e:
            print(f"Failed to initialize trading system: {e}")
            return False
    
    def start_trading(self, args):
        """Start live trading."""
        if not self.init_system():
            return 1
        
        symbols = args.symbols or ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        interval = args.interval or 5
        
        print(f"Starting trading with symbols: {symbols}")
        print(f"Trading interval: {interval} minutes")
        
        if args.live:
            print("\n⚠️  WARNING: LIVE TRADING MODE - REAL MONEY WILL BE USED ⚠️")
            confirm = input("Type 'YES' to confirm live trading: ")
            if confirm != "YES":
                print("Live trading cancelled")
                return 0
            
            if not self.live_system.switch_to_live_trading():
                print("Failed to switch to live trading")
                return 1
        else:
            print("Starting in PAPER TRADING mode")
        
        try:
            self.live_system.start_live_trading(symbols, interval)
            
            print("\nTrading started successfully!")
            print("Press Ctrl+C to stop trading")
            
            # Monitor and display status
            while True:
                self.display_status()
                time.sleep(30)  # Update every 30 seconds
                
        except KeyboardInterrupt:
            print("\nStopping trading...")
            self.live_system.stop_live_trading()
            self.display_final_status()
            return 0
    
    def stop_trading(self, args):
        """Stop live trading."""
        if not self.init_system():
            return 1
        
        print("Stopping trading system...")
        self.live_system.stop_live_trading()
        print("Trading stopped")
        return 0
    
    def emergency_stop(self, args):
        """Activate emergency stop."""
        if not self.init_system():
            return 1
        
        reason = args.reason or "CLI emergency stop"
        print(f"Activating emergency stop: {reason}")
        
        self.live_system.activate_emergency_stop(reason)
        print("Emergency stop activated - all positions closed")
        return 0
    
    def status(self, args):
        """Display trading status."""
        if not self.init_system():
            return 1
        
        self.display_status()
        return 0
    
    def performance(self, args):
        """Display performance report."""
        if not self.init_system():
            return 1
        
        report = self.live_system.get_performance_report()
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        # Summary
        summary = report['summary']
        print(f"Total Return: {summary['total_return']:.2%}")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Win Rate: {summary['win_rate']:.2%}")
        print(f"Average Trade PnL: ${summary['avg_trade_pnl']:.2f}")
        print(f"Max Drawdown: {summary['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        
        # Current positions
        if report['positions']:
            print(f"\nCurrent Positions ({len(report['positions'])}):")
            print("-" * 60)
            for symbol, pos in report['positions'].items():
                pnl_color = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
                print(f"{symbol}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
                print(f"  Current: ${pos['current_price']:.2f} | PnL: {pnl_color} ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.2%})")
        
        # Recent trades
        if report['recent_trades']:
            print(f"\nRecent Trades ({len(report['recent_trades'])}):")
            print("-" * 60)
            for trade in report['recent_trades']:
                side_emoji = "🟢" if trade['side'] == 'buy' else "🔴"
                print(f"{trade['timestamp']} | {side_emoji} {trade['side'].upper()} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
        
        return 0
    
    def switch_mode(self, args):
        """Switch between paper and live trading."""
        if not self.init_system():
            return 1
        
        if args.mode == "live":
            if self.live_system.switch_to_live_trading():
                print("Switched to LIVE TRADING mode")
            else:
                print("Failed to switch to live trading")
                return 1
        elif args.mode == "paper":
            self.live_system.switch_to_paper_trading()
            print("Switched to PAPER TRADING mode")
        
        return 0
    
    def display_status(self):
        """Display current trading status."""
        if not self.live_system:
            return
        
        status = self.live_system.get_trading_status()
        
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        print("="*60)
        print("LIVE TRADING SYSTEM STATUS")
        print("="*60)
        print(f"Trading Mode: {status['trading_mode'].upper()}")
        print(f"Market Open: {'🟢 YES' if status['market_open'] else '🔴 NO'}")
        print(f"Emergency Stop: {'🔴 ACTIVE' if status['emergency_stop']['active'] else '🟢 INACTIVE'}")
        
        print(f"\nAccount:")
        print(f"  Value: ${status['account_value']:,.2f}")
        print(f"  Cash: ${status['cash']:,.2f}")
        print(f"  Total PnL: ${status['total_pnl']:,.2f}")
        print(f"  Daily PnL: ${status['daily_pnl']:,.2f}")
        
        print(f"\nPerformance:")
        print(f"  Total Trades: {status['total_trades']}")
        print(f"  Win Rate: {status['win_rate']:.2%}")
        print(f"  Active Positions: {status['positions']}")
        print(f"  Uptime: {status['uptime']:.1f} hours")
        
        print(f"\nSafety Checks:")
        safety_checks = status['safety_checks']
        for check, passed in safety_checks.items():
            status_emoji = "🟢" if passed else "🔴"
            print(f"  {check.replace('_', ' ').title()}: {status_emoji}")
        
        print(f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    
    def display_final_status(self):
        """Display final status when stopping."""
        if not self.live_system:
            return
        
        status = self.live_system.get_trading_status()
        
        print("\n" + "="*60)
        print("FINAL TRADING STATUS")
        print("="*60)
        print(f"Trading Mode: {status['trading_mode']}")
        print(f"Account Value: ${status['account_value']:,.2f}")
        print(f"Total PnL: ${status['total_pnl']:,.2f}")
        print(f"Total Trades: {status['total_trades']}")
        print(f"Win Rate: {status['win_rate']:.2%}")
        print(f"Uptime: {status['uptime']:.1f} hours")
        print("="*60)

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Live Trading System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start paper trading
  python3 -m src.trading.cli start --symbols AAPL MSFT GOOGL --interval 5
  
  # Start live trading (real money)
  python3 -m src.trading.cli start --live --symbols AAPL MSFT --interval 10
  
  # Check status
  python3 -m src.trading.cli status
  
  # View performance
  python3 -m src.trading.cli performance
  
  # Emergency stop
  python3 -m src.trading.cli emergency-stop --reason "Market crash"
  
  # Switch to live trading
  python3 -m src.trading.cli switch --mode live
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start trading')
    start_parser.add_argument('--symbols', nargs='+', help='Trading symbols')
    start_parser.add_argument('--interval', type=int, help='Trading interval in minutes')
    start_parser.add_argument('--live', action='store_true', help='Enable live trading (real money)')
    start_parser.add_argument('--config', default='config/alpaca_config.yaml', help='Configuration file')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop trading')
    
    # Emergency stop command
    emergency_parser = subparsers.add_parser('emergency-stop', help='Activate emergency stop')
    emergency_parser.add_argument('--reason', help='Reason for emergency stop')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Display trading status')
    
    # Performance command
    perf_parser = subparsers.add_parser('performance', help='Display performance report')
    
    # Switch mode command
    switch_parser = subparsers.add_parser('switch', help='Switch trading mode')
    switch_parser.add_argument('--mode', choices=['paper', 'live'], required=True, help='Trading mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize CLI
    cli = TradingCLI()
    cli.config_path = args.config if hasattr(args, 'config') else "config/alpaca_config.yaml"
    
    # Execute command
    if args.command == 'start':
        return cli.start_trading(args)
    elif args.command == 'stop':
        return cli.stop_trading(args)
    elif args.command == 'emergency-stop':
        return cli.emergency_stop(args)
    elif args.command == 'status':
        return cli.status(args)
    elif args.command == 'performance':
        return cli.performance(args)
    elif args.command == 'switch':
        return cli.switch_mode(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 