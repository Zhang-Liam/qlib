#!/usr/bin/env python3
"""
Integrated Trading System
Combines backtesting, real-time streaming, and ML-based signal generation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Import our modules
from .engine import AutomatedTradingEngine
from .backtesting import BacktestingEngine, BacktestResult
from .realtime import RealTimeTradingEngine, RealTimeDataStream

# Optional ML imports
try:
    from .ml_signals import MLSignalGenerator, EnsembleSignalGenerator
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class IntegratedTradingSystem:
    """Integrated trading system combining all features."""
    
    def __init__(self, config_path: str = "config/alpaca_config.yaml"):
        """Initialize integrated trading system."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.trading_engine = AutomatedTradingEngine(self.config_path)
        self.backtesting_engine = BacktestingEngine(
            initial_capital=self.config.get('backtesting', {}).get('initial_capital', 100000),
            commission_rate=self.config.get('backtesting', {}).get('commission_rate', 0.001),
            slippage_rate=self.config.get('backtesting', {}).get('slippage_rate', 0.0005)
        )
        
        # ML components
        if ML_AVAILABLE:
            try:
                self.ml_generator = MLSignalGenerator(self.config)
                self.ensemble_generator = EnsembleSignalGenerator(self.config)
                self.ensemble_generator.set_technical_generator(self.trading_engine.signal_generator)
                self.ml_available = True
                self.logger.info("ML components initialized successfully")
            except Exception as e:
                self.logger.warning(f"ML components initialization failed: {e}")
                self.ml_available = False
        else:
            self.logger.warning("ML components not available")
            self.ml_available = False
        
        # Real-time components
        self.realtime_engine = None
        self.realtime_active = False
        
        # State
        self.current_mode = "paper_trading"  # paper_trading, backtesting, realtime
        
    def _load_config(self) -> dict:
        """Load configuration."""
        import yaml
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/integrated_trading.log'),
                logging.StreamHandler()
            ]
        )
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str, 
                    use_ml_signals: bool = True) -> Optional[BacktestResult]:
        """Run comprehensive backtest with optional ML signals."""
        self.logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
        
        # Fetch historical data
        market_data = {}
        for symbol in symbols:
            try:
                data = self.trading_engine.fetch_historical_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    market_data[symbol] = data
                    self.logger.info(f"Loaded {len(data)} data points for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
        
        if not market_data:
            self.logger.error("No market data available for backtesting")
            return None
        
        # Choose signal generator
        if use_ml_signals and self.ml_available:
            self.logger.info("Using ML-based signals for backtesting")
            strategy = self.ensemble_generator
        else:
            self.logger.info("Using technical analysis signals for backtesting")
            strategy = self.trading_engine.signal_generator
        
        # Run backtest
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        results = self.backtesting_engine.run_backtest(strategy, market_data, start_dt, end_dt)
        
        # Generate report
        report = self.backtesting_engine.generate_report(results)
        self.logger.info(report)
        
        # Save results
        self._save_backtest_results(results, symbols, start_date, end_date, use_ml_signals)
        
        return results
    
    def _save_backtest_results(self, results: BacktestResult, symbols: List[str], 
                             start_date: str, end_date: str, use_ml_signals: bool):
        """Save backtest results to file."""
        try:
            results_dir = Path("backtest_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            signal_type = "ml" if use_ml_signals else "technical"
            filename = f"backtest_{signal_type}_{'_'.join(symbols)}_{start_date}_{end_date}_{timestamp}.json"
            
            # Convert results to serializable format
            results_dict = {
                'total_return': results.total_return,
                'annualized_return': results.annualized_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'average_win': results.average_win,
                'average_loss': results.average_loss,
                'max_consecutive_wins': results.max_consecutive_wins,
                'max_consecutive_losses': results.max_consecutive_losses,
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'signal_type': signal_type,
                'timestamp': timestamp
            }
            
            with open(results_dir / filename, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            self.logger.info(f"Backtest results saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")
    
    async def start_realtime_trading(self, symbols: List[str], use_ml_signals: bool = True):
        """Start real-time trading with streaming data."""
        self.logger.info(f"Starting real-time trading for {symbols}")
        
        if not self.ml_available and use_ml_signals:
            self.logger.warning("ML not available, falling back to technical signals")
            use_ml_signals = False
        
        # Initialize real-time engine
        self.realtime_engine = RealTimeTradingEngine(self.config)
        
        # Start real-time trading
        success = await self.realtime_engine.start(symbols)
        
        if success:
            self.realtime_active = True
            self.current_mode = "realtime"
            self.logger.info("Real-time trading started successfully")
            
            # Keep running
            try:
                while self.realtime_active:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Stopping real-time trading...")
                await self.stop_realtime_trading()
        else:
            self.logger.error("Failed to start real-time trading")
    
    async def stop_realtime_trading(self):
        """Stop real-time trading."""
        if self.realtime_engine and self.realtime_active:
            await self.realtime_engine.stop()
            self.realtime_active = False
            self.current_mode = "paper_trading"
            self.logger.info("Real-time trading stopped")
    
    def start_paper_trading(self, symbols: List[str], use_ml_signals: bool = True, 
                           continuous: bool = True):
        """Start paper trading with optional ML signals."""
        self.logger.info(f"Starting paper trading for {symbols}")
        
        if not self.ml_available and use_ml_signals:
            self.logger.warning("ML not available, falling back to technical signals")
            use_ml_signals = False
        
        # Set signal generator
        if use_ml_signals:
            self.trading_engine.signal_generator = self.ensemble_generator
            self.logger.info("Using ensemble ML signals")
        else:
            self.logger.info("Using technical analysis signals")
        
        # Start trading
        self.current_mode = "paper_trading"
        self.trading_engine.run_trading_cycle(symbols, continuous=continuous)
    
    def train_ml_models(self, symbols: List[str], start_date: str, end_date: str):
        """Train ML models with historical data."""
        if not self.ml_available:
            self.logger.error("ML components not available")
            return False
        
        self.logger.info(f"Training ML models with data from {start_date} to {end_date}")
        
        # Fetch training data
        market_data = {}
        for symbol in symbols:
            try:
                data = self.trading_engine.fetch_historical_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    market_data[symbol] = data
            except Exception as e:
                self.logger.error(f"Error fetching training data for {symbol}: {e}")
        
        if not market_data:
            self.logger.error("No training data available")
            return False
        
        # Train models
        success = self.ml_generator.train_models(market_data)
        
        if success:
            self.logger.info("ML models trained successfully")
            
            # Save models
            models_dir = Path("ml_models")
            models_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = models_dir / f"trained_models_{timestamp}.joblib"
            self.ml_generator.save_models(str(model_path))
            
            # Evaluate models
            evaluation = self.ml_generator.evaluate_models(market_data)
            self.logger.info("Model evaluation:")
            for model_name, metrics in evaluation.items():
                self.logger.info(f"{model_name}: Accuracy={metrics['accuracy']:.3f}, "
                               f"F1={metrics['f1_score']:.3f}")
        
        return success
    
    def load_ml_models(self, model_path: str):
        """Load pre-trained ML models."""
        if not self.ml_available:
            self.logger.error("ML components not available")
            return False
        
        try:
            self.ml_generator.load_models(model_path)
            self.logger.info(f"ML models loaded from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading ML models: {e}")
            return False
    
    def compare_strategies(self, symbols: List[str], start_date: str, end_date: str):
        """Compare different signal generation strategies."""
        self.logger.info("Comparing trading strategies...")
        
        strategies = {
            'technical': self.trading_engine.signal_generator,
            'ml': self.ml_generator if self.ml_available else None,
            'ensemble': self.ensemble_generator if self.ml_available else None
        }
        
        results = {}
        
        for strategy_name, strategy in strategies.items():
            if strategy is None:
                continue
            
            self.logger.info(f"Testing {strategy_name} strategy...")
            
            # Run backtest for this strategy
            backtest_engine = BacktestingEngine(
                initial_capital=self.config.get('backtesting', {}).get('initial_capital', 100000),
                commission_rate=self.config.get('backtesting', {}).get('commission_rate', 0.001),
                slippage_rate=self.config.get('backtesting', {}).get('slippage_rate', 0.0005)
            )
            
            # Fetch data
            market_data = {}
            for symbol in symbols:
                data = self.trading_engine.fetch_historical_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    market_data[symbol] = data
            
            if market_data:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
                result = backtest_engine.run_backtest(strategy, market_data, start_dt, end_dt)
                results[strategy_name] = result
        
        # Compare results
        self._print_strategy_comparison(results)
        
        return results
    
    def _print_strategy_comparison(self, results: Dict[str, BacktestResult]):
        """Print strategy comparison results."""
        print("\n" + "="*80)
        print("STRATEGY COMPARISON RESULTS")
        print("="*80)
        
        for strategy_name, result in results.items():
            print(f"\n{strategy_name.upper()} STRATEGY:")
            print(f"  Total Return: {result.total_return:.2%}")
            print(f"  Annualized Return: {result.annualized_return:.2%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {result.max_drawdown:.2%}")
            print(f"  Win Rate: {result.win_rate:.2%}")
            print(f"  Total Trades: {result.total_trades}")
        
        print("\n" + "="*80)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            'mode': self.current_mode,
            'components': {
                'ml_available': self.ml_available,
                'realtime_available': self.realtime_engine is not None,
                'backtesting_available': self.backtesting_engine is not None,
                'trading_engine_available': self.trading_engine is not None
            },
            'realtime_active': self.realtime_active,
            'trading_engine_status': 'active' if self.trading_engine else 'inactive',
            'last_update': datetime.now().isoformat()
        }
        
        if self.trading_engine:
            try:
                status['account_info'] = self.trading_engine.get_account_info()
                status['positions'] = self.trading_engine.get_positions()
            except Exception as e:
                status['account_info'] = None
                status['positions'] = None
        
        return status

def main():
    """Main function to run the integrated trading system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated Trading System")
    parser.add_argument("--mode", choices=["backtest", "paper", "realtime", "train", "compare"], 
                       required=True, help="Trading mode")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOGL"], 
                       help="Trading symbols")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date for backtesting")
    parser.add_argument("--end-date", default="2024-12-31", help="End date for backtesting")
    parser.add_argument("--use-ml", action="store_true", help="Use ML signals")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--config", default="config/alpaca_config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Initialize system
    system = IntegratedTradingSystem(args.config)
    
    try:
        if args.mode == "backtest":
            results = system.run_backtest(args.symbols, args.start_date, args.end_date, args.use_ml)
            if results:
                print("Backtest completed successfully!")
        
        elif args.mode == "paper":
            system.start_paper_trading(args.symbols, args.use_ml, args.continuous)
        
        elif args.mode == "realtime":
            asyncio.run(system.start_realtime_trading(args.symbols, args.use_ml))
        
        elif args.mode == "train":
            success = system.train_ml_models(args.symbols, args.start_date, args.end_date)
            if success:
                print("ML models trained successfully!")
        
        elif args.mode == "compare":
            system.compare_strategies(args.symbols, args.start_date, args.end_date)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
        if system.realtime_active:
            asyncio.run(system.stop_realtime_trading())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 