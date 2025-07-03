#!/usr/bin/env python3
"""
Advanced Backtesting Engine
Comprehensive backtesting system with realistic market simulation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TradeType(Enum):
    """Types of trades."""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    symbol: str
    trade_type: TradeType
    quantity: int
    price: float
    commission: float
    slippage: float
    signal_confidence: float
    signal_type: str

@dataclass
class Position:
    """Represents a position in a symbol."""
    symbol: str
    quantity: int
    average_price: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float

@dataclass
class BacktestResult:
    """Results of a backtest."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    equity_curve: pd.Series
    trades: List[Trade]
    positions: Dict[str, Position]
    daily_returns: pd.Series
    monthly_returns: pd.Series

class BacktestingEngine:
    """Advanced backtesting engine with realistic market simulation."""
    
    def __init__(self, initial_capital: float = 100000, commission_rate: float = 0.001, 
                 slippage_rate: float = 0.0005):
        """Initialize backtesting engine."""
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def reset(self):
        """Reset backtesting engine to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve = []
        self.daily_returns = []
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
    
    def calculate_slippage(self, price: float, quantity: int, trade_type: TradeType) -> float:
        """Calculate slippage based on trade size and type."""
        # Simple slippage model: larger trades = more slippage
        slippage_multiplier = min(quantity / 1000, 1.0)  # Cap at 1000 shares
        base_slippage = price * self.slippage_rate * slippage_multiplier
        
        # Add direction bias (buying pushes price up, selling pushes down)
        if trade_type == TradeType.BUY:
            return base_slippage
        else:
            return -base_slippage
    
    def calculate_commission(self, price: float, quantity: int) -> float:
        """Calculate commission for a trade."""
        trade_value = price * quantity
        return trade_value * self.commission_rate
    
    def execute_trade(self, timestamp: datetime, symbol: str, trade_type: TradeType, 
                     quantity: int, price: float, signal_confidence: float, signal_type: str) -> bool:
        """Execute a trade with realistic market conditions."""
        
        # Calculate costs
        commission = self.calculate_commission(price, quantity)
        slippage = self.calculate_slippage(price, quantity, trade_type)
        
        # Adjust price for slippage
        execution_price = price + slippage
        
        # Calculate total cost
        total_cost = execution_price * quantity + commission
        
        # Check if we have enough cash for buy orders
        if trade_type == TradeType.BUY:
            if total_cost > self.cash:
                self.logger.warning(f"Insufficient cash for {symbol} buy order: {total_cost} > {self.cash}")
                return False
            
            # Update cash
            self.cash -= total_cost
            
            # Update position
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_quantity = pos.quantity + quantity
                total_cost_basis = (pos.average_price * pos.quantity) + (execution_price * quantity)
                pos.average_price = total_cost_basis / total_quantity
                pos.quantity = total_quantity
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=execution_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    total_pnl=0.0
                )
        
        else:  # SELL
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                self.logger.warning(f"Insufficient shares for {symbol} sell order")
                return False
            
            # Update cash
            self.cash += (execution_price * quantity - commission)
            
            # Update position
            pos = self.positions[symbol]
            pos.quantity -= quantity
            
            # Calculate realized P&L
            realized_pnl = (execution_price - pos.average_price) * quantity - commission
            pos.realized_pnl += realized_pnl
            
            # Remove position if quantity becomes zero
            if pos.quantity == 0:
                del self.positions[symbol]
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            trade_type=trade_type,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage,
            signal_confidence=signal_confidence,
            signal_type=signal_type
        )
        self.trades.append(trade)
        
        # Update consecutive wins/losses
        if trade_type == TradeType.SELL and realized_pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
        elif trade_type == TradeType.SELL and realized_pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        return True
    
    def update_positions(self, timestamp: datetime, market_data: Dict[str, pd.DataFrame]):
        """Update unrealized P&L for all positions."""
        for symbol, position in self.positions.items():
            try:
                if symbol in market_data and len(market_data[symbol]) > 0:
                    # Try to get the exact timestamp, fall back to latest data
                    if timestamp in market_data[symbol].index:
                        current_price = market_data[symbol].loc[timestamp, 'close']
                    else:
                        current_price = market_data[symbol].iloc[-1]['close']
                    
                    position.unrealized_pnl = (current_price - position.average_price) * position.quantity
                    position.total_pnl = position.realized_pnl + position.unrealized_pnl
            except Exception as e:
                self.logger.warning(f"Error updating position for {symbol}: {e}")
    
    def calculate_equity(self, timestamp: datetime, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio equity."""
        self.update_positions(timestamp, market_data)
        
        equity = self.cash
        for position in self.positions.values():
            equity += position.total_pnl
        
        # Update peak equity and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        current_drawdown = (self.peak_equity - equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        return equity
    
    def run_backtest(self, strategy, market_data: Dict[str, pd.DataFrame], 
                    start_date: datetime, end_date: datetime) -> BacktestResult:
        """Run a complete backtest."""
        self.reset()
        
        # Validate market data
        if not market_data:
            raise ValueError("No market data provided for backtesting")
        
        # Check data length for each symbol
        for symbol, data in market_data.items():
            if len(data) < 20:  # Minimum data points needed
                raise ValueError(f"Insufficient data for {symbol}: {len(data)} points (minimum 20 required)")
        
        # Prepare date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        equity_values = []
        daily_returns = []
        
        for date in date_range:
            # Get market data for this date
            current_data = {}
            for symbol, data in market_data.items():
                # Handle different data structures
                if hasattr(data.index, 'date') and date.date() in [d.date() for d in data.index]:
                    # Datetime index
                    current_data[symbol] = data.loc[:date]
                elif 'date' in data.columns:
                    # Date column - find matching date
                    matching_data = data[data['date'].dt.date == date.date()]
                    if len(matching_data) > 0:
                        # Get all data up to this date
                        date_mask = data['date'].dt.date <= date.date()
                        current_data[symbol] = data[date_mask]
                else:
                    # Numeric index - use all data
                    current_data[symbol] = data
            
            if not current_data:
                continue
            
            # Generate signals
            try:
                signals = strategy.generate_signals(current_data)
            except Exception as e:
                self.logger.warning(f"Error generating signals for {date}: {e}")
                signals = {}
            
            # Execute trades based on signals
            for symbol, signal in signals.items():
                if symbol not in current_data or len(current_data[symbol]) == 0:
                    continue
                
                try:
                    current_price = current_data[symbol].iloc[-1]['close']
                except IndexError:
                    self.logger.warning(f"No data available for {symbol} on {date}")
                    continue
                
                # Determine trade action
                if signal.signal_type in ['strong_buy', 'buy']:
                    # Calculate position size (simplified)
                    available_cash = self.cash * 0.1  # Use 10% of cash per trade
                    quantity = int(available_cash / current_price)
                    
                    if quantity > 0:
                        self.execute_trade(
                            timestamp=date,
                            symbol=symbol,
                            trade_type=TradeType.BUY,
                            quantity=quantity,
                            price=current_price,
                            signal_confidence=signal.confidence,
                            signal_type=signal.signal_type
                        )
                
                elif signal.signal_type in ['strong_sell', 'sell']:
                    if symbol in self.positions:
                        quantity = self.positions[symbol].quantity
                        if quantity > 0:
                            self.execute_trade(
                                timestamp=date,
                                symbol=symbol,
                                trade_type=TradeType.SELL,
                                quantity=quantity,
                                price=current_price,
                                signal_confidence=signal.confidence,
                                signal_type=signal.signal_type
                            )
            
            # Calculate equity for this date
            try:
                equity = self.calculate_equity(date, current_data)
                equity_values.append(equity)
                
                # Calculate daily return
                if len(equity_values) > 1:
                    daily_return = (equity - equity_values[-2]) / equity_values[-2]
                    daily_returns.append(daily_return)
                else:
                    daily_returns.append(0.0)
            except Exception as e:
                self.logger.warning(f"Error calculating equity for {date}: {e}")
                # Use previous equity value or initial capital
                if equity_values:
                    equity_values.append(equity_values[-1])
                else:
                    equity_values.append(self.initial_capital)
                daily_returns.append(0.0)
        
        # Ensure we have equity values
        if not equity_values:
            self.logger.warning("No equity values generated during backtest")
            # Create a simple result with no trading
            return BacktestResult(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                average_win=0.0,
                average_loss=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                equity_curve=pd.Series([self.initial_capital]),
                trades=[],
                positions={},
                daily_returns=pd.Series([0.0]),
                monthly_returns=pd.Series([0.0])
            )
        
        # Calculate performance metrics
        return self._calculate_performance_metrics(equity_values, daily_returns)
    
    def _calculate_performance_metrics(self, equity_values: List[float], 
                                     daily_returns: List[float]) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        
        equity_series = pd.Series(equity_values)
        returns_series = pd.Series(daily_returns)
        
        # Basic metrics
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return (assuming 252 trading days)
        days = len(equity_series)
        annualized_return = ((equity_series.iloc[-1] / self.initial_capital) ** (252 / days)) - 1
        
        # Sharpe ratio
        if returns_series.std() > 0:
            sharpe_ratio = (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0.0
        
        # Win rate and trade analysis
        winning_trades = [t for t in self.trades if t.trade_type == TradeType.SELL and 
                         (t.price - self._get_position_avg_price(t.symbol)) * t.quantity > 0]
        losing_trades = [t for t in self.trades if t.trade_type == TradeType.SELL and 
                        (t.price - self._get_position_avg_price(t.symbol)) * t.quantity < 0]
        
        total_trades = len([t for t in self.trades if t.trade_type == TradeType.SELL])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        # Profit factor
        gross_profit = sum([(t.price - self._get_position_avg_price(t.symbol)) * t.quantity 
                           for t in winning_trades])
        gross_loss = abs(sum([(t.price - self._get_position_avg_price(t.symbol)) * t.quantity 
                             for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average win/loss
        average_win = gross_profit / len(winning_trades) if winning_trades else 0.0
        average_loss = gross_loss / len(losing_trades) if losing_trades else 0.0
        
        # Monthly returns
        equity_df = pd.DataFrame({'equity': equity_series})
        equity_df.index = pd.date_range(start=equity_df.index[0], periods=len(equity_df), freq='D')
        monthly_returns = equity_df['equity'].resample('M').last().pct_change().dropna()
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            average_win=average_win,
            average_loss=average_loss,
            max_consecutive_wins=self.max_consecutive_wins,
            max_consecutive_losses=self.max_consecutive_losses,
            equity_curve=equity_series,
            trades=self.trades,
            positions=self.positions.copy(),
            daily_returns=returns_series,
            monthly_returns=monthly_returns
        )
    
    def _get_position_avg_price(self, symbol: str) -> float:
        """Get average price for a position (simplified)."""
        # This is a simplified version - in reality, you'd track this properly
        for trade in reversed(self.trades):
            if trade.symbol == symbol and trade.trade_type == TradeType.BUY:
                return trade.price
        return 0.0
    
    def plot_results(self, results: BacktestResult, save_path: Optional[str] = None):
        """Plot comprehensive backtest results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Equity curve
        axes[0, 0].plot(results.equity_curve.index, results.equity_curve.values)
        axes[0, 0].set_title('Portfolio Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Equity ($)')
        axes[0, 0].grid(True)
        
        # Drawdown
        running_max = results.equity_curve.expanding().max()
        drawdown = (results.equity_curve - running_max) / running_max
        axes[0, 1].fill_between(results.equity_curve.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # Monthly returns
        axes[1, 0].bar(range(len(results.monthly_returns)), results.monthly_returns.values)
        axes[1, 0].set_title('Monthly Returns')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].grid(True)
        
        # Performance metrics
        metrics_text = f"""
        Total Return: {results.total_return:.2%}
        Annualized Return: {results.annualized_return:.2%}
        Sharpe Ratio: {results.sharpe_ratio:.2f}
        Max Drawdown: {results.max_drawdown:.2%}
        Win Rate: {results.win_rate:.2%}
        Total Trades: {results.total_trades}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, results: BacktestResult) -> str:
        """Generate a comprehensive backtest report."""
        report = f"""
        ========================================
        BACKTEST RESULTS REPORT
        ========================================
        
        PERFORMANCE METRICS:
        - Total Return: {results.total_return:.2%}
        - Annualized Return: {results.annualized_return:.2%}
        - Sharpe Ratio: {results.sharpe_ratio:.2f}
        - Maximum Drawdown: {results.max_drawdown:.2%}
        - Win Rate: {results.win_rate:.2%}
        - Profit Factor: {results.profit_factor:.2f}
        
        TRADE STATISTICS:
        - Total Trades: {results.total_trades}
        - Winning Trades: {results.winning_trades}
        - Losing Trades: {results.losing_trades}
        - Average Win: ${results.average_win:.2f}
        - Average Loss: ${results.average_loss:.2f}
        - Max Consecutive Wins: {results.max_consecutive_wins}
        - Max Consecutive Losses: {results.max_consecutive_losses}
        
        RISK METRICS:
        - Volatility: {results.daily_returns.std() * np.sqrt(252):.2%}
        - VaR (95%): {np.percentile(results.daily_returns, 5):.2%}
        - CVaR (95%): {results.daily_returns[results.daily_returns <= np.percentile(results.daily_returns, 5)].mean():.2%}
        
        ========================================
        """
        return report 