#!/usr/bin/env python3
"""
Advanced Risk Management System
Handles position sizing, stop losses, portfolio limits, and market risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    """Risk levels for position sizing."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class PositionRisk:
    """Risk metrics for a position."""
    symbol: str
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    var_95: float  # 95% Value at Risk
    max_drawdown: float
    days_held: int
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    var_95: float
    max_drawdown: float
    sharpe_ratio: float
    beta: float
    correlation: float
    sector_concentration: Dict[str, float]
    position_concentration: Dict[str, float]

class RiskManager:
    """Advanced risk management system."""
    
    def __init__(self, config: dict):
        """Initialize risk manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_position_size = config['trading']['max_position_size']
        self.max_positions = config['trading']['max_positions']
        self.min_cash_reserve = config['trading']['min_cash_reserve']
        self.max_daily_loss = config['trading']['max_daily_loss']
        self.stop_loss_pct = config['trading']['stop_loss_pct']
        self.take_profit_pct = config['trading']['take_profit_pct']
        
        # Risk level
        self.risk_level = RiskLevel(config.get('risk', {}).get('level', 'moderate'))
        
        # Position tracking
        self.positions = {}
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 0.0
        
        # Market risk tracking
        self.market_volatility = 0.0
        self.correlation_threshold = 0.7
        
    def calculate_position_size(self, symbol: str, confidence: float, 
                              current_price: float, account_value: float) -> int:
        """Calculate optimal position size based on risk management rules."""
        
        # Base position size from confidence
        base_size_pct = self.max_position_size * confidence
        
        # Adjust for risk level
        risk_multiplier = {
            RiskLevel.CONSERVATIVE: 0.5,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.AGGRESSIVE: 1.5
        }[self.risk_level]
        
        adjusted_size_pct = base_size_pct * risk_multiplier
        
        # Adjust for market volatility
        volatility_adjustment = max(0.5, 1 - self.market_volatility)
        adjusted_size_pct *= volatility_adjustment
        
        # Calculate position value
        available_capital = account_value * (1 - self.min_cash_reserve)
        position_value = available_capital * adjusted_size_pct
        
        # Calculate shares
        shares = int(position_value / current_price)
        
        # Apply minimum and maximum constraints
        min_shares = 1
        max_shares = int(available_capital * 0.1 / current_price)  # Max 10% per position
        
        shares = max(min_shares, min(shares, max_shares))
        
        return shares
    
    def check_portfolio_limits(self, account_value: float, positions: Dict) -> Dict[str, bool]:
        """Check if portfolio is within risk limits."""
        checks = {}
        
        # Check cash reserve
        total_position_value = sum(pos.market_value for pos in positions.values())
        cash_ratio = (account_value - total_position_value) / account_value
        checks['cash_reserve'] = cash_ratio >= self.min_cash_reserve
        
        # Check position count
        checks['position_count'] = len(positions) <= self.max_positions
        
        # Check daily loss limit
        checks['daily_loss'] = self.daily_pnl >= -account_value * self.max_daily_loss
        
        # Check concentration limits
        max_concentration = 0.2  # Max 20% in any single position
        for symbol, pos in positions.items():
            concentration = pos.market_value / account_value
            if concentration > max_concentration:
                checks['concentration'] = False
                break
        else:
            checks['concentration'] = True
        
        return checks
    
    def calculate_position_risk(self, symbol: str, position, hist_data: pd.DataFrame) -> PositionRisk:
        """Calculate risk metrics for a specific position."""
        
        # Calculate returns
        returns = hist_data['close'].pct_change().dropna()
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * position.market_value
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * position.market_value
        
        # Days held (simplified)
        days_held = 1  # Would need to track entry date
        
        # Unrealized PnL percentage
        unrealized_pnl_pct = position.unrealized_pnl / (position.quantity * position.average_price)
        
        return PositionRisk(
            symbol=symbol,
            current_value=position.market_value,
            unrealized_pnl=position.unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            var_95=var_95,
            max_drawdown=max_drawdown,
            days_held=days_held
        )
    
    def calculate_portfolio_risk(self, account_value: float, positions: Dict, 
                               market_data: Dict[str, pd.DataFrame]) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics."""
        
        if not positions:
            return PortfolioRisk(
                total_value=account_value,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                var_95=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                beta=1.0,
                correlation=0.0,
                sector_concentration={},
                position_concentration={}
            )
        
        # Calculate total PnL
        total_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        total_pnl_pct = total_pnl / account_value
        
        # Calculate position concentrations
        position_concentration = {}
        for symbol, pos in positions.items():
            position_concentration[symbol] = pos.market_value / account_value
        
        # Calculate sector concentration (simplified)
        sector_concentration = {'Technology': 0.5, 'Finance': 0.3, 'Other': 0.2}
        
        # Calculate portfolio VaR
        portfolio_returns = []
        for symbol, pos in positions.items():
            if symbol in market_data:
                returns = market_data[symbol]['close'].pct_change().dropna()
                weighted_returns = returns * (pos.market_value / account_value)
                portfolio_returns.append(weighted_returns)
        
        if portfolio_returns:
            portfolio_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            var_95 = np.percentile(portfolio_returns, 5) * account_value
            sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
        else:
            var_95 = 0.0
            sharpe_ratio = 0.0
        
        # Update max drawdown
        if account_value > self.peak_value:
            self.peak_value = account_value
        
        current_drawdown = (self.peak_value - account_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        return PortfolioRisk(
            total_value=account_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            var_95=var_95,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe_ratio,
            beta=1.0,  # Simplified
            correlation=0.0,  # Simplified
            sector_concentration=sector_concentration,
            position_concentration=position_concentration
        )
    
    def should_exit_position(self, symbol: str, position, current_price: float) -> Tuple[bool, str]:
        """Determine if a position should be exited based on risk rules."""
        
        # Check stop loss
        if position.average_price:
            stop_loss_price = position.average_price * (1 - self.stop_loss_pct)
            if current_price <= stop_loss_price:
                return True, f"Stop loss triggered at {current_price:.2f}"
        
        # Check take profit
        if position.average_price:
            take_profit_price = position.average_price * (1 + self.take_profit_pct)
            if current_price >= take_profit_price:
                return True, f"Take profit triggered at {current_price:.2f}"
        
        # Check unrealized loss limit
        if position.unrealized_pnl < -position.market_value * 0.1:  # 10% loss
            return True, f"Unrealized loss limit exceeded: {position.unrealized_pnl:.2f}"
        
        # Check time-based exit (simplified)
        # In practice, you'd track entry time and exit after certain days
        
        return False, ""
    
    def update_market_risk(self, market_data: Dict[str, pd.DataFrame]):
        """Update market risk metrics."""
        
        if not market_data:
            return
        
        # Calculate average market volatility
        volatilities = []
        for symbol, data in market_data.items():
            if len(data) > 20:
                returns = data['close'].pct_change().dropna()
                volatility = returns.std()
                volatilities.append(volatility)
        
        if volatilities:
            self.market_volatility = np.mean(volatilities)
    
    def get_risk_report(self, account_value: float, positions: Dict, 
                       market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate comprehensive risk report."""
        
        # Portfolio risk metrics
        portfolio_risk = self.calculate_portfolio_risk(account_value, positions, market_data)
        
        # Position risk metrics
        position_risks = {}
        for symbol, pos in positions.items():
            if symbol in market_data:
                position_risks[symbol] = self.calculate_position_risk(symbol, pos, market_data[symbol])
        
        # Portfolio limit checks
        limit_checks = self.check_portfolio_limits(account_value, positions)
        
        # Risk alerts
        alerts = []
        if portfolio_risk.total_pnl_pct < -0.05:
            alerts.append("Portfolio down more than 5%")
        if portfolio_risk.max_drawdown < -0.1:
            alerts.append("Portfolio drawdown exceeds 10%")
        if not limit_checks['cash_reserve']:
            alerts.append("Cash reserve below minimum")
        if not limit_checks['daily_loss']:
            alerts.append("Daily loss limit exceeded")
        
        return {
            'portfolio_risk': portfolio_risk,
            'position_risks': position_risks,
            'limit_checks': limit_checks,
            'alerts': alerts,
            'market_volatility': self.market_volatility,
            'risk_level': self.risk_level.value
        }
    
    def adjust_for_market_conditions(self, base_size: int, market_volatility: float) -> int:
        """Adjust position size based on market conditions."""
        
        # Reduce size in high volatility
        if market_volatility > 0.03:  # 3% daily volatility
            adjustment = 0.5
        elif market_volatility > 0.02:  # 2% daily volatility
            adjustment = 0.75
        else:
            adjustment = 1.0
        
        return int(base_size * adjustment)
    
    def calculate_correlation_risk(self, positions: Dict, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate portfolio correlation risk."""
        
        if len(positions) < 2:
            return 0.0
        
        # Calculate correlation matrix
        returns_data = {}
        for symbol in positions.keys():
            if symbol in market_data:
                returns = market_data[symbol]['close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            return 0.0
        
        # Create correlation matrix
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        # Calculate average correlation
        correlations = []
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                correlations.append(correlation_matrix.iloc[i, j])
        
        return np.mean(correlations) if correlations else 0.0 