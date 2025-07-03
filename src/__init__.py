"""
Qlib Trading System - Main Package

A professional-grade algorithmic trading system built on top of Qlib,
featuring advanced backtesting, real-time streaming, ML-based signals,
and comprehensive risk management.
"""

__version__ = "1.0.0"
__author__ = "Qlib Trading System Team"

from .trading import *
from .broker import *
from .utils import *

__all__ = [
    # Trading components
    "AutomatedTradingEngine",
    "OptimizedSignalGenerator", 
    "MLSignalGenerator",
    "BacktestingEngine",
    "RealTimeDataStream",
    "RiskManager",
    "IntegratedTradingSystem",
    # Broker components
    "AlpacaConnector",
    # Utils
    "ConfigManager"
] 