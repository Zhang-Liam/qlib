"""
Trading Module - Core Trading Components

This module contains all the core trading functionality including:
- Trading Engine
- Signal Generation (Technical & ML-based)
- Backtesting Engine
- Real-time Data Streaming
- Risk Management
- Integrated Trading System
"""

from .engine import AutomatedTradingEngine
from .signals import OptimizedSignalGenerator
from .ml_signals import MLSignalGenerator
from .backtesting import BacktestingEngine
from .realtime import RealTimeDataStream
from .risk_manager import RiskManager
from .integrated import IntegratedTradingSystem

__all__ = [
    "AutomatedTradingEngine",
    "OptimizedSignalGenerator",
    "MLSignalGenerator", 
    "BacktestingEngine",
    "RealTimeDataStream",
    "RiskManager",
    "IntegratedTradingSystem"
] 