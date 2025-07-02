# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Production configuration for live trading.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class BrokerConfig:
    """Configuration for broker connection."""
    
    broker_type: str  # "interactive_brokers", "alpaca", "td_ameritrade"
    host: str
    port: int
    client_id: int
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    paper_trading: bool = True  # Start with paper trading for safety
    
    # Interactive Brokers specific
    ib_account: Optional[str] = None
    
    # Alpaca specific
    alpaca_base_url: Optional[str] = None
    
    # Connection settings
    timeout: int = 30
    max_retries: int = 3


@dataclass
class RiskConfig:
    """Risk management configuration."""
    
    max_position_size: float = 100000  # USD
    max_daily_loss: float = 5000       # USD
    max_sector_exposure: float = 0.3   # 30%
    max_single_stock_exposure: float = 0.1  # 10%
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.05  # 5%
    position_sizing_method: str = "kelly"  # "kelly", "equal", "volatility"


@dataclass
class DataConfig:
    """Real-time data configuration."""
    
    data_provider: str = "polygon"  # "polygon", "iex", "alpha_vantage"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    update_frequency: str = "1min"  # "1min", "5min", "1hour"
    cache_duration: int = 300  # seconds


@dataclass
class TradingConfig:
    """Trading execution configuration."""
    
    default_order_type: str = "market"  # "market", "limit", "stop"
    max_slippage: float = 0.001  # 0.1%
    min_order_size: float = 100  # USD
    max_order_size: float = 10000  # USD
    execution_delay: float = 0.1  # seconds
    retry_failed_orders: bool = True
    max_order_retries: int = 3
    symbols: Optional[List[str]] = None  # List of symbols to trade
    enable_trading: bool = False  # Whether to enable actual trading


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    
    alert_email: Optional[str] = None
    slack_webhook: Optional[str] = None
    log_level: str = "INFO"
    performance_metrics: bool = True
    risk_alerts: bool = True
    trade_logging: bool = True


class ProductionConfig:
    """Main production configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize production configuration.
        
        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file
        """
        self.broker = BrokerConfig(
            broker_type="interactive_brokers",
            host="127.0.0.1",
            port=7497,  # 7496 for live, 7497 for paper
            client_id=1,
            paper_trading=True
        )
        
        self.risk = RiskConfig()
        self.data = DataConfig()
        self.trading = TradingConfig()
        self.monitoring = MonitoringConfig()
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self.load_from_env()
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        self._update_from_dict(config_data)
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        # Broker settings
        self.broker.broker_type = os.getenv("BROKER_TYPE", "interactive_brokers")
        self.broker.host = os.getenv("BROKER_HOST", "127.0.0.1")
        self.broker.port = int(os.getenv("BROKER_PORT", "7497"))
        self.broker.client_id = int(os.getenv("BROKER_CLIENT_ID", "1"))
        self.broker.api_key = os.getenv("BROKER_API_KEY")
        self.broker.api_secret = os.getenv("BROKER_API_SECRET")
        self.broker.paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        
        # Risk settings
        self.risk.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "100000"))
        self.risk.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "5000"))
        self.risk.max_sector_exposure = float(os.getenv("MAX_SECTOR_EXPOSURE", "0.3"))
        
        # Data settings
        self.data.data_provider = os.getenv("DATA_PROVIDER", "polygon")
        self.data.api_key = os.getenv("DATA_API_KEY")
        
        # Monitoring settings
        self.monitoring.alert_email = os.getenv("ALERT_EMAIL")
        self.monitoring.slack_webhook = os.getenv("SLACK_WEBHOOK")
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        if "broker" in config_data:
            for key, value in config_data["broker"].items():
                if hasattr(self.broker, key):
                    setattr(self.broker, key, value)
        
        if "risk" in config_data:
            for key, value in config_data["risk"].items():
                if hasattr(self.risk, key):
                    setattr(self.risk, key, value)
        
        if "data" in config_data:
            for key, value in config_data["data"].items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        
        if "trading" in config_data:
            for key, value in config_data["trading"].items():
                if hasattr(self.trading, key):
                    setattr(self.trading, key, value)
        
        if "monitoring" in config_data:
            for key, value in config_data["monitoring"].items():
                if hasattr(self.monitoring, key):
                    setattr(self.monitoring, key, value)
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file."""
        config_data = {
            "broker": self.broker.__dict__,
            "risk": self.risk.__dict__,
            "data": self.data.__dict__,
            "trading": self.trading.__dict__,
            "monitoring": self.monitoring.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Validate broker settings
        if not self.broker.host:
            errors.append("Broker host is required")
        
        if self.broker.port <= 0:
            errors.append("Broker port must be positive")
        
        # Validate risk settings
        if self.risk.max_position_size <= 0:
            errors.append("Max position size must be positive")
        
        if self.risk.max_daily_loss <= 0:
            errors.append("Max daily loss must be positive")
        
        if not 0 <= self.risk.max_sector_exposure <= 1:
            errors.append("Max sector exposure must be between 0 and 1")
        
        # Validate data settings
        if not self.data.api_key:
            errors.append("Data API key is required for live data")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True 