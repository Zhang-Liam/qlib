# Qlib Production Trading System

This module provides production-ready components for live US market trading using Qlib. It includes broker integration, risk management, live data feeds, and a complete workflow system.

## Components

### 1. Configuration (`config.py`)
- **ProductionConfig**: Centralized configuration management
- Supports YAML configuration files
- Environment variable substitution
- Validation and error handling

### 2. Broker Integration (`broker.py`)
- **BrokerInterface**: Abstract broker interface
- **IBKRBroker**: Interactive Brokers integration using ib_insync
- **AlpacaBroker**: Alpaca Markets integration
- Order placement, account management, position tracking

### 3. Risk Management (`risk_manager.py`)
- **RiskManager**: Comprehensive risk controls
- Position size limits
- Daily loss limits
- Exposure limits (sector, single stock)
- Leverage limits
- Volatility limits
- Trading frequency limits

### 4. Live Data (`live_data.py`)
- **LiveDataProvider**: Abstract data provider interface
- **IBKRLiveDataProvider**: Interactive Brokers real-time data
- **AlpacaLiveDataProvider**: Alpaca Markets data
- **MockLiveDataProvider**: Mock data for testing

### 5. Workflow (`workflow.py`)
- **ProductionWorkflow**: Main orchestration class
- Connects all components
- Runs trading cycles
- Handles continuous operation
- Graceful shutdown

## Quick Start

### 1. Install Dependencies

```bash
# For Interactive Brokers
pip install ib_insync

# For Alpaca Markets
pip install alpaca-trade-api

# For configuration
pip install pyyaml
```

### 2. Create Configuration

Create a `production_config.yaml` file:

```yaml
broker:
  provider: "mock"  # Use "mock" for testing
  mock:
    paper_trading: true

risk:
  max_position_size: 0.05
  max_daily_loss: 0.02
  max_total_positions: 0.80

data:
  provider: "mock"
  mock:
    volatility: 0.01

trading:
  enable_trading: false  # Set to true for live trading
  symbols: ["AAPL", "GOOGL", "MSFT"]
```

### 3. Run the Workflow

```python
from qlib.production.workflow import ProductionWorkflow

# Create workflow
workflow = ProductionWorkflow("production_config.yaml")

# Initialize components
if workflow.initialize_components():
    # Run single cycle
    workflow.run_single_cycle()
    
    # Or run continuously
    # workflow.run_continuous(cycle_interval=60)
    
    # Shutdown
    workflow.shutdown()
```

### 4. Test with Mock Data

```bash
cd examples
python test_production_workflow.py
```

## Configuration Details

### Broker Configuration

#### Interactive Brokers
```yaml
broker:
  provider: "ibkr"
  ibkr:
    host: "127.0.0.1"
    port: 7497  # 7497 for TWS, 4001 for Gateway
    client_id: 1
    paper_trading: true
```

#### Alpaca Markets
```yaml
broker:
  provider: "alpaca"
  alpaca:
    api_key: "${ALPACA_API_KEY}"
    secret_key: "${ALPACA_SECRET_KEY}"
    base_url: "https://paper-api.alpaca.markets"
    paper_trading: true
```

### Risk Management Configuration

```yaml
risk:
  # Position limits
  max_position_size: 0.05      # 5% per position
  max_total_positions: 0.80    # 80% total positions
  
  # Loss limits
  max_daily_loss: 0.02         # 2% daily loss
  max_total_loss: 0.10         # 10% total loss
  
  # Exposure limits
  max_sector_exposure: 0.30    # 30% sector exposure
  max_single_stock_exposure: 0.10  # 10% single stock
  
  # Trading limits
  max_trades_per_day: 50
  min_trade_interval: 60       # seconds
```

### Live Data Configuration

```yaml
data:
  provider: "ibkr"  # or "alpaca", "mock"
  ibkr:
    host: "127.0.0.1"
    port: 7497
    client_id: 2
    timeout: 5
```

## Production Deployment

### 1. Environment Setup

```bash
# Set environment variables
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
export DB_USERNAME="your_db_user"
export DB_PASSWORD="your_db_password"
```

### 2. Directory Structure

```
qlib/
├── production/
│   ├── config.py
│   ├── broker.py
│   ├── risk_manager.py
│   ├── live_data.py
│   ├── workflow.py
│   └── README.md
├── logs/
│   └── production.log
├── data/
│   └── trading.db
└── models/
    └── latest_model.pkl
```

### 3. Systemd Service (Linux)

Create `/etc/systemd/system/qlib-trading.service`:

```ini
[Unit]
Description=Qlib Trading System
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/path/to/qlib
Environment=PYTHONPATH=/path/to/qlib
ExecStart=/usr/bin/python3 -m qlib.production.workflow --config production_config.yaml --continuous
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 4. Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "qlib.production.workflow", "--config", "production_config.yaml", "--continuous"]
```

## Monitoring and Alerts

### Performance Metrics
- Returns and Sharpe ratio
- Maximum drawdown
- Volatility tracking
- Trade count and win rate

### Alert Conditions
- Daily loss threshold exceeded
- Maximum drawdown reached
- Connection failures
- Data feed issues

### Logging
- Structured logging with timestamps
- File rotation (10MB files, 5 backups)
- Different log levels (DEBUG, INFO, WARNING, ERROR)

## Safety Features

### Emergency Stops
- Automatic position closure on risk limit breaches
- Trading disable on connection loss
- Manual approval required for restart

### Risk Controls
- Real-time position monitoring
- Pre-trade risk checks
- Post-trade validation
- Circuit breakers

### Data Validation
- Price sanity checks
- Volume validation
- Timestamp verification
- Source verification

## Development and Testing

### Mock Mode
Use mock providers for development and testing:

```yaml
broker:
  provider: "mock"

data:
  provider: "mock"
  mock:
    volatility: 0.01
```

### Unit Tests
```bash
python -m pytest tests/production/
```

### Integration Tests
```bash
python examples/test_production_workflow.py
```

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check broker credentials
   - Verify network connectivity
   - Ensure TWS/Gateway is running (for IBKR)

2. **Risk Manager Rejections**
   - Review risk limits in configuration
   - Check account balance and positions
   - Verify order sizes

3. **Data Feed Issues**
   - Confirm market hours
   - Check data provider credentials
   - Verify symbol availability

### Log Analysis
```bash
# View recent logs
tail -f logs/production.log

# Search for errors
grep "ERROR" logs/production.log

# Monitor specific component
grep "broker" logs/production.log
```

## Contributing

When adding new components:

1. Follow the existing interface patterns
2. Add comprehensive error handling
3. Include logging statements
4. Write unit tests
5. Update documentation

## License

This module is part of the Qlib project and follows the same license terms. 