# Alpaca Integration Setup Guide

This guide will help you set up Alpaca integration for your Qlib trading system.

## Prerequisites

1. **Python 3.8+** installed
2. **Alpaca account** (free paper trading available)
3. **Alpaca API credentials**

## Step 1: Get Alpaca API Credentials

1. **Sign up for Alpaca** at [alpaca.markets](https://alpaca.markets)
2. **Navigate to Paper Trading** (recommended for testing)
3. **Get your API keys**:
   - Go to **Paper Trading** section
   - Click **"View"** next to your API keys
   - Copy your **API Key ID** and **Secret Key**

## Step 2: Install Dependencies

```bash
# Install Alpaca Python SDK
pip3 install alpaca-py

# Install other required packages
pip3 install pandas numpy
```

## Step 3: Set Environment Variables

Set your Alpaca API credentials as environment variables:

```bash
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
```

**For permanent setup**, add these to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.):

```bash
echo 'export ALPACA_API_KEY="your_api_key_here"' >> ~/.zshrc
echo 'export ALPACA_SECRET_KEY="your_secret_key_here"' >> ~/.zshrc
source ~/.zshrc
```

## Step 4: Test the Integration

Run the Alpaca integration test:

```bash
python3 test_alpaca_integration.py
```

This test will:
- ✅ Connect to Alpaca
- ✅ Retrieve account information
- ✅ Get real-time market prices
- ✅ Fetch historical data
- ✅ Test order placement (paper trading)
- ✅ Test order cancellation

## Step 5: Configure Your Trading System

### Option A: Use Configuration File

1. **Copy the configuration template**:
   ```bash
   cp config/alpaca_config.yaml config/my_alpaca_config.yaml
   ```

2. **Edit the configuration**:
   ```yaml
   broker:
     broker_type: "alpaca"
     paper_trading: true  # Set to false for live trading
   
   trading:
     symbols: ["AAPL", "GOOGL", "MSFT", "SPY"]  # Your symbols
     enable_trading: false  # Set to true for live trading
   ```

### Option B: Use Environment Variables

Set these environment variables:

```bash
export BROKER_TYPE="alpaca"
export PAPER_TRADING="true"
export TRADING_SYMBOLS="AAPL,GOOGL,MSFT,SPY"
export ENABLE_TRADING="false"
```

## Step 6: Run Your Trading System

### For Development/Testing:

```bash
# Use mock components for testing
export BROKER_TYPE="mock"
export DATA_PROVIDER="mock"
python3 simple_mock_test.py
```

### For Alpaca Paper Trading:

```bash
# Use Alpaca paper trading
export BROKER_TYPE="alpaca"
export DATA_PROVIDER="alpaca"
python3 test_alpaca_integration.py
```

### For Docker:

```bash
# Build and run with Alpaca configuration
docker-compose up -d
```

## Step 7: Monitor and Debug

### Check Logs

```bash
# View application logs
docker logs qlib-production-trading

# View test logs
tail -f test_alpaca_integration.log
```

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| **"API key not found"** | Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` environment variables |
| **"Account not active"** | Check your Alpaca account status in the dashboard |
| **"Market data not available"** | Ensure you're using paper trading or have market data subscription |
| **"Order rejected"** | Check account buying power and order parameters |

## Step 8: Production Setup

### For Live Trading:

1. **Switch to live trading**:
   ```yaml
   broker:
     paper_trading: false
   
   trading:
     enable_trading: true
   ```

2. **Use live API endpoints**:
   ```yaml
   broker:
     alpaca_base_url: "https://api.alpaca.markets"
   ```

3. **Set up proper risk management**:
   ```yaml
   risk:
     max_position_size: 50000  # Adjust based on your account
     max_daily_loss: 1000      # Conservative limits
   ```

### Security Best Practices:

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **Start with paper trading** to test strategies
4. **Set conservative risk limits** initially
5. **Monitor your system** regularly

## API Rate Limits

Alpaca has the following rate limits:
- **Paper Trading**: 200 requests per minute
- **Live Trading**: 200 requests per minute
- **Market Data**: Varies by subscription

## Support

- **Alpaca Documentation**: [docs.alpaca.markets](https://docs.alpaca.markets)
- **Alpaca Community**: [forum.alpaca.markets](https://forum.alpaca.markets)
- **Qlib Issues**: [GitHub Issues](https://github.com/microsoft/qlib/issues)

## Next Steps

1. **Test with paper trading** first
2. **Implement your trading strategy**
3. **Set up monitoring and alerts**
4. **Gradually increase position sizes**
5. **Monitor performance and adjust**

---

**⚠️ Important**: Always test thoroughly with paper trading before using real money. Start with small amounts and gradually increase as you gain confidence in your system. 