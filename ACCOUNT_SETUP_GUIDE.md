# 📊 Account Setup Guide for Live Trading

## 🎯 **What You Need for Live Trading**

### **1. Alpaca Account Setup**

You need **TWO separate accounts** on Alpaca:

#### **Paper Trading Account (Free)**
- **Purpose**: Test strategies safely
- **URL**: https://paper-api.alpaca.markets
- **Cost**: Free
- **Risk**: No real money involved

#### **Live Trading Account (Free)**
- **Purpose**: Real money trading
- **URL**: https://api.alpaca.markets
- **Cost**: Free (commission-free trading)
- **Risk**: Real money involved

## 🚀 **Step-by-Step Setup**

### **Step 1: Create Alpaca Accounts**

1. **Go to Alpaca**: https://alpaca.markets/
2. **Sign up for Paper Trading**:
   - Click "Get Started"
   - Choose "Paper Trading"
   - Complete registration
   - Get your **Paper Trading API Keys**

3. **Sign up for Live Trading**:
   - Go to your dashboard
   - Apply for live trading account
   - Complete identity verification
   - Get your **Live Trading API Keys**

### **Step 2: Set Up Environment Variables**

Create a `.env` file in your project root:

```bash
# Paper Trading (for testing)
ALPACA_API_KEY=your_paper_trading_api_key_here
ALPACA_SECRET_KEY=your_paper_trading_secret_key_here

# Live Trading (for real money - BE CAREFUL!)
ALPACA_LIVE_API_KEY=your_live_trading_api_key_here
ALPACA_LIVE_SECRET_KEY=your_live_trading_secret_key_here
```

### **Step 3: Test Paper Trading First**

```bash
# Test with paper trading (safe)
python3 start_trading.py --paper --symbols AAPL MSFT

# Check status
python3 start_trading.py --status
```

### **Step 4: Move to Live Trading (When Ready)**

```bash
# Use live trading configuration
python3 start_trading.py --live --config config/live_trading_config.yaml --symbols AAPL
```

## 🔧 **Configuration Files**

### **Paper Trading Config** (`config/alpaca_config.yaml`)
- Uses paper trading API
- `paper_trading: true`
- Conservative settings for testing

### **Live Trading Config** (`config/live_trading_config.yaml`)
- Uses live trading API
- `paper_trading: false`
- Conservative safety settings for real money

## ⚠️ **Important Safety Notes**

### **Before Live Trading:**
1. **Test thoroughly** with paper trading first
2. **Start small** with minimal amounts
3. **Monitor closely** during initial live trading
4. **Know your emergency stop** procedures
5. **Understand the risks** involved

### **Live Trading Safety Settings:**
- **Max Position Size**: 5% of account per position
- **Daily Loss Limit**: 2% maximum daily loss
- **Cash Reserve**: 30% minimum cash buffer
- **Max Positions**: 5 concurrent positions
- **Stop Loss**: 3% automatic stop loss

## 🚨 **Why Your System Showed "Live Trading"**

From your log, I can see the issue:

```
Trading Mode: LIVE_TRADING
Market Open: 🔴 NO
```

**The Problem:**
- Your system switched to "live trading mode" in the software
- But it's still using the **paper trading API endpoint**
- The market is closed, causing connection errors
- No real money is actually at risk

**The Fix:**
- Use the correct configuration file for live trading
- Set up separate API keys for live trading
- Only use live trading when you're ready and the market is open

## 📋 **Account Verification Requirements**

### **For Live Trading Account:**
- **Identity Verification**: SSN, driver's license
- **Address Verification**: Utility bill or bank statement
- **Employment Information**: Job details
- **Financial Information**: Income, net worth
- **Trading Experience**: Previous trading history
- **Risk Tolerance**: Investment objectives

### **Funding Requirements:**
- **Minimum Deposit**: $0 (Alpaca has no minimum)
- **Recommended Start**: $1,000-$5,000 for testing
- **Funding Method**: Bank transfer (ACH)

## 🎯 **Recommended Approach**

### **Phase 1: Paper Trading (1-2 weeks)**
```bash
python3 start_trading.py --paper --symbols AAPL MSFT GOOGL
```
- Test your strategy thoroughly
- Monitor performance metrics
- Adjust parameters as needed

### **Phase 2: Small Live Trading (1-2 weeks)**
```bash
python3 start_trading.py --live --config config/live_trading_config.yaml --symbols AAPL
```
- Start with one symbol only
- Use small position sizes
- Monitor closely

### **Phase 3: Scale Up (ongoing)**
- Add more symbols gradually
- Increase position sizes based on performance
- Continue monitoring and adjusting

## 🔍 **Troubleshooting**

### **Connection Errors:**
```bash
# Check if market is open
python3 start_trading.py --status

# Check API keys
echo $ALPACA_API_KEY
echo $ALPACA_LIVE_API_KEY
```

### **Configuration Issues:**
```bash
# Verify config files exist
ls -la config/
cat config/live_trading_config.yaml
```

### **Account Issues:**
- Check Alpaca dashboard for account status
- Verify API keys are correct
- Ensure account is funded (for live trading)

## 📞 **Support Resources**

- **Alpaca Support**: https://alpaca.markets/support/
- **API Documentation**: https://alpaca.markets/docs/
- **Trading Hours**: 9:30 AM - 4:00 PM ET (Mon-Fri)
- **Emergency**: Use emergency stop feature immediately

---

**🚀 Ready to start? Begin with paper trading to test your strategy safely!** 