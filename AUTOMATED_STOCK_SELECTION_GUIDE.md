# 🤖 Automated Stock Selection System

## 🎯 **What This System Does**

Instead of manually picking stocks like `AAPL` and `MSFT`, this system **automatically analyzes the entire market** and selects the best stocks for trading based on multiple intelligent criteria.

### **Key Features:**
- **Market Analysis**: Analyzes 100+ stocks across all sectors
- **Multi-Criteria Scoring**: Uses 7 different factors to rank stocks
- **Real-time Data**: Uses live market data for analysis
- **Intelligent Filtering**: Removes unsuitable stocks automatically
- **Performance Optimization**: Focuses on stocks with high profit potential

## 🔍 **How Stock Selection Works**

### **1. Stock Universe (100+ Stocks)**
The system analyzes stocks from multiple categories:
- **Technology**: AAPL, MSFT, GOOGL, TSLA, NVDA, META, etc.
- **Healthcare**: JNJ, PFE, UNH, ABBV, TMO, etc.
- **Consumer**: PG, KO, WMT, HD, MCD, etc.
- **Financial**: JPM, BAC, WFC, GS, etc.
- **Energy**: XOM, CVX, COP, etc.

### **2. Selection Criteria (7 Factors)**

#### **📈 Momentum Score (25% weight)**
- **Price Momentum**: 20-day price change
- **Moving Average Trend**: MA20 vs MA50
- **Trend Strength**: Direction and magnitude

#### **📊 Volatility Score (20% weight)**
- **Optimal Volatility**: 15-35% annualized
- **Stability**: Recent vs historical volatility
- **Risk Assessment**: Not too low, not too high

#### **📈 Volume Score (15% weight)**
- **Volume Increase**: Recent vs average volume
- **Liquidity**: Minimum 10M shares daily
- **Volume Trend**: Increasing volume patterns

#### **🔧 Technical Score (20% weight)**
- **RSI**: Relative Strength Index (30-70 range)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Price position within bands

#### **💰 Fundamental Score (10% weight)**
- **P/E Ratio**: Price-to-Earnings (10-30 range)
- **Revenue Growth**: Year-over-year growth
- **Profit Margins**: Company profitability

#### **🏭 Sector Score (5% weight)**
- **Sector Rotation**: Hot sectors (Tech, Healthcare, Consumer)
- **Market Trends**: Sector performance analysis

#### **💎 Market Cap Score (5% weight)**
- **Size Preference**: Mid to large cap ($10B-$1T)
- **Liquidity**: Large enough for easy trading

### **3. Scoring Algorithm**
```
Total Score = (Momentum × 0.25) + (Volatility × 0.20) + (Volume × 0.15) + 
              (Technical × 0.20) + (Fundamental × 0.10) + (Sector × 0.05) + 
              (Market Cap × 0.05)
```

## 🚀 **How to Use Automated Stock Selection**

### **Method 1: Standalone Stock Selection**
```bash
# Select top 10 stocks
python3 select_stocks.py

# Select top 5 stocks
python3 select_stocks.py --max-stocks 5

# Save selected stocks to file
python3 select_stocks.py --save --output my_stocks.txt
```

### **Method 2: Integrated with Trading System**
```bash
# Auto-select stocks for paper trading
python3 start_trading.py --paper --auto-select --max-stocks 10

# Auto-select stocks for live trading
python3 start_trading.py --live --auto-select --max-stocks 5
```

### **Method 3: Use Pre-selected Stocks**
```bash
# Use stocks from file
python3 start_trading.py --paper --symbols $(cat selected_stocks.txt)

# Use specific stocks
python3 start_trading.py --paper --symbols AAPL MSFT GOOGL TSLA NVDA
```

## 📊 **Example Output**

```
🔍 Automated Stock Selection System
==================================================
Analyzing 100+ stocks...
Criteria: Momentum, Volatility, Volume, Technical Indicators, Fundamentals

📊 Top 10 Stocks Selected:
================================================================================
 1. NVDA    | Total Score: 0.847
     Momentum: 0.85 | Volatility: 0.72 | Volume: 0.78
     Technical: 0.82 | Fundamental: 0.65
     Top Reasons: Strong price momentum: 12.3%, High volume: 2.1x average, Positive MACD: 0.245

 2. TSLA    | Total Score: 0.823
     Momentum: 0.78 | Volatility: 0.68 | Volume: 0.85
     Technical: 0.75 | Fundamental: 0.45
     Top Reasons: Above 50-day moving average, Strong MA trend: 8.7%, High liquidity

 3. AAPL    | Total Score: 0.798
     Momentum: 0.72 | Volatility: 0.65 | Volume: 0.82
     Technical: 0.78 | Fundamental: 0.78
     Top Reasons: RSI neutral: 52.3, Good profit margins: 25.3%, Large-cap: $2.8T
```

## ⚙️ **Configuration Options**

### **Stock Selection Config** (`config/stock_selection_config.yaml`)
```yaml
selection:
  max_stocks: 10
  min_market_cap: 10000000000  # $10B minimum
  max_market_cap: 1000000000000  # $1T maximum
  min_volume: 1000000  # 1M shares minimum
  min_price: 10.0
  max_price: 500.0

scoring:
  momentum_weight: 0.25
  volatility_weight: 0.20
  volume_weight: 0.15
  technical_weight: 0.20
  fundamental_weight: 0.10
  sector_weight: 0.05
  market_cap_weight: 0.05
```

## 🎯 **Trading Strategies**

### **Conservative Strategy**
```bash
# Select 5-7 stocks with high fundamentals
python3 select_stocks.py --max-stocks 5
python3 start_trading.py --paper --symbols $(cat selected_stocks.txt) --interval 10
```

### **Aggressive Strategy**
```bash
# Select 10-15 stocks with high momentum
python3 select_stocks.py --max-stocks 15
python3 start_trading.py --paper --symbols $(cat selected_stocks.txt) --interval 5
```

### **Sector Rotation Strategy**
```bash
# Focus on specific sectors
# Modify config to weight sector_score higher
python3 select_stocks.py --max-stocks 8
```

## 🔄 **When to Re-select Stocks**

### **Automatic Re-selection**
- **Daily**: Before market open
- **Weekly**: Every Monday
- **Monthly**: First trading day of month

### **Manual Re-selection Triggers**
- **Market Crash**: After major market events
- **Sector Rotation**: When hot sectors change
- **Performance Issues**: If current stocks underperform

### **Re-selection Commands**
```bash
# Daily re-selection
python3 select_stocks.py --save --output daily_stocks.txt

# Weekly re-selection
python3 select_stocks.py --max-stocks 15 --save --output weekly_stocks.txt

# Emergency re-selection
python3 select_stocks.py --max-stocks 5 --save --output emergency_stocks.txt
```

## 📈 **Performance Tracking**

### **Track Selection Performance**
```bash
# Save selection with timestamp
python3 select_stocks.py --save --output stocks_$(date +%Y%m%d).txt

# Compare performance
python3 start_trading.py --performance
```

### **Selection Metrics**
- **Win Rate**: How often selected stocks perform well
- **Average Return**: Average performance of selected stocks
- **Sector Distribution**: Balance across sectors
- **Market Cap Distribution**: Size distribution

## 🛡️ **Risk Management**

### **Diversification**
- **Sector Limits**: Max 30% in any sector
- **Market Cap Limits**: Mix of mid and large cap
- **Position Limits**: Max 10% per stock

### **Quality Filters**
- **Minimum Volume**: Ensures liquidity
- **Price Range**: Avoid penny stocks and expensive stocks
- **Market Cap**: Focus on established companies

### **Dynamic Adjustments**
- **Volatility Limits**: Avoid extremely volatile stocks
- **Momentum Filters**: Focus on trending stocks
- **Technical Filters**: Avoid overbought/oversold conditions

## 🚨 **Important Notes**

### **Market Conditions**
- **Bull Market**: Focus on momentum and growth
- **Bear Market**: Focus on fundamentals and stability
- **Sideways Market**: Focus on technical indicators

### **Trading Hours**
- **Pre-market**: Use previous day's selection
- **Market Hours**: Real-time selection available
- **After Hours**: Use current day's selection

### **Data Quality**
- **Real-time Data**: Uses Yahoo Finance API
- **Fallback Options**: Default stocks if data unavailable
- **Error Handling**: Graceful degradation

## 💡 **Pro Tips**

### **1. Start Conservative**
```bash
# Begin with 5 stocks, paper trading
python3 select_stocks.py --max-stocks 5
python3 start_trading.py --paper --symbols $(cat selected_stocks.txt)
```

### **2. Monitor Performance**
```bash
# Check performance daily
python3 start_trading.py --performance
```

### **3. Adjust Based on Results**
- If win rate < 60%: Reduce number of stocks
- If volatility too high: Increase volatility weight
- If returns low: Increase momentum weight

### **4. Combine with Manual Selection**
```bash
# Use auto-selected stocks + your favorites
python3 select_stocks.py --max-stocks 7
# Add your picks: AAPL MSFT
# Final list: auto_selected + AAPL MSFT
```

## 🎯 **Next Steps**

1. **Test the System**: Run stock selection to see results
2. **Paper Trading**: Test with selected stocks
3. **Monitor Performance**: Track how selections perform
4. **Adjust Parameters**: Fine-tune based on results
5. **Scale Up**: Move to live trading when ready

---

**🤖 Ready to let AI pick your stocks? Start with automated selection for better results!** 