#!/usr/bin/env python3
"""
Automated Stock Selection System
Intelligently picks the best stocks for trading based on multiple criteria.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path
import time

class SelectionCriteria(Enum):
    """Stock selection criteria."""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SECTOR_ROTATION = "sector_rotation"
    MARKET_CAP = "market_cap"

@dataclass
class StockScore:
    """Stock scoring result."""
    symbol: str
    total_score: float
    momentum_score: float
    volatility_score: float
    volume_score: float
    technical_score: float
    fundamental_score: float
    sector_score: float
    market_cap_score: float
    reasons: List[str]

class AutomatedStockSelector:
    """Automated stock selection system."""
    
    def __init__(self, config_path: str = "config/stock_selection_config.yaml"):
        """Initialize stock selector."""
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Stock universe - simplified to avoid API calls during init
        self.stock_universe = self._load_stock_universe()
        
        # Cache for performance
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'selection': {
                'max_stocks': 10,
                'min_market_cap': 1000000000,   # $1B (reduced from $10B)
                'max_market_cap': 1000000000000,  # $1T
                'min_volume': 100000,           # 100K shares (reduced from 1M)
                'min_price': 5.0,               # $5 (reduced from $10)
                'max_price': 1000.0,            # $1000 (increased from $500)
                'exclude_sectors': ['REIT', 'Financial Services'],
                'include_sectors': ['Technology', 'Healthcare', 'Consumer Cyclical']
            },
            'scoring': {
                'momentum_weight': 0.25,
                'volatility_weight': 0.20,
                'volume_weight': 0.15,
                'technical_weight': 0.20,
                'fundamental_weight': 0.10,
                'sector_weight': 0.05,
                'market_cap_weight': 0.05
            },
            'filters': {
                'min_rsi': 30,
                'max_rsi': 70,
                'min_macd_signal': 0.1,
                'min_volume_ratio': 1.2,
                'max_volatility': 0.05
            }
        }
    
    def _load_stock_universe(self) -> List[str]:
        """Load stock universe from various sources."""
        # Popular tech and growth stocks
        tech_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO',
            'TXN', 'MU', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MRVL', 'WDC',
            'STX', 'HPQ', 'DELL', 'CSCO', 'IBM', 'V', 'MA', 'PYPL', 'SQ'
        ]
        
        # Healthcare stocks
        healthcare_stocks = [
            'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'DHR', 'LLY',
            'BMY', 'AMGN', 'GILD', 'CVS', 'CI', 'ANTM', 'HUM', 'CNC'
        ]
        
        # Consumer stocks
        consumer_stocks = [
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'SBUX', 'NKE',
            'DIS', 'CMCSA', 'VZ', 'T', 'COST', 'TGT', 'LOW', 'TJX'
        ]
        
        # Financial stocks
        financial_stocks = [
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
            'AXP', 'BLK', 'SCHW', 'COF', 'AIG', 'MET', 'PRU', 'ALL'
        ]
        
        # Energy and industrial stocks
        energy_stocks = [
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'PSX',
            'VLO', 'MPC', 'OXY', 'PXD', 'DVN', 'HES', 'APA', 'FANG'
        ]
        
        # Combine all stocks
        all_stocks = tech_stocks + healthcare_stocks + consumer_stocks + financial_stocks + energy_stocks
        
        # Remove duplicates
        unique_stocks = list(set(all_stocks))
        
        self.logger.info(f"Loaded {len(unique_stocks)} stocks for selection")
        return unique_stocks
    
    async def get_market_data(self, symbols: List[str], period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """Get market data for multiple symbols asynchronously with rate limiting."""
        data = {}
        
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                task = self._fetch_stock_data_with_semaphore(session, symbol, period, semaphore)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    data[symbols[i]] = result
                elif isinstance(result, Exception):
                    self.logger.warning(f"Failed to fetch data for {symbols[i]}: {result}")
        
        return data
    
    async def _fetch_stock_data_with_semaphore(self, session: aiohttp.ClientSession, symbol: str, period: str, semaphore: asyncio.Semaphore) -> pd.DataFrame:
        """Fetch stock data with semaphore for rate limiting."""
        async with semaphore:
            return await self._fetch_stock_data(session, symbol, period)
    
    async def _fetch_stock_data(self, session: aiohttp.ClientSession, symbol: str, period: str) -> pd.DataFrame:
        """Fetch stock data for a single symbol."""
        try:
            # Add delay to avoid rate limiting
            await asyncio.sleep(0.1)
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={period}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                        result = data['chart']['result'][0]
                        timestamps = result.get('timestamp', [])
                        quotes = result.get('indicators', {}).get('quote', [{}])[0]
                        
                        if timestamps and quotes.get('close'):
                            df = pd.DataFrame({
                                'timestamp': pd.to_datetime(timestamps, unit='s'),
                                'open': quotes.get('open', [None] * len(timestamps)),
                                'high': quotes.get('high', [None] * len(timestamps)),
                                'low': quotes.get('low', [None] * len(timestamps)),
                                'close': quotes.get('close', [None] * len(timestamps)),
                                'volume': quotes.get('volume', [None] * len(timestamps))
                            })
                            df.set_index('timestamp', inplace=True)
                            df = df.dropna()  # Remove rows with missing data
                            
                            if len(df) > 5:  # Ensure we have enough data
                                return df
                else:
                    self.logger.warning(f"HTTP {response.status} for {symbol}")
                    
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching data for {symbol}")
        except Exception as e:
            self.logger.warning(f"Failed to fetch data for {symbol}: {e}")
        
        return pd.DataFrame()  # Return empty DataFrame on failure
    
    def calculate_momentum_score(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate momentum score."""
        if data.empty:
            return 0.0, []
        
        reasons = []
        score = 0.0
        
        # Price momentum (20-day return)
        if len(data) >= 20:
            price_momentum = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) * 100
            if price_momentum > 5:
                score += 0.4
                reasons.append(f"Strong price momentum: {price_momentum:.1f}%")
            elif price_momentum > 0:
                score += 0.2
                reasons.append(f"Positive price momentum: {price_momentum:.1f}%")
        
        # Moving average momentum
        if len(data) >= 50:
            ma20 = data['close'].rolling(20).mean()
            ma50 = data['close'].rolling(50).mean()
            
            # Handle NaN values
            if not pd.isna(ma20.iloc[-1]) and not pd.isna(ma50.iloc[-1]):
                if ma20.iloc[-1] > ma50.iloc[-1]:
                    score += 0.3
                    reasons.append("Above 50-day moving average")
                
                if len(ma20) >= 10 and not pd.isna(ma20.iloc[-10]):
                    ma_trend = (ma20.iloc[-1] / ma20.iloc[-10] - 1) * 100
                    if ma_trend > 2:
                        score += 0.3
                        reasons.append(f"Strong MA trend: {ma_trend:.1f}%")
        
        return min(score, 1.0), reasons
    
    def calculate_volatility_score(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate volatility score."""
        if data.empty:
            return 0.0, []
        
        reasons = []
        score = 0.0
        
        # Calculate daily returns
        returns = data['close'].pct_change().dropna()
        
        if len(returns) >= 20:
            # Volatility (standard deviation of returns)
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Optimal volatility range (not too low, not too high)
            if 0.15 <= volatility <= 0.35:
                score += 0.5
                reasons.append(f"Optimal volatility: {volatility:.2f}")
            elif 0.10 <= volatility <= 0.40:
                score += 0.3
                reasons.append(f"Acceptable volatility: {volatility:.2f}")
            
            # Recent volatility vs historical
            recent_vol = returns.tail(10).std() * np.sqrt(252)
            if recent_vol <= volatility * 1.2:  # Not increasing too much
                score += 0.5
                reasons.append("Stable recent volatility")
        
        return min(score, 1.0), reasons
    
    def calculate_volume_score(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate volume score."""
        if data.empty:
            return 0.0, []
        
        reasons = []
        score = 0.0
        
        if len(data) >= 20:
            # Average volume
            avg_volume = data['volume'].mean()
            recent_volume = data['volume'].tail(5).mean()
            
            # Volume increase
            volume_ratio = recent_volume / avg_volume
            if volume_ratio > 1.5:
                score += 0.4
                reasons.append(f"High volume: {volume_ratio:.1f}x average")
            elif volume_ratio > 1.2:
                score += 0.2
                reasons.append(f"Above average volume: {volume_ratio:.1f}x")
            
            # Volume trend
            volume_trend = (recent_volume / data['volume'].tail(20).mean() - 1) * 100
            if volume_trend > 10:
                score += 0.3
                reasons.append(f"Increasing volume trend: {volume_trend:.1f}%")
            
            # Absolute volume
            if avg_volume > 10000000:  # 10M shares
                score += 0.3
                reasons.append("High liquidity")
        
        return min(score, 1.0), reasons
    
    def calculate_technical_score(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate technical indicators score."""
        if data.empty:
            return 0.0, []
        
        reasons = []
        score = 0.0
        
        if len(data) >= 14:
            # RSI
            rsi = self._calculate_rsi(data['close'])
            if 30 <= rsi <= 70:
                score += 0.3
                reasons.append(f"RSI in range: {rsi:.1f}")
            elif 40 <= rsi <= 60:
                score += 0.5
                reasons.append(f"RSI neutral: {rsi:.1f}")
            
            # MACD
            macd_signal = self._calculate_macd_signal(data['close'])
            if macd_signal > 0.1:
                score += 0.4
                reasons.append(f"Positive MACD: {macd_signal:.3f}")
            
            # Bollinger Bands
            bb_position = self._calculate_bb_position(data['close'])
            if 0.2 <= bb_position <= 0.8:
                score += 0.3
                reasons.append(f"BB position: {bb_position:.2f}")
        
        return min(score, 1.0), reasons
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if prices.empty or len(prices) < period:
            return 50.0  # Neutral RSI if not enough data
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle NaN values
        if pd.isna(rsi.iloc[-1]):
            return 50.0
        return float(rsi.iloc[-1])
    
    def _calculate_macd_signal(self, prices: pd.Series) -> float:
        """Calculate MACD signal."""
        if prices.empty or len(prices) < 26:
            return 0.0  # No signal if not enough data
        
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        
        # Handle NaN values
        if pd.isna(macd.iloc[-1]) or pd.isna(signal.iloc[-1]):
            return 0.0
        return float(macd.iloc[-1] - signal.iloc[-1])
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate Bollinger Bands position."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        current_price = prices.iloc[-1]
        bb_range = upper_band.iloc[-1] - lower_band.iloc[-1]
        if bb_range > 0:
            return (current_price - lower_band.iloc[-1]) / bb_range
        return 0.5
    
    def calculate_fundamental_score(self, symbol: str) -> Tuple[float, List[str]]:
        """Calculate fundamental score."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            reasons = []
            score = 0.0
            
            # P/E ratio
            pe_ratio = info.get('trailingPE', 0)
            if 10 <= pe_ratio <= 30:
                score += 0.3
                reasons.append(f"Reasonable P/E: {pe_ratio:.1f}")
            
            # Revenue growth
            revenue_growth = info.get('revenueGrowth', 0)
            if revenue_growth and revenue_growth > 0.1:
                score += 0.4
                reasons.append(f"Strong revenue growth: {revenue_growth:.1%}")
            
            # Profit margins
            profit_margins = info.get('profitMargins', 0)
            if profit_margins and profit_margins > 0.1:
                score += 0.3
                reasons.append(f"Good profit margins: {profit_margins:.1%}")
            
            return min(score, 1.0), reasons
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate fundamental score for {symbol}: {e}")
            return 0.0, []
    
    def calculate_sector_score(self, symbol: str) -> Tuple[float, List[str]]:
        """Calculate sector rotation score."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            sector = info.get('sector', '')
            
            # Hot sectors (can be updated based on market conditions)
            hot_sectors = ['Technology', 'Healthcare', 'Consumer Cyclical']
            
            if sector in hot_sectors:
                return 0.8, [f"Hot sector: {sector}"]
            else:
                return 0.3, [f"Sector: {sector}"]
                
        except Exception as e:
            return 0.5, ["Sector info unavailable"]
    
    def calculate_market_cap_score(self, symbol: str) -> Tuple[float, List[str]]:
        """Calculate market cap score."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            market_cap = info.get('marketCap', 0)
            
            # Prefer mid to large cap stocks
            if 10000000000 <= market_cap <= 100000000000:  # $10B - $100B
                return 0.8, [f"Mid-cap: ${market_cap/1e9:.1f}B"]
            elif 100000000000 <= market_cap <= 1000000000000:  # $1T
                return 0.9, [f"Large-cap: ${market_cap/1e9:.1f}B"]
            else:
                return 0.5, [f"Market cap: ${market_cap/1e9:.1f}B"]
                
        except Exception as e:
            return 0.5, ["Market cap info unavailable"]
    
    async def select_best_stocks(self, max_stocks: int = 10) -> List[StockScore]:
        """Select the best stocks for trading (always return top N, even if scores are low)."""
        self.logger.info(f"Selecting best {max_stocks} stocks from {len(self.stock_universe)} candidates")
        
        # Get market data for all stocks
        market_data = await self.get_market_data(self.stock_universe)
        print(f"[DEBUG] Market data fetched for {len(market_data)} / {len(self.stock_universe)} stocks")
        
        stock_scores = []
        
        for symbol in self.stock_universe:
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            
            # Calculate individual scores
            momentum_score, momentum_reasons = self.calculate_momentum_score(data)
            volatility_score, volatility_reasons = self.calculate_volatility_score(data)
            volume_score, volume_reasons = self.calculate_volume_score(data)
            technical_score, technical_reasons = self.calculate_technical_score(data)
            fundamental_score, fundamental_reasons = self.calculate_fundamental_score(symbol)
            sector_score, sector_reasons = self.calculate_sector_score(symbol)
            market_cap_score, market_cap_reasons = self.calculate_market_cap_score(symbol)
            
            # Calculate weighted total score
            weights = self.config['scoring']
            total_score = (
                momentum_score * weights['momentum_weight'] +
                volatility_score * weights['volatility_weight'] +
                volume_score * weights['volume_weight'] +
                technical_score * weights['technical_weight'] +
                fundamental_score * weights['fundamental_weight'] +
                sector_score * weights['sector_weight'] +
                market_cap_score * weights['market_cap_weight']
            )
            
            # Combine all reasons
            all_reasons = (momentum_reasons + volatility_reasons + volume_reasons + 
                          technical_reasons + fundamental_reasons + sector_reasons + 
                          market_cap_reasons)
            
            stock_score = StockScore(
                symbol=symbol,
                total_score=total_score,
                momentum_score=momentum_score,
                volatility_score=volatility_score,
                volume_score=volume_score,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                sector_score=sector_score,
                market_cap_score=market_cap_score,
                reasons=all_reasons
            )
            
            stock_scores.append(stock_score)
        
        # Sort by total score and return top N stocks (even if scores are low)
        stock_scores.sort(key=lambda x: x.total_score, reverse=True)
        selected_stocks = stock_scores[:max_stocks]
        
        self.logger.info(f"Selected {len(selected_stocks)} stocks:")
        for stock in selected_stocks:
            self.logger.info(f"  {stock.symbol}: {stock.total_score:.3f} - {', '.join(stock.reasons[:3])}")
        
        return selected_stocks
    
    def get_stock_universe(self) -> List[str]:
        """Get current stock universe."""
        return self.stock_universe.copy()
    
    def update_stock_universe(self, new_stocks: List[str]):
        """Update stock universe."""
        self.stock_universe = list(set(self.stock_universe + new_stocks))
        self.logger.info(f"Updated stock universe: {len(self.stock_universe)} stocks")

async def main():
    """Test the stock selector."""
    selector = AutomatedStockSelector()
    
    print("🔍 Selecting best stocks for trading...")
    selected_stocks = await selector.select_best_stocks(max_stocks=10)
    
    print(f"\n📊 Top {len(selected_stocks)} Stocks Selected:")
    print("=" * 80)
    
    for i, stock in enumerate(selected_stocks, 1):
        print(f"{i:2d}. {stock.symbol:6s} | Score: {stock.total_score:.3f}")
        print(f"     Momentum: {stock.momentum_score:.2f} | Volatility: {stock.volatility_score:.2f} | Volume: {stock.volume_score:.2f}")
        print(f"     Technical: {stock.technical_score:.2f} | Fundamental: {stock.fundamental_score:.2f}")
        print(f"     Reasons: {', '.join(stock.reasons[:3])}")
        print()

if __name__ == "__main__":
    asyncio.run(main()) 