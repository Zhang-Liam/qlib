#!/usr/bin/env python3
"""
Optimized Signal Generator
Performance-optimized version with caching and vectorized calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import threading
from collections import defaultdict

class SignalType(Enum):
    """Signal types for trading decisions."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class SignalResult:
    """Result of signal analysis."""
    signal_type: SignalType
    confidence: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: Optional[List[str]] = None

class OptimizedTechnicalIndicators:
    """Optimized technical analysis indicators with caching."""
    
    def __init__(self):
        self._cache = {}
        self._cache_lock = threading.RLock()
    
    def _get_cache_key(self, data_hash: int, indicator: str, **params) -> str:
        """Generate cache key for indicator calculation."""
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        return f"{indicator}_{data_hash}_{param_str}"
    
    def _get_data_hash(self, data: pd.Series) -> int:
        """Get hash of data for caching."""
        return hash(tuple(data.tail(10).to_numpy()))  # Use last 10 values for hash
    
    @lru_cache(maxsize=1000)
    def sma_cached(self, data_tuple: tuple, window: int) -> tuple:
        """Cached Simple Moving Average."""
        data = pd.Series(data_tuple)
        result = data.rolling(window=window).mean()
        return tuple(result.to_numpy())
    
    def sma(self, data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average with caching."""
        data_hash = self._get_data_hash(data)
        cache_key = self._get_cache_key(data_hash, 'sma', window=window)
        
        with self._cache_lock:
            if cache_key in self._cache:
                return pd.Series(self._cache[cache_key], index=data.index)
        
        # Calculate and cache
        result = data.rolling(window=window).mean()
        
        with self._cache_lock:
            self._cache[cache_key] = tuple(result.to_numpy())
        
        return result
    
    @lru_cache(maxsize=1000)
    def ema_cached(self, data_tuple: tuple, window: int) -> tuple:
        """Cached Exponential Moving Average."""
        data = pd.Series(data_tuple)
        result = data.ewm(span=window).mean()
        return tuple(result.values)
    
    def ema(self, data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average with caching."""
        data_hash = self._get_data_hash(data)
        cache_key = self._get_cache_key(data_hash, 'ema', window=window)
        
        with self._cache_lock:
            if cache_key in self._cache:
                return pd.Series(self._cache[cache_key], index=data.index)
        
        # Calculate and cache
        result = data.ewm(span=window).mean()
        
        with self._cache_lock:
            self._cache[cache_key] = tuple(result.values)
        
        return result
    
    def rsi_vectorized(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Vectorized RSI calculation for better performance."""
        delta = data.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd_vectorized(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Vectorized MACD calculation."""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def bollinger_bands_vectorized(self, data: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Vectorized Bollinger Bands calculation."""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def atr_vectorized(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Vectorized ATR calculation."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

class OptimizedSignalGenerator:
    """Optimized signal generator with caching and vectorized calculations."""
    
    def __init__(self, config: dict):
        """Initialize optimized signal generator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = OptimizedTechnicalIndicators()
        
        # Pre-computed indicator cache
        self._indicator_cache = {}
        self._cache_lock = threading.RLock()
    
    def _get_indicator_cache_key(self, symbol: str, indicator: str, **params) -> str:
        """Generate cache key for indicator."""
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        return f"{symbol}_{indicator}_{param_str}"
    
    def _calculate_all_indicators(self, symbol: str, hist_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all indicators for a symbol with caching."""
        cache_key = f"{symbol}_all_indicators"
        
        with self._cache_lock:
            if cache_key in self._indicator_cache:
                return self._indicator_cache[cache_key]
        
        close_prices = hist_data['close']
        high_prices = hist_data['high']
        low_prices = hist_data['low']
        volumes = hist_data['volume']
        
        # Calculate all indicators vectorized
        indicators = {
            'sma_5': self.indicators.sma(close_prices, 5),
            'sma_20': self.indicators.sma(close_prices, 20),
            'sma_50': self.indicators.sma(close_prices, 50),
            'ema_12': self.indicators.ema(close_prices, 12),
            'ema_26': self.indicators.ema(close_prices, 26),
            'rsi': self.indicators.rsi_vectorized(close_prices, 14),
            'macd_line': self.indicators.macd_vectorized(close_prices)[0],
            'macd_signal': self.indicators.macd_vectorized(close_prices)[1],
            'macd_histogram': self.indicators.macd_vectorized(close_prices)[2],
            'bb_upper': self.indicators.bollinger_bands_vectorized(close_prices)[0],
            'bb_middle': self.indicators.bollinger_bands_vectorized(close_prices)[1],
            'bb_lower': self.indicators.bollinger_bands_vectorized(close_prices)[2],
            'atr': self.indicators.atr_vectorized(high_prices, low_prices, close_prices, 14),
            'volume_sma': self.indicators.sma(volumes, 20)
        }
        
        with self._cache_lock:
            self._indicator_cache[cache_key] = indicators
        
        return indicators
    
    def generate_signal(self, symbol: str, hist_data: pd.DataFrame) -> SignalResult:
        """Generate comprehensive trading signal (optimized)."""
        try:
            if len(hist_data) < 50:  # Need sufficient data
                return SignalResult(
                    signal_type=SignalType.HOLD,
                    confidence=0.5,
                    reasoning=["Insufficient historical data"]
                )
            
            # Get all indicators (cached)
            indicators = self._calculate_all_indicators(symbol, hist_data)
            
            # 1. Trend Analysis (vectorized)
            trend_score = self._analyze_trend_vectorized(hist_data['close'], indicators)
            
            # 2. Momentum Analysis (vectorized)
            momentum_score = self._analyze_momentum_vectorized(hist_data['close'], indicators)
            
            # 3. Volatility Analysis (vectorized)
            volatility_score = self._analyze_volatility_vectorized(hist_data, indicators)
            
            # 4. Volume Analysis (vectorized)
            volume_score = self._analyze_volume_vectorized(hist_data['volume'], hist_data['close'], indicators)
            
            # 5. Support/Resistance Analysis (vectorized)
            support_resistance_score = self._analyze_support_resistance_vectorized(hist_data, indicators)
            
            # 6. Market Regime Analysis (vectorized)
            market_regime_score = self._analyze_market_regime_vectorized(hist_data['close'])
            
            # Combine all scores (vectorized)
            final_score = self._combine_scores_vectorized([
                trend_score,
                momentum_score,
                volatility_score,
                volume_score,
                support_resistance_score,
                market_regime_score
            ])
            
            # Generate signal
            signal_type, confidence = self._score_to_signal(final_score)
            
            # Calculate price targets
            current_price = hist_data['close'].iloc[-1]
            price_target, stop_loss, take_profit = self._calculate_price_targets_vectorized(
                current_price, hist_data, indicators, confidence
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning_vectorized([
                ("Trend", trend_score),
                ("Momentum", momentum_score),
                ("Volatility", volatility_score),
                ("Volume", volume_score),
                ("Support/Resistance", support_resistance_score),
                ("Market Regime", market_regime_score)
            ])
            
            return SignalResult(
                signal_type=signal_type,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning
            )
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return SignalResult(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reasoning=[f"Error: {str(e)}"]
            )
    
    def _analyze_trend_vectorized(self, prices: pd.Series, indicators: Dict[str, pd.Series]) -> float:
        """Vectorized trend analysis."""
        current_price = prices.iloc[-1]
        
        # Get cached indicators
        sma_5 = indicators['sma_5']
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        
        # Trend alignment score (vectorized)
        short_trend = 1 if current_price > sma_5.iloc[-1] else -1
        medium_trend = 1 if current_price > sma_20.iloc[-1] else -1
        long_trend = 1 if current_price > sma_50.iloc[-1] else -1
        
        # Weighted trend score
        trend_score = (short_trend * 0.3 + medium_trend * 0.4 + long_trend * 0.3)
        
        # Additional trend strength (vectorized)
        trend_strength = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        trend_score *= (1 + trend_strength)
        
        return np.clip(trend_score, -1, 1)
    
    def _analyze_momentum_vectorized(self, prices: pd.Series, indicators: Dict[str, pd.Series]) -> float:
        """Vectorized momentum analysis."""
        # Get cached indicators
        rsi = indicators['rsi']
        macd_line = indicators['macd_line']
        macd_signal = indicators['macd_signal']
        
        current_rsi = rsi.iloc[-1]
        macd_signal_value = 1 if macd_line.iloc[-1] > macd_signal.iloc[-1] else -1
        
        # Price momentum (vectorized)
        roc_5 = (prices.iloc[-1] / prices.iloc[-6] - 1) * 100
        roc_10 = (prices.iloc[-1] / prices.iloc[-11] - 1) * 100
        
        # RSI signals
        rsi_signal = 0
        if current_rsi < 30:
            rsi_signal = 1  # Oversold
        elif current_rsi > 70:
            rsi_signal = -1  # Overbought
        else:
            rsi_signal = (current_rsi - 50) / 50  # Neutral
        
        # Combine momentum signals
        momentum_score = (
            rsi_signal * 0.4 +
            macd_signal_value * 0.3 +
            np.clip(roc_5 / 10, -1, 1) * 0.2 +
            np.clip(roc_10 / 20, -1, 1) * 0.1
        )
        
        return np.clip(momentum_score, -1, 1)
    
    def _analyze_volatility_vectorized(self, hist_data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> float:
        """Vectorized volatility analysis."""
        # Get cached indicators
        atr = indicators['atr']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        bb_middle = indicators['bb_middle']
        
        current_price = hist_data['close'].iloc[-1]
        current_atr = atr.iloc[-1]
        
        # Bollinger Band position
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # Volatility score
        volatility_score = 0
        
        # High volatility (wide bands) is good for trading
        bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
        if bb_width > 0.1:  # Wide bands
            volatility_score += 0.3
        
        # ATR-based volatility
        avg_atr = atr.tail(20).mean()
        if current_atr > avg_atr * 1.2:  # High volatility
            volatility_score += 0.3
        
        # Price near bands (potential reversal)
        if bb_position < 0.2 or bb_position > 0.8:
            volatility_score += 0.4
        
        return np.clip(volatility_score, -1, 1)
    
    def _analyze_volume_vectorized(self, volumes: pd.Series, prices: pd.Series, indicators: Dict[str, pd.Series]) -> float:
        """Vectorized volume analysis."""
        # Get cached indicators
        volume_sma = indicators['volume_sma']
        
        current_volume = volumes.iloc[-1]
        avg_volume = volume_sma.iloc[-1]
        
        # Volume ratio
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Price-volume relationship
        price_change = prices.iloc[-1] / prices.iloc[-2] - 1
        volume_score = 0
        
        # High volume with price increase is bullish
        if price_change > 0 and volume_ratio > 1.5:
            volume_score = 0.8
        # High volume with price decrease is bearish
        elif price_change < 0 and volume_ratio > 1.5:
            volume_score = -0.8
        # Low volume suggests weak moves
        elif volume_ratio < 0.5:
            volume_score = 0.1
        else:
            volume_score = 0.2 * (volume_ratio - 1)
        
        return np.clip(volume_score, -1, 1)
    
    def _analyze_support_resistance_vectorized(self, hist_data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> float:
        """Vectorized support/resistance analysis."""
        # Get cached indicators
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        sma_20 = indicators['sma_20']
        
        current_price = hist_data['close'].iloc[-1]
        high_prices = hist_data['high']
        low_prices = hist_data['low']
        
        # Find recent support and resistance levels
        recent_highs = high_prices.tail(20)
        recent_lows = low_prices.tail(20)
        
        resistance_level = recent_highs.max()
        support_level = recent_lows.min()
        
        # Distance to support/resistance
        distance_to_resistance = (resistance_level - current_price) / current_price
        distance_to_support = (current_price - support_level) / current_price
        
        # Support/resistance score
        if distance_to_resistance < 0.02:  # Near resistance
            return -0.8
        elif distance_to_support < 0.02:  # Near support
            return 0.8
        else:
            # Neutral position
            return 0.1
    
    def _analyze_market_regime_vectorized(self, prices: pd.Series) -> float:
        """Vectorized market regime analysis."""
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Market regime indicators
        volatility = returns.std()
        trend = returns.tail(20).mean()
        
        # Regime score
        if volatility > 0.02:  # High volatility
            if trend > 0:  # Bullish high volatility
                return 0.6
            else:  # Bearish high volatility
                return -0.6
        else:  # Low volatility
            if trend > 0:  # Bullish low volatility
                return 0.3
            else:  # Bearish low volatility
                return -0.3
    
    def _combine_scores_vectorized(self, scores: List[float]) -> float:
        """Vectorized score combination."""
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]  # Trend, Momentum, Volatility, Volume, S/R, Regime
        
        # Weighted average
        final_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return np.clip(final_score, -1, 1)
    
    def _score_to_signal(self, score: float) -> Tuple[SignalType, float]:
        """Convert score to signal type and confidence."""
        confidence = abs(score)
        
        if score >= 0.7:
            signal_type = SignalType.STRONG_BUY
        elif score >= 0.3:
            signal_type = SignalType.BUY
        elif score >= -0.3:
            signal_type = SignalType.HOLD
        elif score >= -0.7:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.STRONG_SELL
        
        return signal_type, confidence
    
    def _calculate_price_targets_vectorized(self, current_price: float, hist_data: pd.DataFrame, 
                                          indicators: Dict[str, pd.Series], confidence: float) -> Tuple[float, float, float]:
        """Vectorized price target calculation."""
        # Get cached indicators
        atr = indicators['atr']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        
        current_atr = atr.iloc[-1]
        
        # Price targets based on ATR
        target_distance = current_atr * 2 * confidence
        stop_distance = current_atr * 1.5
        
        price_target = current_price + target_distance
        stop_loss = current_price - stop_distance
        take_profit = current_price + target_distance * 1.5
        
        return price_target, stop_loss, take_profit
    
    def _generate_reasoning_vectorized(self, analysis_results: List[Tuple[str, float]]) -> List[str]:
        """Vectorized reasoning generation."""
        reasoning = []
        
        for name, score in analysis_results:
            if abs(score) > 0.5:
                direction = "bullish" if score > 0 else "bearish"
                strength = "strong" if abs(score) > 0.7 else "moderate"
                reasoning.append(f"{strength.capitalize()} {direction} {name} signal")
        
        if not reasoning:
            reasoning.append("Neutral market conditions")
        
        return reasoning
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, SignalResult]:
        """Generate signals for multiple symbols."""
        signals = {}
        
        for symbol, data in market_data.items():
            try:
                if len(data) < 50:  # Need sufficient data
                    continue
                
                signal = self.generate_signal(symbol, data)
                signals[symbol] = signal
                
            except Exception as e:
                self.logger.warning(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals
    
    def clear_cache(self):
        """Clear all caches."""
        with self._cache_lock:
            self._indicator_cache.clear()
        self.indicators._cache.clear() 