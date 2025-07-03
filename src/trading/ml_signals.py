#!/usr/bin/env python3
"""
Machine Learning Signal Generator
Advanced ML-based trading signal generation using multiple models and ensemble methods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    ML_AVAILABLE = True
    XGBOOST_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False
        
except ImportError:
    ML_AVAILABLE = False
    XGBOOST_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False
    logging.warning("ML libraries not available. Install sklearn")

class ModelType(Enum):
    """Types of ML models."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"

@dataclass
class MLSignal:
    """ML-generated trading signal."""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    model_predictions: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    model_type: str = "ml"

class FeatureEngineer:
    """Feature engineering for ML models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features."""
        features = df.copy()
        
        # Price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        features['price_change'] = features['close'] - features['close'].shift(1)
        features['price_change_pct'] = features['price_change'] / features['close'].shift(1)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = features['close'].rolling(window=window).mean()
            features[f'ema_{window}'] = features['close'].ewm(span=window).mean()
            features[f'price_sma_{window}_ratio'] = features['close'] / features[f'sma_{window}']
            features[f'price_ema_{window}_ratio'] = features['close'] / features[f'ema_{window}']
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()
            features[f'volatility_annualized_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)
        
        # RSI
        for window in [14, 21]:
            features[f'rsi_{window}'] = self._calculate_rsi(features['close'], window)
        
        # MACD
        features['macd'], features['macd_signal'], features['macd_histogram'] = self._calculate_macd(features['close'])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(features['close'])
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_position'] = (features['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume features
        if 'volume' in features.columns:
            features['volume_ma_5'] = features['volume'].rolling(window=5).mean()
            features['volume_ma_20'] = features['volume'].rolling(window=20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_ma_20']
            features['volume_price_trend'] = (features['volume'] * features['returns']).rolling(window=10).sum()
        
        # Momentum features
        for window in [5, 10, 20]:
            features[f'momentum_{window}'] = features['close'] / features['close'].shift(window) - 1
            features[f'roc_{window}'] = (features['close'] - features['close'].shift(window)) / features['close'].shift(window)
        
        # ATR (Average True Range)
        features['atr'] = self._calculate_atr(features)
        
        # Stochastic Oscillator
        features['stoch_k'], features['stoch_d'] = self._calculate_stochastic(features)
        
        # Williams %R
        features['williams_r'] = self._calculate_williams_r(features)
        
        # Remove NaN values
        features = features.dropna()
        
        # Store feature names
        self.feature_names = [col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = df['low'].rolling(window=k_window).min()
        highest_high = df['high'].rolling(window=k_window).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = df['high'].rolling(window=window).max()
        lowest_low = df['low'].rolling(window=window).min()
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        return williams_r

class MLSignalGenerator:
    """Machine learning-based signal generator."""
    
    def __init__(self, config: dict):
        """Initialize ML signal generator."""
        if not ML_AVAILABLE:
            raise ImportError("ML libraries not available. Install required packages.")
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        
        # Models
        self.models = {}
        self.model_weights = {}
        self.is_trained = False
        
        # Configuration
        self.lookback_period = config.get('ml_lookback_period', 60)
        self.prediction_threshold = config.get('ml_prediction_threshold', 0.6)
        self.retrain_frequency = config.get('ml_retrain_frequency', 30)  # days
        self.last_retrain = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models."""
        models_config = self.config.get('ml_models', {})
        
        # Random Forest
        if models_config.get('random_forest', True):
            self.models[ModelType.RANDOM_FOREST] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model_weights[ModelType.RANDOM_FOREST] = 0.2
        
        # Gradient Boosting
        if models_config.get('gradient_boosting', True):
            self.models[ModelType.GRADIENT_BOOSTING] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.model_weights[ModelType.GRADIENT_BOOSTING] = 0.2
        
        # XGBoost (only if available)
        if models_config.get('xgboost', True):
            try:
                import xgboost as xgb
                self.models[ModelType.XGBOOST] = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                self.model_weights[ModelType.XGBOOST] = 0.25
                XGBOOST_AVAILABLE = True
            except ImportError:
                self.logger.warning("XGBoost not available. Skipping XGBoost model.")
                XGBOOST_AVAILABLE = False
        
        # LightGBM (only if available)
        if models_config.get('lightgbm', True):
            try:
                import lightgbm as lgb
                self.models[ModelType.LIGHTGBM] = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                self.model_weights[ModelType.LIGHTGBM] = 0.25
                LIGHTGBM_AVAILABLE = True
            except ImportError:
                self.logger.warning("LightGBM not available. Skipping LightGBM model.")
                LIGHTGBM_AVAILABLE = False
        
        # Logistic Regression
        if models_config.get('logistic_regression', True):
            self.models[ModelType.LOGISTIC_REGRESSION] = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            self.model_weights[ModelType.LOGISTIC_REGRESSION] = 0.1
    
    def prepare_training_data(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for ML models."""
        all_features = []
        all_labels = []
        
        for symbol, data in market_data.items():
            if len(data) < self.lookback_period + 10:
                continue
            
            try:
                # Create features
                features = self.feature_engineer.create_technical_features(data)
                
                if len(features) < 20:
                    continue
                
                # Create labels (future returns)
                features['future_return'] = features['close'].shift(-5) / features['close'] - 1
                
                # Create binary labels
                features['label'] = np.where(features['future_return'] > 0.01, 1,  # Buy signal
                                   np.where(features['future_return'] < -0.01, -1, 0))  # Sell signal
                
                # Remove rows with NaN labels
                features = features.dropna(subset=['label'])
                
                if len(features) == 0:
                    continue
                
                # Select feature columns (only numerical)
                feature_cols = []
                for col in features.columns:
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'future_return', 'label']:
                        # Check if column is numerical
                        if pd.api.types.is_numeric_dtype(features[col]):
                            feature_cols.append(col)
                        else:
                            self.logger.warning(f"Skipping non-numerical column: {col}")
                
                if len(feature_cols) == 0:
                    self.logger.warning(f"No valid numerical features found for {symbol}")
                    continue
                
                all_features.append(features[feature_cols])
                all_labels.append(features['label'])
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data found")
        
        # Combine all data
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_labels, axis=0)
        
        # Remove any remaining NaN values
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def train_models(self, market_data: Dict[str, pd.DataFrame]):
        """Train ML models."""
        self.logger.info("Preparing training data...")
        X, y = self.prepare_training_data(market_data)
        
        if len(X) < 100:
            self.logger.warning("Insufficient training data")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"Training data shape: {X_train.shape}")
        self.logger.info(f"Test data shape: {X_test.shape}")
        
        # Train each model
        for model_type, model in self.models.items():
            self.logger.info(f"Training {model_type.value}...")
            
            try:
                if model_type in [ModelType.LOGISTIC_REGRESSION]:
                    model.fit(X_train_scaled, y_train)
                    train_score = model.score(X_train_scaled, y_train)
                    test_score = model.score(X_test_scaled, y_test)
                else:
                    model.fit(X_train, y_train)
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                
                self.logger.info(f"{model_type.value} - Train score: {train_score:.3f}, Test score: {test_score:.3f}")
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                self.logger.info(f"{model_type.value} - CV scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
            except Exception as e:
                self.logger.error(f"Error training {model_type.value}: {e}")
        
        self.is_trained = True
        self.last_retrain = datetime.now()
        
        return True
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, MLSignal]:
        """Generate ML-based trading signals."""
        if not self.is_trained:
            self.logger.warning("Models not trained. Training now...")
            if not self.train_models(market_data):
                return {}
        
        # Check if retraining is needed
        if self.last_retrain and (datetime.now() - self.last_retrain).days >= self.retrain_frequency:
            self.logger.info("Retraining models...")
            self.train_models(market_data)
        
        signals = {}
        
        for symbol, data in market_data.items():
            if len(data) < self.lookback_period:
                continue
            
            try:
                # Create features
                features = self.feature_engineer.create_technical_features(data)
                
                if len(features) == 0:
                    continue
                
                # Get latest features
                latest_features = features.iloc[-1:]
                feature_cols = [col for col in latest_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                
                if len(feature_cols) == 0:
                    continue
                
                X = latest_features[feature_cols]
                
                # Get predictions from all models
                predictions = {}
                ensemble_prediction = 0
                
                for model_type, model in self.models.items():
                    try:
                        if model_type == ModelType.LOGISTIC_REGRESSION:
                            X_scaled = self.scaler.transform(X)
                            pred_proba = model.predict_proba(X_scaled)[0]
                        else:
                            pred_proba = model.predict_proba(X)[0]
                        
                        # Convert probabilities to prediction (-1, 0, 1)
                        if len(pred_proba) == 2:  # Binary classification
                            pred = 1 if pred_proba[1] > 0.5 else -1
                            confidence = max(pred_proba)
                        else:  # Multi-class
                            pred = np.argmax(pred_proba) - 1  # Convert to -1, 0, 1
                            confidence = max(pred_proba)
                        
                        predictions[model_type.value] = pred
                        ensemble_prediction += pred * self.model_weights[model_type]
                        
                    except Exception as e:
                        self.logger.error(f"Error getting prediction from {model_type.value}: {e}")
                
                # Determine final signal
                if abs(ensemble_prediction) > self.prediction_threshold:
                    signal_type = 'buy' if ensemble_prediction > 0 else 'sell'
                    confidence = min(abs(ensemble_prediction), 1.0)
                else:
                    signal_type = 'hold'
                    confidence = 0.0
                
                # Create signal
                signal = MLSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_type=signal_type,
                    confidence=confidence,
                    model_predictions=predictions,
                    model_type="ml"
                )
                
                signals[symbol] = signal
                
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def get_feature_importance(self, model_type: ModelType = ModelType.RANDOM_FOREST) -> Dict[str, float]:
        """Get feature importance from a model."""
        if not self.is_trained or model_type not in self.models:
            return {}
        
        model = self.models[model_type]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = self.feature_engineer.feature_names
            
            if len(importance) == len(feature_names):
                return dict(zip(feature_names, importance))
        
        return {}
    
    def save_models(self, filepath: str):
        """Save trained models to disk."""
        if not self.is_trained:
            self.logger.warning("No trained models to save")
            return
        
        try:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_names': self.feature_engineer.feature_names,
                'model_weights': self.model_weights,
                'last_retrain': self.last_retrain
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_engineer.feature_names = model_data['feature_names']
            self.model_weights = model_data['model_weights']
            self.last_retrain = model_data['last_retrain']
            self.is_trained = True
            
            self.logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def evaluate_models(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Evaluate model performance."""
        if not self.is_trained:
            return {}
        
        X, y = self.prepare_training_data(market_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        results = {}
        
        for model_type, model in self.models.items():
            try:
                if model_type == ModelType.LOGISTIC_REGRESSION:
                    X_test_scaled = self.scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                results[model_type.value] = {
                    'accuracy': report['accuracy'],
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1_score': report['weighted avg']['f1-score'],
                    'confusion_matrix': conf_matrix.tolist()
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_type.value}: {e}")
        
        return results

class EnsembleSignalGenerator:
    """Ensemble signal generator combining multiple approaches."""
    
    def __init__(self, config: dict):
        """Initialize ensemble signal generator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Signal generators
        self.ml_generator = MLSignalGenerator(config)
        self.technical_generator = None  # Will be set from main trading engine
        
        # Ensemble weights
        self.ml_weight = config.get('ensemble_ml_weight', 0.6)
        self.technical_weight = config.get('ensemble_technical_weight', 0.4)
    
    def set_technical_generator(self, technical_generator):
        """Set the technical signal generator."""
        self.technical_generator = technical_generator
    
    def generate_ensemble_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, MLSignal]:
        """Generate ensemble signals combining ML and technical analysis."""
        signals = {}
        
        # Get ML signals
        ml_signals = self.ml_generator.generate_signals(market_data)
        
        # Get technical signals (if available)
        technical_signals = {}
        if self.technical_generator:
            technical_signals = self.technical_generator.generate_signals(market_data)
        
        # Combine signals
        for symbol in market_data.keys():
            ml_signal = ml_signals.get(symbol)
            technical_signal = technical_signals.get(symbol)
            
            if ml_signal and technical_signal:
                # Combine signals with weights
                combined_confidence = (
                    ml_signal.confidence * self.ml_weight +
                    technical_signal.confidence * self.technical_weight
                )
                
                # Determine signal type based on combined confidence
                if combined_confidence > 0.6:
                    signal_type = 'buy' if ml_signal.confidence > technical_signal.confidence else technical_signal.signal_type
                elif combined_confidence < -0.6:
                    signal_type = 'sell'
                else:
                    signal_type = 'hold'
                
                # Create ensemble signal
                ensemble_signal = MLSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_type=signal_type,
                    confidence=abs(combined_confidence),
                    model_predictions=ml_signal.model_predictions,
                    model_type="ensemble"
                )
                
                signals[symbol] = ensemble_signal
                
            elif ml_signal:
                signals[symbol] = ml_signal
            elif technical_signal:
                signals[symbol] = technical_signal
        
        return signals 