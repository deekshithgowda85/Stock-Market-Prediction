
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
from datetime import datetime

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not installed. Please run: pip install lightgbm")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiStockLightGBM:
    """
    Multi-Stock LightGBM model - much faster than LSTM
    Trains on all stocks in 2-5 minutes
    """
    
    def __init__(self, lookback: int = 60):
        """
        Initialize Multi-Stock LightGBM
        
        Args:
            lookback: Number of days to look back for features
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Create features from stock data
        
        Args:
            df: Stock dataframe with OHLCV data
            symbol: Stock symbol
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Price features
        features['close'] = df['close']
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume']
        
        # Returns
        features['return_1d'] = df['close'].pct_change(1)
        features['return_5d'] = df['close'].pct_change(5)
        features['return_20d'] = df['close'].pct_change(20)
        
        # Moving averages
        features['ma_5'] = df['close'].rolling(5).mean()
        features['ma_10'] = df['close'].rolling(10).mean()
        features['ma_20'] = df['close'].rolling(20).mean()
        features['ma_50'] = df['close'].rolling(50).mean()
        
        # MA ratios
        features['close_to_ma5'] = df['close'] / features['ma_5']
        features['close_to_ma20'] = df['close'] / features['ma_20']
        features['ma5_to_ma20'] = features['ma_5'] / features['ma_20']
        
        # Volatility
        features['volatility_5d'] = df['close'].rolling(5).std()
        features['volatility_20d'] = df['close'].rolling(20).std()
        
        # Volume features
        features['volume_ma_5'] = df['volume'].rolling(5).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma_5']
        
        # Price ranges
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Momentum
        features['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # Lag features (previous N days)
        for lag in [1, 2, 3, 5, 10]:
            features[f'close_lag_{lag}'] = df['close'].shift(lag)
            features[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Target: next day's close price
        features['target'] = df['close'].shift(-1)
        
        # Add symbol as categorical feature
        features['symbol'] = symbol
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_data(self, dataset_path: Path, use_spark: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare all stock data with optional PySpark preprocessing
        
        Args:
            dataset_path: Path to dataset folder
            use_spark: Use PySpark for distributed preprocessing (default: True)
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        logger.info("Loading all stock datasets...")
        
        if use_spark:
            # Use PySpark for distributed data loading and cleaning
            try:
                from src.preprocessing.spark_loader import SparkPreprocessor
                logger.info("🔥 Using PySpark for data preprocessing...")
                
                preprocessor = SparkPreprocessor()
                
                # Load all stocks with PySpark (faster, more memory efficient)
                all_data = preprocessor.load_and_clean_stocks(str(dataset_path))
                
                logger.info(f"PySpark loaded {len(all_data):,} total records")
                
                # Process each symbol
                all_features = []
                for symbol in all_data['symbol'].unique():
                    df = all_data[all_data['symbol'] == symbol].copy()
                    
                    if len(df) < self.lookback + 50:
                        logger.warning(f"Skipping {symbol}: insufficient data ({len(df)} rows)")
                        continue
                    
                    # Create features
                    features = self.create_features(df, symbol)
                    all_features.append(features)
                    logger.info(f"Processed {symbol}: {len(df)} rows")
                
                # Combine all stocks
                combined_df = pd.concat(all_features, ignore_index=True)
                
            except Exception as e:
                logger.warning(f"PySpark preprocessing failed: {e}. Falling back to pandas...")
                use_spark = False
        
        if not use_spark:
            # Fallback to pandas-based loading
            all_features = []
            csv_files = list(dataset_path.glob("*.csv"))
            logger.info(f"Found {len(csv_files)} stock files")
            
            for csv_file in csv_files:
                symbol = csv_file.stem
                
                # Skip metadata and NIFTY50_all
                if symbol in ['stock_metadata', 'NIFTY50_all']:
                    continue
                
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Normalize column names to lowercase
                    df.columns = df.columns.str.lower()
                    
                    # Check required columns
                    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                    if not all(col in df.columns for col in required_cols):
                        logger.warning(f"Skipping {symbol}: missing required columns")
                        continue
                    
                    # Clean data
                    df = df.dropna(subset=required_cols)
                    
                    if len(df) < self.lookback + 50:
                        logger.warning(f"Skipping {symbol}: insufficient data ({len(df)} rows)")
                        continue
                    
                    # Create features
                    features = self.create_features(df, symbol)
                    all_features.append(features)
                    
                    logger.info(f"Loaded {symbol}: {len(df)} rows")
                    
                except Exception as e:
                    logger.error(f"Error loading {symbol}: {e}")
                    continue
            
            # Combine all stocks
            combined_df = pd.concat(all_features, ignore_index=True)
        
        # Drop rows with NaN (from feature engineering)
        combined_df = combined_df.dropna()
        
        logger.info(f"Total samples after feature engineering: {len(combined_df)}")
        
        return combined_df
    
    def train(self, dataset_path: Path, test_size: float = 0.2):
        """
        Train LightGBM on all stocks
        
        Args:
            dataset_path: Path to dataset folder
            test_size: Fraction of data for validation
        """
        logger.info("="*80)
        logger.info("Multi-Stock LightGBM Training")
        logger.info("="*80)
        
        # Prepare data
        data = self.prepare_data(dataset_path)
        
        # Separate features and target
        X = data.drop(['target', 'symbol'], axis=1)
        y = data['target']
        symbols = data['symbol']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test, sym_train, sym_test = train_test_split(
            X, y, symbols, test_size=test_size, random_state=42, shuffle=True
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Features: {len(self.feature_names)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        
        # Parameters optimized for speed and accuracy
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1,
            'num_threads': -1,  # Use all CPU cores
            'force_col_wise': True,
        }
        
        logger.info("Starting training...")
        start_time = datetime.now()
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")
        
        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
        
        return self.model
    
    def predict(self, symbol: str, df: pd.DataFrame, days: int = 30) -> Dict:
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        dates = []
        
        # Create features from historical data
        features_df = self.create_features(df, symbol)
        features_df = features_df.dropna()
        
        # Get last row for prediction
        last_features = features_df.drop(['target', 'symbol'], axis=1).iloc[-1:].copy()
        
        # Keep track of recent close prices for rolling calculations
        recent_closes = df['close'].tail(50).tolist()
        recent_volumes = df['volume'].tail(10).tolist() if 'volume' in df.columns else []
        
        # Make predictions
        for i in range(days):
            # Scale features
            X_scaled = self.scaler.transform(last_features)
            
            # Predict
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            
            # Update features for next prediction - properly update all rolling features
            recent_closes.append(pred)
            
            # Update price-based features
            last_features.loc[:, 'close'] = pred
            
            # Update moving averages
            if len(recent_closes) >= 5:
                last_features.loc[:, 'ma_5'] = np.mean(recent_closes[-5:])
            if len(recent_closes) >= 10:
                last_features.loc[:, 'ma_10'] = np.mean(recent_closes[-10:])
            if len(recent_closes) >= 20:
                last_features.loc[:, 'ma_20'] = np.mean(recent_closes[-20:])
            if len(recent_closes) >= 50:
                last_features.loc[:, 'ma_50'] = np.mean(recent_closes[-50:])
            
            # Update MA ratios
            if 'ma_5' in last_features.columns and 'ma_20' in last_features.columns:
                ma_5 = last_features['ma_5'].iloc[0]
                ma_20 = last_features['ma_20'].iloc[0]
                if ma_5 > 0:
                    last_features.loc[:, 'close_to_ma5'] = (pred / ma_5 - 1) * 100
                if ma_20 > 0:
                    last_features.loc[:, 'close_to_ma20'] = (pred / ma_20 - 1) * 100
                    last_features.loc[:, 'ma5_to_ma20'] = (ma_5 / ma_20 - 1) * 100
            
            # Update returns
            if len(recent_closes) >= 2:
                last_features.loc[:, 'return_1d'] = (recent_closes[-1] / recent_closes[-2] - 1) * 100
            if len(recent_closes) >= 6:
                last_features.loc[:, 'return_5d'] = (recent_closes[-1] / recent_closes[-6] - 1) * 100
            if len(recent_closes) >= 21:
                last_features.loc[:, 'return_20d'] = (recent_closes[-1] / recent_closes[-21] - 1) * 100
            
            # Update volatility
            if len(recent_closes) >= 5:
                returns_5d = [(recent_closes[j] / recent_closes[j-1] - 1) for j in range(-5, 0)]
                last_features.loc[:, 'volatility_5d'] = np.std(returns_5d) * 100
            if len(recent_closes) >= 20:
                returns_20d = [(recent_closes[j] / recent_closes[j-1] - 1) for j in range(-20, 0)]
                last_features.loc[:, 'volatility_20d'] = np.std(returns_20d) * 100
            
            # Update lag features
            if len(recent_closes) >= 2:
                last_features.loc[:, 'close_lag_1'] = recent_closes[-2]
            if len(recent_closes) >= 3:
                last_features.loc[:, 'close_lag_2'] = recent_closes[-3]
            if len(recent_closes) >= 4:
                last_features.loc[:, 'close_lag_3'] = recent_closes[-4]
            if len(recent_closes) >= 6:
                last_features.loc[:, 'close_lag_5'] = recent_closes[-6]
            if len(recent_closes) >= 11:
                last_features.loc[:, 'close_lag_10'] = recent_closes[-11]
            
        return {
            'symbol': symbol,
            'predictions': predictions,
            'days': days
        }
    
    def save_model(self, save_path: Path):
        """Save trained model"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        model_file = save_path / 'lightgbm_model.txt'
        self.model.save_model(str(model_file))
        
        # Save scaler and metadata
        metadata = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'lookback': self.lookback
        }
        
        with open(save_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Path):
        """Load trained model"""
        # Load LightGBM model
        model_file = load_path / 'lightgbm_model.txt'
        self.model = lgb.Booster(model_file=str(model_file))
        
        # Load scaler and metadata
        with open(load_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.feature_names = metadata['feature_names']
        self.lookback = metadata['lookback']
        
        logger.info(f"Model loaded from {load_path}")
