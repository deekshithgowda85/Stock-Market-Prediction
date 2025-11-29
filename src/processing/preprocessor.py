"""Data preprocessing pipeline for stock data."""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StockDataPreprocessor:
    """Preprocessor for cleaning and preparing stock data."""

    def __init__(self):
        """Initialize preprocessor."""
        self.logger = logger
        self.scaler_params = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw stock data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        try:
            self.logger.info("Cleaning data...")
            df_clean = df.copy()

            # Remove duplicates
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=['date'], keep='last')
            removed_dups = initial_rows - len(df_clean)
            if removed_dups > 0:
                self.logger.info(f"Removed {removed_dups} duplicate rows")

            # Handle missing values
            missing_before = df_clean.isnull().sum().sum()
            
            # Forward fill for OHLCV
            df_clean[['open', 'high', 'low', 'close']] = df_clean[['open', 'high', 'low', 'close']].fillna(method='ffill')
            
            # Backward fill remaining
            df_clean[['open', 'high', 'low', 'close']] = df_clean[['open', 'high', 'low', 'close']].fillna(method='bfill')
            
            # Volume - fill with 0 or median
            if 'volume' in df_clean.columns:
                df_clean['volume'] = df_clean['volume'].fillna(df_clean['volume'].median())

            missing_after = df_clean.isnull().sum().sum()
            self.logger.info(f"Filled {missing_before - missing_after} missing values")

            # Remove rows with invalid prices (zero or negative)
            invalid_mask = (df_clean['close'] <= 0) | (df_clean['open'] <= 0) | (df_clean['high'] <= 0) | (df_clean['low'] <= 0)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                self.logger.warning(f"Removing {invalid_count} rows with invalid prices")
                df_clean = df_clean[~invalid_mask]

            # Ensure OHLC consistency
            df_clean['high'] = df_clean[['open', 'high', 'low', 'close']].max(axis=1)
            df_clean['low'] = df_clean[['open', 'high', 'low', 'close']].min(axis=1)

            # Sort by date
            df_clean = df_clean.sort_values('date').reset_index(drop=True)

            self.logger.info(f"Data cleaning complete. Final rows: {len(df_clean)}")
            return df_clean

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            raise

    def handle_outliers(self, df: pd.DataFrame, columns: list = None, method: str = 'iqr') -> pd.DataFrame:
        """
        Handle outliers in data.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (default: price columns)
            method: Method to use ('iqr', 'zscore', 'clip')

        Returns:
            DataFrame with outliers handled
        """
        try:
            if columns is None:
                columns = ['open', 'high', 'low', 'close']

            df_processed = df.copy()
            
            for col in columns:
                if col not in df_processed.columns:
                    continue

                if method == 'iqr':
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    outliers = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
                    if outliers > 0:
                        self.logger.info(f"Clipping {outliers} outliers in {col}")
                        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)

                elif method == 'zscore':
                    mean = df_processed[col].mean()
                    std = df_processed[col].std()
                    z_scores = np.abs((df_processed[col] - mean) / std)
                    outliers = (z_scores > 3).sum()
                    if outliers > 0:
                        self.logger.info(f"Clipping {outliers} outliers in {col} (z-score)")
                        df_processed.loc[z_scores > 3, col] = mean + 3 * std * np.sign(df_processed.loc[z_scores > 3, col] - mean)

            return df_processed

        except Exception as e:
            self.logger.error(f"Outlier handling failed: {str(e)}")
            raise

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators as features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with technical features
        """
        try:
            self.logger.info("Adding technical features...")
            df_features = df.copy()

            # Returns
            df_features['daily_return'] = df_features['close'].pct_change()
            df_features['log_return'] = np.log(df_features['close'] / df_features['close'].shift(1))

            # Moving Averages
            for window in [5, 10, 20, 50, 200]:
                df_features[f'ma_{window}'] = df_features['close'].rolling(window=window).mean()
                df_features[f'ma_{window}_ratio'] = df_features['close'] / df_features[f'ma_{window}']

            # Exponential Moving Averages
            for span in [12, 26]:
                df_features[f'ema_{span}'] = df_features['close'].ewm(span=span, adjust=False).mean()

            # MACD
            ema_12 = df_features['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df_features['close'].ewm(span=26, adjust=False).mean()
            df_features['macd'] = ema_12 - ema_26
            df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
            df_features['macd_histogram'] = df_features['macd'] - df_features['macd_signal']

            # Bollinger Bands
            df_features['bb_middle'] = df_features['close'].rolling(window=20).mean()
            bb_std = df_features['close'].rolling(window=20).std()
            df_features['bb_upper'] = df_features['bb_middle'] + (bb_std * 2)
            df_features['bb_lower'] = df_features['bb_middle'] - (bb_std * 2)
            df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
            df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / (df_features['bb_upper'] - df_features['bb_lower'])

            # RSI
            delta = df_features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_features['rsi'] = 100 - (100 / (1 + rs))

            # Volatility
            df_features['volatility_20'] = df_features['daily_return'].rolling(window=20).std()
            df_features['volatility_50'] = df_features['daily_return'].rolling(window=50).std()

            # Volume indicators
            if 'volume' in df_features.columns:
                df_features['volume_ma_20'] = df_features['volume'].rolling(window=20).mean()
                df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma_20']

            # Price momentum
            for period in [5, 10, 20]:
                df_features[f'momentum_{period}'] = df_features['close'] - df_features['close'].shift(period)
                df_features[f'roc_{period}'] = ((df_features['close'] - df_features['close'].shift(period)) / df_features['close'].shift(period)) * 100

            # High-Low range
            df_features['hl_ratio'] = (df_features['high'] - df_features['low']) / df_features['close']
            df_features['oc_ratio'] = (df_features['close'] - df_features['open']) / df_features['open']

            self.logger.info(f"Added {len(df_features.columns) - len(df.columns)} technical features")
            return df_features

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise

    def prepare_for_modeling(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'close',
        drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling.

        Args:
            df: Input DataFrame with features
            target_column: Target variable column
            drop_na: Whether to drop rows with NaN values

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            self.logger.info("Preparing data for modeling...")
            
            df_model = df.copy()
            
            # Drop NaN values if requested
            if drop_na:
                rows_before = len(df_model)
                df_model = df_model.dropna()
                rows_after = len(df_model)
                if rows_before > rows_after:
                    self.logger.info(f"Dropped {rows_before - rows_after} rows with NaN values")

            # Separate features and target
            if target_column in df_model.columns:
                target = df_model[target_column]
                features = df_model.drop(columns=[target_column])
            else:
                raise ValueError(f"Target column '{target_column}' not found")

            # Keep only date and numeric columns for features
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features = features[['date'] + list(numeric_cols)]

            self.logger.info(f"Prepared {len(features)} samples with {len(features.columns)} features")
            return features, target

        except Exception as e:
            self.logger.error(f"Data preparation for modeling failed: {str(e)}")
            raise

    def full_pipeline(
        self, 
        df: pd.DataFrame,
        add_features: bool = True,
        handle_outliers: bool = True
    ) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.

        Args:
            df: Raw DataFrame
            add_features: Whether to add technical features
            handle_outliers: Whether to handle outliers

        Returns:
            Fully preprocessed DataFrame
        """
        try:
            self.logger.info("Running full preprocessing pipeline...")
            
            # Step 1: Clean data
            df_processed = self.clean_data(df)
            
            # Step 2: Handle outliers
            if handle_outliers:
                df_processed = self.handle_outliers(df_processed)
            
            # Step 3: Add technical features
            if add_features:
                df_processed = self.add_technical_features(df_processed)
            
            self.logger.info("Preprocessing pipeline complete")
            return df_processed

        except Exception as e:
            self.logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of the data.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with summary statistics
        """
        try:
            summary = {
                'total_rows': len(df),
                'date_range': {
                    'start': str(df['date'].min()),
                    'end': str(df['date'].max()),
                },
                'missing_values': df.isnull().sum().to_dict(),
                'price_stats': {
                    'close_mean': float(df['close'].mean()),
                    'close_std': float(df['close'].std()),
                    'close_min': float(df['close'].min()),
                    'close_max': float(df['close'].max()),
                },
                'columns': list(df.columns)
            }
            return summary

        except Exception as e:
            self.logger.error(f"Summary generation failed: {str(e)}")
            return {}
