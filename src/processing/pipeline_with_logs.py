"""
Enhanced preprocessing pipeline with detailed logging for UI display
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from src.utils.logger import get_logger
from src.processing.preprocessor import StockDataPreprocessor

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Preprocessing pipeline with detailed step logging"""
    
    def __init__(self):
        self.preprocessor = StockDataPreprocessor()
        self.logs: List[Dict] = []
        
    def add_log(self, step: str, message: str, status: str = "success", data: Dict = None):
        """Add a log entry"""
        log_entry = {
            "step": step,
            "message": message,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        self.logs.append(log_entry)
        logger.info(f"[{step}] {message}")
        
    def get_logs(self) -> List[Dict]:
        """Get all logs"""
        return self.logs
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs = []
    
    def run_full_pipeline(self, df: pd.DataFrame, add_features: bool = True, handle_outliers: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Run complete preprocessing pipeline with detailed logging
        
        Args:
            df: Input DataFrame
            add_features: Whether to add technical features
            handle_outliers: Whether to handle outliers
            
        Returns:
            Tuple of (processed DataFrame, logs list)
        """
        self.clear_logs()
        
        try:
            # Step 1: Data Validation
            self.add_log(
                "1. Data Validation",
                f"Starting with {len(df)} records",
                "running",
                {"initial_rows": len(df), "columns": df.columns.tolist()}
            )
            
            initial_shape = df.shape
            initial_nulls = df.isnull().sum().sum()
            initial_duplicates = df.duplicated(subset=['date']).sum()
            
            self.add_log(
                "1. Data Validation",
                f"Initial state: {initial_shape[0]} rows, {initial_shape[1]} columns, {initial_nulls} null values, {initial_duplicates} duplicates",
                "success",
                {
                    "rows": initial_shape[0],
                    "columns": initial_shape[1],
                    "null_values": int(initial_nulls),
                    "duplicates": int(initial_duplicates)
                }
            )
            
            # Step 2: Data Cleaning
            self.add_log(
                "2. Data Cleaning",
                "Removing duplicates and handling missing values...",
                "running"
            )
            
            df_clean = self.preprocessor.clean_data(df)
            
            removed_rows = len(df) - len(df_clean)
            filled_nulls = initial_nulls - df_clean.isnull().sum().sum()
            
            self.add_log(
                "2. Data Cleaning",
                f"Removed {removed_rows} invalid rows, filled {filled_nulls} missing values",
                "success",
                {
                    "removed_rows": removed_rows,
                    "filled_nulls": int(filled_nulls),
                    "remaining_rows": len(df_clean),
                    "remaining_nulls": int(df_clean.isnull().sum().sum())
                }
            )
            
            # Step 3: Outlier Detection & Handling
            if handle_outliers:
                self.add_log(
                    "3. Outlier Handling",
                    "Detecting and handling outliers using IQR method...",
                    "running"
                )
                
                # Calculate outliers before
                outliers_before = {}
                for col in ['open', 'high', 'low', 'close']:
                    if col in df_clean.columns:
                        Q1 = df_clean[col].quantile(0.25)
                        Q3 = df_clean[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((df_clean[col] < Q1 - 3*IQR) | (df_clean[col] > Q3 + 3*IQR)).sum()
                        outliers_before[col] = int(outliers)
                
                df_clean = self.preprocessor.handle_outliers(df_clean, method='iqr')
                
                total_outliers = sum(outliers_before.values())
                self.add_log(
                    "3. Outlier Handling",
                    f"Handled {total_outliers} outliers across price columns",
                    "success",
                    {"outliers_by_column": outliers_before, "total_outliers": total_outliers}
                )
            else:
                self.add_log(
                    "3. Outlier Handling",
                    "Skipped (disabled)",
                    "info"
                )
            
            # Step 4: Feature Engineering
            if add_features:
                self.add_log(
                    "4. Feature Engineering",
                    "Adding technical indicators and features...",
                    "running"
                )
                
                features_before = len(df_clean.columns)
                df_features = self.preprocessor.add_technical_features(df_clean)
                features_added = len(df_features.columns) - features_before
                
                new_features = [col for col in df_features.columns if col not in df_clean.columns]
                
                self.add_log(
                    "4. Feature Engineering",
                    f"Added {features_added} new features (MA, EMA, RSI, MACD, Bollinger Bands, etc.)",
                    "success",
                    {
                        "features_added": features_added,
                        "total_features": len(df_features.columns),
                        "new_features": new_features[:10]  # Show first 10
                    }
                )
                
                df_clean = df_features
            else:
                self.add_log(
                    "4. Feature Engineering",
                    "Skipped (disabled)",
                    "info"
                )
            
            # Step 5: Data Normalization
            self.add_log(
                "5. Data Normalization",
                "Normalizing data for model input...",
                "running"
            )
            
            # Calculate statistics before normalization
            price_stats = {
                "close_mean": float(df_clean['close'].mean()),
                "close_std": float(df_clean['close'].std()),
                "close_min": float(df_clean['close'].min()),
                "close_max": float(df_clean['close'].max())
            }
            
            self.add_log(
                "5. Data Normalization",
                f"Price range: ₹{price_stats['close_min']:.2f} - ₹{price_stats['close_max']:.2f}",
                "success",
                price_stats
            )
            
            # Step 6: Final Validation
            self.add_log(
                "6. Final Validation",
                "Validating processed data...",
                "running"
            )
            
            final_shape = df_clean.shape
            final_nulls = df_clean.isnull().sum().sum()
            date_range = f"{df_clean['date'].min()} to {df_clean['date'].max()}"
            
            self.add_log(
                "6. Final Validation",
                f"Pipeline complete! {final_shape[0]} rows, {final_shape[1]} features ready for modeling",
                "success",
                {
                    "final_rows": final_shape[0],
                    "final_columns": final_shape[1],
                    "remaining_nulls": int(final_nulls),
                    "date_range": date_range,
                    "data_quality": "excellent" if final_nulls == 0 else "good"
                }
            )
            
            return df_clean, self.get_logs()
            
        except Exception as e:
            self.add_log(
                "Pipeline Error",
                f"Pipeline failed: {str(e)}",
                "error",
                {"error": str(e)}
            )
            logger.error(f"Pipeline error: {str(e)}")
            raise
