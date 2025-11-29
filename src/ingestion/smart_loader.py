"""Smart data loader that combines CSV and live data from multiple sources with failsafe."""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional

from src.ingestion.csv_handler import CSVHandler
from src.ingestion.multi_source_fetcher import MultiSourceFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SmartDataLoader:
    """Smart data loader with multi-source failsafe mechanism."""
    
    def __init__(self, dataset_path: str = "dataset", days_threshold: int = 7):
        """
        Initialize smart data loader with multi-source fetcher.
        
        Args:
            dataset_path: Path to CSV dataset folder
            days_threshold: Number of days before data is considered stale
        """
        self.dataset_path = Path(dataset_path)
        self.days_threshold = days_threshold
        self.csv_handler = CSVHandler()
        self.multi_fetcher = MultiSourceFetcher()
        self.logger = logger
        
        # Log available sources
        sources = self.multi_fetcher.get_available_sources()
        self.logger.info(f"Initialized with {len(sources)} data sources: {', '.join(sources)}")
    
    def is_data_fresh(self, df: pd.DataFrame) -> Tuple[bool, Optional[datetime]]:
        """
        Check if data is fresh enough.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            Tuple of (is_fresh, last_date)
        """
        try:
            if 'date' not in df.columns:
                return False, None
            
            df['date'] = pd.to_datetime(df['date'])
            last_date = df['date'].max()
            days_old = (datetime.now() - last_date).days
            
            is_fresh = days_old <= self.days_threshold
            
            self.logger.info(f"Data age: {days_old} days, Fresh: {is_fresh}")
            return is_fresh, last_date
            
        except Exception as e:
            self.logger.error(f"Error checking data freshness: {str(e)}")
            return False, None
    
    def load_and_update(
        self, 
        symbol: str, 
        force_update: bool = False,
        auto_save: bool = True
    ) -> pd.DataFrame:
        """
        Load data from CSV first (reliable historical data).
        Only attempt YFinance updates if CSV is stale and force_update is True.
        
        Args:
            symbol: Stock symbol
            force_update: Try to update with YFinance (may fail, will use CSV anyway)
            auto_save: Automatically save updated data to CSV
            
        Returns:
            DataFrame with complete data (always from CSV)
        """
        csv_path = self.dataset_path / f"{symbol}.csv"
        
        # STEP 1: Load CSV data (primary source)
        df_csv = None
        
        if csv_path.exists():
            try:
                self.logger.info(f"Loading CSV data for {symbol}")
                df_csv = pd.read_csv(csv_path)
                df_csv = self.csv_handler.validate_csv(df_csv)
                
                is_fresh, last_date = self.is_data_fresh(df_csv)
                
                if is_fresh or not force_update:
                    self.logger.info(f"✓ Using CSV data ({len(df_csv)} records, last: {last_date})")
                    return df_csv
                    
                self.logger.info(f"CSV data is stale (last: {last_date}), will try to update...")
                
            except Exception as e:
                self.logger.warning(f"CSV load error: {str(e)}")
                df_csv = None
        
        # STEP 2: If no CSV exists, return error (we rely on CSV dataset)
        if df_csv is None:
            raise Exception(f"No CSV data found for {symbol}. Please ensure {symbol}.csv exists in dataset folder.")
        
        # STEP 3: Try YFinance update (optional, may fail - that's OK)
        if force_update:
            try:
                self.logger.info(f"Attempting YFinance update for {symbol}...")
                
                from src.ingestion.yfinance_fetcher import YFinanceFetcher
                yf_fetcher = YFinanceFetcher()
                
                df_new = yf_fetcher.fetch_stock_data(
                    symbol=symbol,
                    start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )
                
                if not df_new.empty:
                    # Merge CSV with new data
                    df_combined = pd.concat([df_csv, df_new], ignore_index=True)
                    df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')
                    df_combined = df_combined.sort_values('date').reset_index(drop=True)
                    
                    if auto_save:
                        self.dataset_path.mkdir(exist_ok=True)
                        df_combined.to_csv(csv_path, index=False)
                        self.logger.info(f"✓ Updated CSV with {len(df_new)} new records from YFinance")
                    
                    return df_combined
                else:
                    self.logger.info("YFinance returned no new data, using CSV")
                    return df_csv
                    
            except Exception as e:
                self.logger.warning(f"YFinance update failed (expected): {str(e)[:100]}")
                self.logger.info("✓ Using existing CSV data")
                return df_csv
        
        # Default: return CSV data
        return df_csv
    
    def get_data_info(self, symbol: str) -> dict:
        """
        Get information about data availability and freshness.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with data info
        """
        csv_path = self.dataset_path / f"{symbol}.csv"
        
        info = {
            "symbol": symbol,
            "csv_exists": csv_path.exists(),
            "csv_fresh": False,
            "last_date": None,
            "days_old": None,
            "total_records": 0
        }
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df = self.csv_handler.validate_csv(df)
                
                is_fresh, last_date = self.is_data_fresh(df)
                
                info["csv_fresh"] = is_fresh
                info["last_date"] = str(last_date) if last_date else None
                info["days_old"] = (datetime.now() - last_date).days if last_date else None
                info["total_records"] = len(df)
                
            except Exception as e:
                self.logger.error(f"Error getting data info: {str(e)}")
        
        return info
