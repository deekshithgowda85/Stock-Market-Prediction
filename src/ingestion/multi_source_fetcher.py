"""
Multi-source stock data fetcher with failsafe mechanism
Tries multiple APIs in order until one succeeds
"""

import pandas as pd
from typing import Optional, List, Tuple
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.exceptions import NoDataException

logger = get_logger(__name__)


class MultiSourceFetcher:
    """
    Fetches stock data from multiple sources with automatic failover
    Priority: CSV (historical data) → YFinance (live updates)
    """
    
    def __init__(self):
        self.sources = []
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize all available data sources"""
        # Import sources dynamically to avoid import errors
        
        # Priority 1: CSV (always reliable, historical data)
        try:
            from src.ingestion.csv_handler import CSVHandler
            self.sources.append(('CSV', CSVHandler()))
            logger.info("✓ CSV source initialized")
        except Exception as e:
            logger.warning(f"CSV source unavailable: {e}")
        
        # Priority 2: YFinance (for live updates when available)
        try:
            from src.ingestion.yfinance_fetcher import YFinanceFetcher
            self.sources.append(('YFinance', YFinanceFetcher()))
            logger.info("✓ YFinance source initialized")
        except Exception as e:
            logger.warning(f"YFinance source unavailable: {e}")
    
    def fetch_stock_data(
        self, 
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1d"
    ) -> Tuple[pd.DataFrame, str]:
        """
        Fetch stock data with automatic failover between sources
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period for data (1d, 5d, 1mo, etc.)
        
        Returns:
            Tuple of (DataFrame, source_name)
        """
        last_error = None
        
        for source_name, fetcher in self.sources:
            try:
                logger.info(f"Attempting to fetch {symbol} from {source_name}...")
                
                # Try to fetch data based on source capabilities
                if hasattr(fetcher, 'fetch_stock_data'):
                    try:
                        if source_name == 'CSV':
                            # CSV handler - load from dataset folder
                            df = fetcher.load_csv(symbol)
                        elif source_name == 'YFinance':
                            # YFinance supports historical data
                            if start_date and end_date:
                                df = fetcher.fetch_stock_data(
                                    symbol=symbol,
                                    start_date=start_date,
                                    end_date=end_date
                                )
                            else:
                                df = fetcher.fetch_stock_data(
                                    symbol=symbol,
                                    period=period
                                )
                        else:
                            # Generic fetcher
                            df = fetcher.fetch_stock_data(symbol=symbol)
                    except Exception as fetch_error:
                        logger.warning(f"✗ {source_name} fetch failed: {str(fetch_error)[:100]}")
                        continue
                    
                    if not df.empty:
                        logger.info(f"✓ Successfully fetched {len(df)} records from {source_name}")
                        return df, source_name
                    else:
                        logger.warning(f"✗ {source_name} returned empty data")
                        
            except Exception as e:
                last_error = str(e)
                logger.warning(f"✗ {source_name} failed: {str(e)[:100]}")
                continue
        
        # All sources failed
        error_msg = f"All data sources failed for {symbol}. Last error: {last_error}"
        logger.error(error_msg)
        raise NoDataException(error_msg)
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources"""
        return [name for name, _ in self.sources]
    
    def test_all_sources(self, symbol: str = "RELIANCE") -> dict:
        """
        Test all sources with a sample symbol
        
        Args:
            symbol: Symbol to test with
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        for source_name, fetcher in self.sources:
            try:
                logger.info(f"Testing {source_name}...")
                
                if source_name == 'YFinance':
                    df = fetcher.fetch_stock_data(symbol=symbol, period="1d")
                elif source_name == 'NSE Official':
                    df = fetcher.fetch_stock_data(symbol)
                
                results[source_name] = {
                    'status': 'success',
                    'records': len(df),
                    'last_price': df['Close'].iloc[-1] if not df.empty else None
                }
                logger.info(f"✓ {source_name}: Success ({len(df)} records)")
                
            except Exception as e:
                results[source_name] = {
                    'status': 'failed',
                    'error': str(e)[:100]
                }
                logger.error(f"✗ {source_name}: {str(e)[:100]}")
        
        return results
