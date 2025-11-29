"""
NSE API Fetcher - Fetches real-time stock data from Indian Stock Market API
Replaces YFinance with more reliable NSE/BSE data source
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from src.utils.logger import get_logger
from src.utils.exceptions import NoDataException

logger = get_logger(__name__)

class NSEAPIFetcher:
    """Fetches stock data from NSE/BSE API"""
    
    BASE_URL = "https://nse-api-khaki.vercel.app"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_stock_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Fetch stock data from NSE API
        
        Args:
            symbol: Stock symbol (e.g., 'AXISBANK', 'RELIANCE')
            exchange: Exchange type - 'NSE' or 'BSE' (default: NSE)
            period: Not used for NSE API (kept for compatibility)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Add exchange suffix
            if exchange.upper() == "BSE":
                ticker = f"{symbol}.BO"
            else:  # Default to NSE
                ticker = f"{symbol}.NS"
            
            logger.info(f"Fetching data from NSE API for {ticker}")
            
            # Fetch current stock data
            url = f"{self.BASE_URL}/stock?symbol={ticker}&res=num"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'success':
                raise NoDataException(f"API returned error status for {ticker}")
            
            stock_data = data.get('data', {})
            
            if not stock_data:
                raise NoDataException(f"No data available for {ticker}")
            
            # Convert to DataFrame format matching CSV structure
            # Since NSE API only returns current data, we'll create a single row
            df = pd.DataFrame([{
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Open': stock_data.get('open', stock_data.get('last_price')),
                'High': stock_data.get('day_high', stock_data.get('last_price')),
                'Low': stock_data.get('day_low', stock_data.get('last_price')),
                'Close': stock_data.get('last_price'),
                'Adj Close': stock_data.get('last_price'),
                'Volume': stock_data.get('volume', 0)
            }])
            
            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            logger.info(f"Successfully fetched data for {ticker}: {len(df)} records")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching {symbol}: {str(e)}")
            raise NoDataException(f"Failed to fetch data from NSE API: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            raise NoDataException(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def search_stock(self, query: str) -> list:
        """
        Search for stocks by company name
        
        Args:
            query: Company name or partial symbol
        
        Returns:
            List of matching stocks with symbols
        """
        try:
            url = f"{self.BASE_URL}/search?q={query}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'success':
                return data.get('results', [])
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching for {query}: {str(e)}")
            return []
    
    def validate_symbol(self, symbol: str, exchange: str = "NSE") -> bool:
        """
        Validate if a stock symbol exists
        
        Args:
            symbol: Stock symbol
            exchange: Exchange type (NSE/BSE)
        
        Returns:
            True if symbol exists, False otherwise
        """
        try:
            self.fetch_stock_data(symbol, exchange)
            return True
        except NoDataException:
            return False
