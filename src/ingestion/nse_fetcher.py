"""
NSE India fetcher - Direct from NSE website
Implements failsafe mechanism with multiple data sources
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from src.utils.logger import get_logger
from src.utils.exceptions import NoDataException

logger = get_logger(__name__)


class NSEFetcher:
    """Fetches real-time stock data from NSE India website"""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.session = requests.Session()
        
        # Headers to mimic browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin'
        }
        
        self.session.headers.update(self.headers)
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize session by visiting homepage to get cookies"""
        try:
            self.session.get(self.base_url, timeout=10)
            logger.info("NSE session initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize NSE session: {e}")
    
    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch current stock data from NSE
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
        
        Returns:
            DataFrame with current stock data
        """
        try:
            url = f"{self.base_url}/api/quote-equity?symbol={symbol}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'priceInfo' not in data:
                raise NoDataException(f"No price data for {symbol}")
            
            price_info = data['priceInfo']
            
            # Extract relevant data
            df = pd.DataFrame([{
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Open': price_info.get('open', price_info.get('lastPrice')),
                'High': price_info.get('intraDayHighLow', {}).get('max', price_info.get('lastPrice')),
                'Low': price_info.get('intraDayHighLow', {}).get('min', price_info.get('lastPrice')),
                'Close': price_info.get('lastPrice'),
                'Adj Close': price_info.get('lastPrice'),
                'Volume': data.get('preOpenMarket', {}).get('totalTradedVolume', 0)
            }])
            
            df['Date'] = pd.to_datetime(df['Date'])
            
            logger.info(f"✓ NSE: Fetched data for {symbol} - ₹{price_info.get('lastPrice')}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"NSE request failed for {symbol}: {str(e)}")
            raise NoDataException(f"NSE fetch failed: {str(e)}")
        except Exception as e:
            logger.error(f"NSE error for {symbol}: {str(e)}")
            raise NoDataException(f"NSE error: {str(e)}")
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote data or None
        """
        try:
            df = self.fetch_stock_data(symbol)
            if not df.empty:
                return {
                    'symbol': symbol,
                    'lastPrice': df['Close'].iloc[0],
                    'open': df['Open'].iloc[0],
                    'high': df['High'].iloc[0],
                    'low': df['Low'].iloc[0],
                    'volume': df['Volume'].iloc[0],
                    'date': df['Date'].iloc[0].strftime('%Y-%m-%d')
                }
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None
