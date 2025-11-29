"""YFinance data fetcher for Indian stock markets."""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from retry import retry

from src.utils.logger import get_logger
from src.utils.exceptions import DataIngestionException, InvalidSymbolException, NoDataException

logger = get_logger(__name__)


class YFinanceFetcher:
    """Fetch stock data from YFinance API."""

    def __init__(self):
        """Initialize the fetcher."""
        self.logger = logger

    def _ensure_nse_suffix(self, symbol: str) -> str:
        """
        Ensure stock symbol has .NS suffix for NSE stocks.

        Args:
            symbol: Stock symbol

        Returns:
            Symbol with .NS suffix if not present
        """
        symbol = symbol.upper().strip()
        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
            symbol = f"{symbol}.NS"
        return symbol

    @retry(exceptions=Exception, tries=3, delay=2, backoff=2, logger=logger)
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data from YFinance.

        Args:
            symbol: Stock symbol (will auto-append .NS for NSE)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data

        Raises:
            InvalidSymbolException: If symbol is invalid
            NoDataException: If no data is available
            DataIngestionException: For other errors
        """
        try:
            # Ensure proper symbol format
            symbol = self._ensure_nse_suffix(symbol)
            self.logger.info(f"Fetching data for {symbol}")

            # Create ticker object
            ticker = yf.Ticker(symbol)

            # Fetch data
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            # Validate data
            if df.empty:
                raise NoDataException(f"No data available for symbol: {symbol}")

            # Reset index to make date a column
            df = df.reset_index()

            # Rename columns for consistency
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Add symbol column
            df['symbol'] = symbol

            # Select relevant columns
            columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            df = df[columns]

            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            self.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except NoDataException:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            if "No data found" in str(e) or "Invalid symbol" in str(e):
                raise InvalidSymbolException(f"Invalid or delisted symbol: {symbol}")
            raise DataIngestionException(f"Failed to fetch data: {str(e)}")

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            period: Data period

        Returns:
            Combined DataFrame for all symbols
        """
        dfs = []
        for symbol in symbols:
            try:
                df = self.fetch_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    period=period
                )
                dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Skipping {symbol}: {str(e)}")
                continue

        if not dfs:
            raise NoDataException("No data fetched for any symbol")

        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Fetched data for {len(dfs)} symbols")
        return combined_df

    def get_latest_price(self, symbol: str) -> dict:
        """
        Get the latest price and basic info for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with latest price information
        """
        try:
            symbol = self._ensure_nse_suffix(symbol)
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice'),
                'previous_close': info.get('previousClose'),
                'open': info.get('open'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            raise DataIngestionException(f"Failed to get latest price: {str(e)}")

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists.

        Args:
            symbol: Stock symbol

        Returns:
            True if valid, False otherwise
        """
        try:
            symbol = self._ensure_nse_suffix(symbol)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return 'regularMarketPrice' in info or 'currentPrice' in info
        except:
            return False
