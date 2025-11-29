"""CSV data handler for manual uploads."""
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationException, DataIngestionException

logger = get_logger(__name__)


class CSVHandler:
    """Handle CSV file uploads and validation."""

    REQUIRED_COLUMNS = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']

    def __init__(self):
        """Initialize CSV handler."""
        self.logger = logger

    def validate_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate CSV DataFrame structure and data.

        Args:
            df: Input DataFrame

        Returns:
            Validated and cleaned DataFrame

        Raises:
            DataValidationException: If validation fails
        """
        try:
            # Create a copy
            df = df.copy()

            # Normalize column names to lowercase
            df.columns = df.columns.str.lower().str.strip()
            
            # Map common column name variations
            column_mappings = {
                'date': ['date', 'datetime', 'timestamp', 'time'],
                'symbol': ['symbol', 'ticker', 'stock', 'name'],
                'open': ['open', 'opening_price', 'opening'],
                'high': ['high', 'highest', 'high_price'],
                'low': ['low', 'lowest', 'low_price'],
                'close': ['close', 'closing_price', 'closing', 'last'],
                'volume': ['volume', 'vol', 'quantity', 'traded_quantity']
            }
            
            # Apply mappings
            for standard_col, variations in column_mappings.items():
                for var in variations:
                    if var in df.columns and standard_col not in df.columns:
                        df.rename(columns={var: standard_col}, inplace=True)
                        self.logger.debug(f"Renamed column '{var}' to '{standard_col}'")
                        break
            
            # If symbol not in columns but only one stock, add it from filename
            if 'symbol' not in df.columns:
                df['symbol'] = 'UNKNOWN'
                self.logger.warning("Symbol column not found, using 'UNKNOWN'")
            
            # Check required columns after mapping
            missing_columns = set(self.REQUIRED_COLUMNS) - set(df.columns)
            if missing_columns:
                self.logger.error(f"Available columns: {list(df.columns)}")
                raise DataValidationException(
                    f"Missing required columns: {missing_columns}"
                )

            # Validate symbol column
            if df['symbol'].isnull().any():
                raise DataValidationException("Symbol column contains null values")

            # Convert date column
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                raise DataValidationException(f"Invalid date format: {str(e)}")

            # Validate numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    raise DataValidationException(f"Invalid {col} values: {str(e)}")

            # Check for null values in numeric columns
            null_counts = df[numeric_columns].isnull().sum()
            if null_counts.any():
                self.logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

            # Validate price relationships
            invalid_prices = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            
            if invalid_prices.any():
                invalid_count = invalid_prices.sum()
                self.logger.warning(f"Found {invalid_count} rows with invalid price relationships")
                df = df[~invalid_prices]

            # Validate volume
            if (df['volume'] < 0).any():
                raise DataValidationException("Volume cannot be negative")

            # Sort by symbol and date
            df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

            # Remove duplicates
            initial_count = len(df)
            df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')
            duplicates_removed = initial_count - len(df)
            if duplicates_removed > 0:
                self.logger.info(f"Removed {duplicates_removed} duplicate records")

            self.logger.info(f"Validation successful: {len(df)} records")
            return df

        except DataValidationException:
            raise
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            raise DataValidationException(f"CSV validation failed: {str(e)}")

    def read_csv(self, file_path: str) -> pd.DataFrame:
        """
        Read and validate CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            Validated DataFrame

        Raises:
            DataIngestionException: If file reading fails
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise DataIngestionException(f"File not found: {file_path}")

            # Read CSV
            df = pd.read_csv(file_path)
            self.logger.info(f"Read {len(df)} records from {file_path}")

            # Validate
            df = self.validate_csv(df)
            return df

        except DataValidationException:
            raise
        except Exception as e:
            self.logger.error(f"Error reading CSV: {str(e)}")
            raise DataIngestionException(f"Failed to read CSV: {str(e)}")

    def read_csv_from_bytes(self, file_bytes: bytes) -> pd.DataFrame:
        """
        Read CSV from bytes (for API uploads).

        Args:
            file_bytes: CSV file content as bytes

        Returns:
            Validated DataFrame
        """
        try:
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_bytes))
            self.logger.info(f"Read {len(df)} records from uploaded file")
            df = self.validate_csv(df)
            return df
        except Exception as e:
            self.logger.error(f"Error reading CSV from bytes: {str(e)}")
            raise DataIngestionException(f"Failed to parse CSV: {str(e)}")

    def merge_with_existing(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge new data with existing data, handling duplicates.

        Args:
            new_df: New DataFrame
            existing_df: Existing DataFrame

        Returns:
            Merged DataFrame with duplicates removed
        """
        try:
            # Combine DataFrames
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Remove duplicates, keeping the most recent entry
            combined_df = combined_df.drop_duplicates(
                subset=['symbol', 'date'],
                keep='last'
            )

            # Sort
            combined_df = combined_df.sort_values(['symbol', 'date']).reset_index(drop=True)

            self.logger.info(
                f"Merged data: {len(existing_df)} existing + {len(new_df)} new "
                f"= {len(combined_df)} total records"
            )

            return combined_df

        except Exception as e:
            self.logger.error(f"Error merging data: {str(e)}")
            raise DataIngestionException(f"Failed to merge data: {str(e)}")

    def save_csv(self, df: pd.DataFrame, file_path: str) -> str:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            file_path: Output file path

        Returns:
            Path to saved file
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(file_path, index=False)
            self.logger.info(f"Saved {len(df)} records to {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving CSV: {str(e)}")
            raise DataIngestionException(f"Failed to save CSV: {str(e)}")

    def get_symbols_from_csv(self, df: pd.DataFrame) -> List[str]:
        """
        Get unique symbols from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            List of unique symbols
        """
        return sorted(df['symbol'].unique().tolist())

    def get_date_range(self, df: pd.DataFrame) -> tuple:
        """
        Get date range from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (min_date, max_date)
        """
        return df['date'].min(), df['date'].max()
