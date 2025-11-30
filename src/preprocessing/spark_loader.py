"""
Hybrid preprocessing: PySpark for data loading/cleaning, then convert to pandas for ML
Integrates seamlessly with existing LightGBM and other models
"""
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pathlib import Path
import pandas as pd
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class SparkPreprocessor:
    """
    Uses PySpark for distributed preprocessing, outputs pandas for ML models
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize preprocessor with optional Spark session
        """
        if spark is None:
            # Create lightweight Spark session for local processing
            # Disable Arrow optimization to avoid Java compatibility issues
            self.spark = SparkSession.builder \
                .appName("StockPreprocessing") \
                .master("local[*]") \
                .config("spark.driver.memory", "2g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
            self.spark.sparkContext.setLogLevel("WARN")
            self.owns_spark = True
        else:
            self.spark = spark
            self.owns_spark = False
    
    def load_and_clean_stocks(
        self, 
        dataset_dir: str = "dataset",
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load and clean stock data using PySpark, return pandas DataFrame
        
        Args:
            dataset_dir: Directory with CSV files
            symbols: Optional list of symbols to load (None = all)
            
        Returns:
            Cleaned pandas DataFrame ready for ML models
        """
        logger.info(f"Loading stocks from {dataset_dir} using PySpark...")
        
        dataset_path = Path(dataset_dir)
        all_spark_dfs = []
        
        # Load CSV files with PySpark
        for csv_file in dataset_path.glob("*.csv"):
            if csv_file.stem in ["stock_metadata", "NIFTY50_all"]:
                continue
            
            symbol = csv_file.stem
            if symbols and symbol not in symbols:
                continue
            
            # Load with PySpark
            df_spark = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(str(csv_file))
            
            # Add symbol column
            df_spark = df_spark.withColumn("symbol", F.lit(symbol))
            all_spark_dfs.append(df_spark)
        
        # Union all DataFrames
        if not all_spark_dfs:
            raise ValueError(f"No stock files found in {dataset_dir}")
        
        combined_spark = all_spark_dfs[0]
        for df in all_spark_dfs[1:]:
            combined_spark = combined_spark.union(df)
        
        # Clean data using Spark
        combined_spark = self._clean_spark_df(combined_spark)
        
        # Convert to pandas with Arrow optimization
        pandas_df = combined_spark.toPandas()
        
        logger.info(f"Loaded {len(pandas_df):,} records from {len(all_spark_dfs)} stocks")
        return pandas_df
    
    def _clean_spark_df(self, df: SparkDataFrame) -> SparkDataFrame:
        """
        Clean DataFrame using Spark operations
        """
        # Standardize column names
        for col in df.columns:
            df = df.withColumnRenamed(col, col.lower().strip())
        
        # Remove duplicates
        df = df.dropDuplicates()
        
        # Drop nulls in critical columns
        df = df.dropna(subset=["date", "close"])
        
        # Convert date
        if "date" in df.columns:
            df = df.withColumn("date", F.to_timestamp("date"))
        
        # Ensure numeric types
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df = df.withColumn(col, F.col(col).cast(DoubleType()))
        
        # Filter invalid prices
        df = df.filter(F.col("close") > 0)
        
        # Sort by symbol and date
        df = df.orderBy("symbol", "date")
        
        return df
    
    def load_single_stock(self, csv_path: str, symbol: str) -> pd.DataFrame:
        """
        Load single stock CSV using PySpark, return clean pandas DataFrame
        
        Args:
            csv_path: Path to CSV file
            symbol: Stock symbol
            
        Returns:
            Cleaned pandas DataFrame
        """
        # Load with PySpark
        df_spark = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(csv_path)
        
        df_spark = df_spark.withColumn("symbol", F.lit(symbol))
        df_spark = self._clean_spark_df(df_spark)
        
        # Convert to pandas
        return df_spark.toPandas()
    
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        output_path: str,
        format: str = "parquet"
    ):
        """
        Save processed data using Spark for efficient storage
        
        Args:
            df: Pandas DataFrame to save
            output_path: Output path
            format: Format (parquet, csv)
        """
        # Convert pandas to Spark
        df_spark = self.spark.createDataFrame(df)
        
        if format == "parquet":
            df_spark.write.mode("overwrite").parquet(output_path)
        else:
            df_spark.write.mode("overwrite") \
                .option("header", "true") \
                .csv(output_path)
        
        logger.info(f"Saved to {output_path}")
    
    def __del__(self):
        """Stop Spark session if we own it"""
        if hasattr(self, 'owns_spark') and self.owns_spark and hasattr(self, 'spark'):
            self.spark.stop()
