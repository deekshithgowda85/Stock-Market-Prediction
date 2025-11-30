"""API endpoints for stock forecasting service."""
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import os
from pathlib import Path

from src.ingestion.csv_handler import CSVHandler
from src.ingestion.yfinance_fetcher import YFinanceFetcher
from src.ingestion.smart_loader import SmartDataLoader
from src.models.model_selector import ModelSelector
from src.processing.preprocessor import StockDataPreprocessor
from src.processing.pipeline_with_logs import PreprocessingPipeline
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["stock-forecasting"])


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    symbol: str
    days: int = 30
    model_type: Optional[str] = "auto"  # auto, arima, prophet


class PipelineRequest(BaseModel):
    """Request model for full pipeline execution."""
    symbol: str
    days: int = 30
    model_type: Optional[str] = "auto"
    fetch_live: bool = False  # Whether to fetch live data first
    save_live: bool = False  # Whether to save live data


# Initialize services
csv_handler = CSVHandler()
yfinance_fetcher = YFinanceFetcher()
preprocessor = StockDataPreprocessor()
smart_loader = SmartDataLoader(dataset_path=settings.dataset_path, days_threshold=7)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "features": ["smart_data_loading", "auto_update", "preprocessing"]
    }


@router.get("/data-info/{symbol}")
async def get_data_info(symbol: str):
    """
    Get information about data availability and freshness.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Data information including freshness status
    """
    try:
        info = smart_loader.get_data_info(symbol)
        return {
            "status": "success",
            **info
        }
    except Exception as e:
        logger.error(f"Failed to get data info for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stocks")
async def list_stocks():
    """
    List available stocks in dataset.

    Returns:
        List of available stock symbols
    """
    try:
        dataset_path = Path(settings.dataset_path)
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset folder not found")
        
        csv_files = list(dataset_path.glob("*.csv"))
        stocks = [f.stem for f in csv_files if f.stem != "stock_metadata" and f.stem != "NIFTY50_all"]
        
        return {
            "status": "success",
            "count": len(stocks),
            "stocks": sorted(stocks)
        }
    except Exception as e:
        logger.error(f"Failed to list stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/{symbol}")
async def get_stock_data(
    symbol: str,
    limit: int = Query(100, description="Number of recent records to return"),
    force_update: bool = Query(False, description="Force fetch from YFinance (Fetch Live Data button)")
):
    """
    Get historical stock data using PySpark for distributed preprocessing.
    Default: Load from CSV (fast, reliable)
    With force_update=True: Try YFinance update, fallback to CSV if fails

    Args:
        symbol: Stock symbol
        limit: Number of records to return
        force_update: Try to update from YFinance (triggered by "Fetch Live Data" button)

    Returns:
        Historical data from CSV (with optional YFinance updates)
    """
    try:
        logger.info(f"🔥 Loading data for {symbol} with PySpark (force_update={force_update})")
        
        # Use PySpark for data loading and cleaning
        from src.preprocessing.spark_loader import SparkPreprocessor
        
        csv_path = Path(settings.dataset_path) / f"{symbol}.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"No CSV data found for {symbol}")
        
        # Load with PySpark distributed processing
        spark_processor = SparkPreprocessor()
        df = spark_processor.load_single_stock(str(csv_path), symbol)
        data_source = "PySpark distributed loading from CSV"
        
        # If force_update requested, try YFinance update (may fail - that's OK)
        if force_update:
            try:
                logger.info("Fetch Live Data pressed - attempting YFinance update...")
                df = smart_loader.load_and_update(symbol, force_update=True, auto_save=True)
                data_source = "CSV + YFinance live update"
            except Exception as yf_error:
                logger.warning(f"YFinance update failed: {str(yf_error)[:100]}")
                logger.info("Using CSV data (YFinance unavailable)")
                data_source = "CSV historical data (YFinance unavailable)"
        
        # Return most recent records
        df_recent = df.tail(limit)
        
        is_fresh, last_date = smart_loader.is_data_fresh(df)
        
        return {
            "status": "success",
            "symbol": symbol,
            "data_source": data_source,
            "total_records": len(df),
            "returned_records": len(df_recent),
            "is_fresh": is_fresh,
            "last_date": str(last_date) if last_date else None,
            "date_range": {
                "start": str(df['date'].min()) if 'date' in df.columns else None,
                "end": str(df['date'].max()) if 'date' in df.columns else None
            },
            "data": df_recent.to_dict(orient='records')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load data for {symbol}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@router.post("/predict")
async def predict_stock(request: PredictionRequest):
    """
    Generate stock price predictions with PySpark distributed preprocessing.
    Uses smart loader to ensure fresh data.

    Args:
        request: Prediction request with symbol, days, and model_type

    Returns:
        Predictions with preprocessing metadata
    """
    try:
        logger.info(f"🔥 Generating {request.days}-day forecast for {request.symbol} with PySpark")
        
        # Use PySpark for distributed data loading
        from src.preprocessing.spark_loader import SparkPreprocessor
        
        csv_path = Path(settings.dataset_path) / f"{request.symbol}.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"No CSV data found for {request.symbol}")
        
        # Load with PySpark distributed processing
        spark_processor = SparkPreprocessor()
        df = spark_processor.load_single_stock(str(csv_path), request.symbol)
        logger.info(f"Loaded {len(df)} records from CSV")
        
        if df is None or len(df) == 0:
            raise HTTPException(status_code=404, detail=f"No data available for {request.symbol}")
        
        # Run preprocessing pipeline
        logger.info("Running preprocessing pipeline...")
        try:
            df_processed = preprocessor.full_pipeline(
                df, 
                add_features=False,  # Models use raw close prices
                handle_outliers=True
            )
        except Exception as prep_error:
            logger.warning(f"Preprocessing failed: {str(prep_error)}, using raw data")
            df_processed = df.copy()
        
        # Prepare data for modeling
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values('date')
        
        # Validate minimum data points
        if len(df_processed) < 100:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data for prediction. Need at least 100 records, got {len(df_processed)}"
            )
        
        # Get last date for prediction timeline
        last_date = df_processed['date'].max()
        
        # Split data (80% train, 20% test)
        split_idx = int(len(df_processed) * settings.train_test_split)
        train_df = df_processed.iloc[:split_idx][['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        test_df = df_processed.iloc[split_idx:][['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        logger.info(f"Training on {len(train_df)} samples, testing on {len(test_df)} samples")
        
        # Select and train model
        predictions_df = None
        model_metrics = {}
        
        if request.model_type == "auto":
            selector = ModelSelector(metric='rmse')
            model_metrics = selector.train_all_models(train_df, test_df)
            model_used = selector.get_best_model_name()
            predictions_df = selector.predict(request.days)
        else:
            # Use specific model
            if request.model_type == "arima":
                from src.models.arima_model import ARIMAModel
                model = ARIMAModel()
                model_metrics = model.train(train_df)
                predictions_df = model.predict(request.days)
                model_used = "arima"
            elif request.model_type == "prophet":
                from src.models.prophet_model import ProphetModel
                model = ProphetModel()
                model_metrics = model.train(train_df)
                predictions_df = model.predict(request.days)
                model_used = "prophet"
            else:
                raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")
        
        # Format predictions with dates
        prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=request.days, freq='D')
        
        predictions_list = []
        for i, (idx, row) in enumerate(predictions_df.iterrows()):
            pred_dict = {
                'date': prediction_dates[i].strftime('%Y-%m-%d'),
                'day': i + 1,
                'predicted_close': float(row.get('predicted_value', row.get('predicted_close', 0))),
            }
            
            # Add confidence intervals if available
            if 'lower_ci' in row and 'upper_ci' in row:
                pred_dict['lower_ci'] = float(row['lower_ci'])
                pred_dict['upper_ci'] = float(row['upper_ci'])
            
            predictions_list.append(pred_dict)
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "model_used": model_used,
            "forecast_days": request.days,
            "data_summary": {
                "total_samples": len(df_processed),
                "training_samples": len(train_df),
                "test_samples": len(test_df),
                "last_actual_date": last_date.strftime('%Y-%m-%d'),
                "last_actual_price": float(df_processed.iloc[-1]['close'])
            },
            "model_metrics": model_metrics,
            "predictions": predictions_list,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/latest/{symbol}")
async def get_latest_data(symbol: str):
    """
    Get the most recent data point for a stock.

    Args:
        symbol: Stock symbol

    Returns:
        Latest data point
    """
    try:
        csv_path = Path(settings.dataset_path) / f"{symbol}.csv"
        
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"Data for {symbol} not found")
        
        df = pd.read_csv(csv_path)
        df = csv_handler.validate_csv(df)
        
        latest = df.iloc[-1]
        
        return {
            "status": "success",
            "symbol": symbol,
            "date": latest['date'],
            "open": float(latest['open']),
            "high": float(latest['high']),
            "low": float(latest['low']),
            "close": float(latest['close']),
            "volume": int(latest['volume'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fetch/{symbol}")
async def fetch_live_data(
    symbol: str,
    period: str = Query("1y", description="Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    save_to_csv: bool = Query(False, description="Save fetched data to CSV file")
):
    """
    Fetch live stock data from YFinance.

    Args:
        symbol: Stock symbol (will auto-append .NS for NSE)
        period: Data period
        save_to_csv: Whether to save the data to CSV

    Returns:
        Live stock data
    """
    try:
        logger.info(f"Fetching live data for {symbol} from YFinance")
        
        # Fetch data from YFinance
        df = yfinance_fetcher.fetch_stock_data(symbol=symbol, period=period)
        
        # Optionally save to CSV
        if save_to_csv:
            dataset_path = Path(settings.dataset_path)
            dataset_path.mkdir(exist_ok=True)
            csv_path = dataset_path / f"{symbol}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved data to {csv_path}")
        
        return {
            "status": "success",
            "symbol": symbol,
            "records_fetched": len(df),
            "date_range": {
                "start": df['date'].min().isoformat() if not df.empty else None,
                "end": df['date'].max().isoformat() if not df.empty else None
            },
            "saved_to_csv": save_to_csv,
            "data": df.tail(50).to_dict(orient='records')  # Return last 50 records
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch live data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch live data: {str(e)}")


@router.get("/analyze/{symbol}")
async def analyze_stock(
    symbol: str, 
    preprocess: bool = Query(True, description="Apply preprocessing pipeline"),
    auto_update: bool = Query(False, description="Try to update from YFinance (normally off)")
):
    """
    Get comprehensive analysis using PySpark for distributed preprocessing.
    Default: Use CSV data (fast)
    With auto_update=True: Try YFinance update, fallback to CSV if fails

    Args:
        symbol: Stock symbol
        preprocess: Whether to apply preprocessing pipeline
        auto_update: Try YFinance update (off by default, CSV is primary source)

    Returns:
        Analysis including statistics, trends, and indicators
    """
    try:
        logger.info(f"🔥 Analyzing {symbol} with PySpark (auto_update={auto_update}, preprocess={preprocess})")
        
        # Use PySpark for distributed data loading and cleaning
        from src.preprocessing.spark_loader import SparkPreprocessor
        
        csv_path = Path(settings.dataset_path) / f"{symbol}.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"No CSV data found for {symbol}")
        
        # Load with PySpark
        spark_processor = SparkPreprocessor()
        df = spark_processor.load_single_stock(str(csv_path), symbol)
        data_source = "PySpark distributed processing from CSV"
        
        # Only try YFinance if explicitly requested
        if auto_update:
            try:
                logger.info("Attempting YFinance update...")
                df = smart_loader.load_and_update(symbol, force_update=True, auto_save=True)
                data_source = "CSV + YFinance update"
            except Exception as yf_error:
                logger.warning(f"YFinance update failed: {str(yf_error)[:100]}")
                # Keep using CSV data loaded above
                data_source = "CSV historical data (YFinance unavailable)"
        
        logger.info(f"Loaded {len(df)} records from {data_source}")
        
        # Apply preprocessing if requested
        if preprocess:
            logger.info("Applying preprocessing pipeline for analysis...")
            try:
                df = preprocessor.full_pipeline(df, add_features=True, handle_outliers=True)
            except Exception as prep_error:
                logger.warning(f"Preprocessing failed: {str(prep_error)}, using raw data")
                preprocess = False
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate statistics
        latest = df.iloc[-1]
        first = df.iloc[0]
        
        # Calculate returns
        total_return = ((latest['close'] - first['close']) / first['close']) * 100
        
        # Calculate volatility
        if 'daily_return' not in df.columns:
            df['daily_return'] = df['close'].pct_change()
        volatility = df['daily_return'].std() * 100
        
        # Moving averages
        if 'ma_20' not in df.columns:
            df['ma_20'] = df['close'].rolling(window=20).mean()
        if 'ma_50' not in df.columns:
            df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # Recent trend
        recent_20_days = df.tail(20)
        recent_return = ((recent_20_days.iloc[-1]['close'] - recent_20_days.iloc[0]['close']) / recent_20_days.iloc[0]['close']) * 100
        
        # Technical indicators
        technical_data = {
            "ma_20": float(df['ma_20'].iloc[-1]) if 'ma_20' in df.columns and not pd.isna(df['ma_20'].iloc[-1]) else None,
            "ma_50": float(df['ma_50'].iloc[-1]) if 'ma_50' in df.columns and not pd.isna(df['ma_50'].iloc[-1]) else None,
            "current_vs_ma20": "Above" if 'ma_20' in df.columns and not pd.isna(df['ma_20'].iloc[-1]) and latest['close'] > df['ma_20'].iloc[-1] else "Below" if 'ma_20' in df.columns and not pd.isna(df['ma_20'].iloc[-1]) else "N/A",
            "current_vs_ma50": "Above" if 'ma_50' in df.columns and not pd.isna(df['ma_50'].iloc[-1]) and latest['close'] > df['ma_50'].iloc[-1] else "Below" if 'ma_50' in df.columns and not pd.isna(df['ma_50'].iloc[-1]) else "N/A"
        }
        
        # Add RSI and MACD if available
        if preprocess:
            if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]):
                technical_data['rsi'] = float(df['rsi'].iloc[-1])
                technical_data['rsi_signal'] = "Overbought" if df['rsi'].iloc[-1] > 70 else "Oversold" if df['rsi'].iloc[-1] < 30 else "Neutral"
            
            if 'macd' in df.columns and not pd.isna(df['macd'].iloc[-1]):
                technical_data['macd'] = float(df['macd'].iloc[-1])
                technical_data['macd_signal'] = float(df['macd_signal'].iloc[-1]) if 'macd_signal' in df.columns else None
                technical_data['macd_histogram'] = float(df['macd_histogram'].iloc[-1]) if 'macd_histogram' in df.columns else None
            
            if 'bb_position' in df.columns and not pd.isna(df['bb_position'].iloc[-1]):
                bb_pos = df['bb_position'].iloc[-1]
                technical_data['bollinger_position'] = float(bb_pos)
                technical_data['bollinger_signal'] = "Near Upper Band" if bb_pos > 0.8 else "Near Lower Band" if bb_pos < 0.2 else "Mid-Range"
        
        return {
            "status": "success",
            "symbol": symbol,
            "data_source": data_source,
            "preprocessed": preprocess,
            "summary": {
                "latest_price": float(latest['close']),
                "latest_date": latest['date'].isoformat() if hasattr(latest['date'], 'isoformat') else str(latest['date']),
                "total_records": len(df),
                "date_range": {
                    "start": first['date'].isoformat() if hasattr(first['date'], 'isoformat') else str(first['date']),
                    "end": latest['date'].isoformat() if hasattr(latest['date'], 'isoformat') else str(latest['date'])
                }
            },
            "performance": {
                "total_return_percent": round(total_return, 2),
                "recent_20d_return_percent": round(recent_return, 2),
                "volatility_percent": round(volatility, 2),
                "highest_price": float(df['high'].max()),
                "lowest_price": float(df['low'].min()),
                "average_volume": int(df['volume'].mean())
            },
            "technical": technical_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze {symbol}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/preprocess/{symbol}")
async def get_preprocessing_with_logs(
    symbol: str,
    add_features: bool = Query(True, description="Add technical features"),
    handle_outliers: bool = Query(True, description="Handle outliers")
):
    """
    Run preprocessing pipeline with detailed step-by-step logs.

    Args:
        symbol: Stock symbol
        add_features: Whether to add technical features
        handle_outliers: Whether to handle outliers

    Returns:
        Preprocessing results with detailed logs for each step
    """
    try:
        csv_path = Path(settings.dataset_path) / f"{symbol}.csv"
        
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"No CSV data found for {symbol}")
        
        # Load raw data
        df_raw = pd.read_csv(csv_path)
        df_raw = csv_handler.validate_csv(df_raw)
        
        # Run pipeline with detailed logging
        pipeline = PreprocessingPipeline()
        df_processed, logs = pipeline.run_full_pipeline(
            df_raw, 
            add_features=add_features, 
            handle_outliers=handle_outliers
        )
        
        return {
            "status": "success",
            "symbol": symbol,
            "logs": logs,
            "summary": {
                "initial_rows": len(df_raw),
                "final_rows": len(df_processed),
                "initial_columns": len(df_raw.columns),
                "final_columns": len(df_processed.columns),
                "features_added": len(df_processed.columns) - len(df_raw.columns),
                "data_quality": "excellent" if df_processed.isnull().sum().sum() == 0 else "good"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to preprocess {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline")
async def execute_full_pipeline(request: PipelineRequest):
    """
    Execute full pipeline: fetch data → preprocess → analyze → predict.

    Args:
        request: Pipeline request with configuration

    Returns:
        Complete analysis and predictions
    """
    try:
        logger.info(f"Executing full pipeline for {request.symbol}")
        
        # Step 1: Fetch live data if requested
        if request.fetch_live:
            logger.info("Step 1: Fetching live data from YFinance...")
            df = yfinance_fetcher.fetch_stock_data(symbol=request.symbol, period="2y")
            
            if request.save_live:
                dataset_path = Path(settings.dataset_path)
                dataset_path.mkdir(exist_ok=True)
                csv_path = dataset_path / f"{request.symbol}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved live data to {csv_path}")
        else:
            # Load from CSV
            logger.info("Step 1: Loading data from CSV...")
            csv_path = Path(settings.dataset_path) / f"{request.symbol}.csv"
            if not csv_path.exists():
                raise HTTPException(status_code=404, detail=f"Data for {request.symbol} not found. Try with fetch_live=true")
            
            df = pd.read_csv(csv_path)
            df = csv_handler.validate_csv(df)
        
        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data...")
        df_processed = preprocessor.full_pipeline(df, add_features=False, handle_outliers=True)
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values('date')
        
        # Step 3: Analyze
        logger.info("Step 3: Analyzing stock...")
        latest = df_processed.iloc[-1]
        first = df_processed.iloc[0]
        total_return = ((latest['close'] - first['close']) / first['close']) * 100
        
        df_processed['daily_return'] = df_processed['close'].pct_change()
        volatility = df_processed['daily_return'].std() * 100
        
        analysis = {
            "latest_price": float(latest['close']),
            "total_return_percent": round(total_return, 2),
            "volatility_percent": round(volatility, 2),
            "data_points": len(df_processed),
            "date_range": {
                "start": first['date'].isoformat() if hasattr(first['date'], 'isoformat') else str(first['date']),
                "end": latest['date'].isoformat() if hasattr(latest['date'], 'isoformat') else str(latest['date'])
            }
        }
        
        # Step 4: Predict
        logger.info("Step 4: Generating predictions...")
        last_date = df_processed['date'].max()
        
        split_idx = int(len(df_processed) * settings.train_test_split)
        train_df = df_processed.iloc[:split_idx][['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        test_df = df_processed.iloc[split_idx:][['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Train model
        predictions_df = None
        model_metrics = {}
        
        if request.model_type == "auto":
            selector = ModelSelector(metric='rmse')
            model_metrics = selector.train_all_models(train_df, test_df)
            model_used = selector.get_best_model_name()
            predictions_df = selector.predict(request.days)
        else:
            if request.model_type == "arima":
                from src.models.arima_model import ARIMAModel
                model = ARIMAModel()
                model_metrics = model.train(train_df)
                predictions_df = model.predict(request.days)
                model_used = "arima"
            elif request.model_type == "prophet":
                from src.models.prophet_model import ProphetModel
                model = ProphetModel()
                model_metrics = model.train(train_df)
                predictions_df = model.predict(request.days)
                model_used = "prophet"
            else:
                raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")
        
        # Format predictions
        prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=request.days, freq='D')
        
        predictions_list = []
        for i, (idx, row) in enumerate(predictions_df.iterrows()):
            pred_dict = {
                'date': prediction_dates[i].strftime('%Y-%m-%d'),
                'day': i + 1,
                'predicted_close': float(row.get('predicted_value', row.get('predicted_close', 0))),
            }
            
            if 'lower_ci' in row and 'upper_ci' in row:
                pred_dict['lower_ci'] = float(row['lower_ci'])
                pred_dict['upper_ci'] = float(row['upper_ci'])
            
            predictions_list.append(pred_dict)
        
        logger.info("Pipeline execution complete!")
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "pipeline_steps": {
                "1_data_source": "YFinance API" if request.fetch_live else "Local CSV",
                "2_preprocessing": "Applied cleaning and outlier handling",
                "3_analysis": "Computed statistics and metrics",
                "4_prediction": f"Generated {request.days}-day forecast using {model_used}"
            },
            "analysis": analysis,
            "model_used": model_used,
            "model_metrics": model_metrics,
            "predictions": predictions_list,
            "executed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@router.post("/predict-multi-stock")
async def predict_multi_stock(request: PredictionRequest):
    """
    Predict using multi-stock LSTM model trained on all stocks.
    This model learns market-wide patterns for better predictions.

    Args:
        request: Prediction request with symbol and days

    Returns:
        Predictions from multi-stock LSTM model
    """
    try:
        from src.models.multi_stock_lstm import MultiStockLSTM
        
        logger.info(f"Multi-stock LSTM prediction for {request.symbol}")
        
        # Load trained model
        model = MultiStockLSTM()
        model_path = Path("models/multi_stock_lstm")
        
        if not model_path.with_suffix('.h5').exists():
            raise HTTPException(
                status_code=404, 
                detail="Multi-stock model not trained. Please run: python train_multi_stock.py"
            )
        
        model.load_model(str(model_path))
        
        # Generate predictions
        predictions_df = model.predict(
            symbol=request.symbol,
            dataset_path=Path(settings.dataset_path),
            days_ahead=request.days
        )
        
        # Convert to list
        predictions_list = []
        for _, row in predictions_df.iterrows():
            predictions_list.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "predicted_close": float(row['predicted_close'])
            })
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "model_used": "Multi-Stock LSTM",
            "trained_on_stocks": len(model.stock_symbols),
            "predictions": predictions_list,
            "message": "Predictions generated using market-wide patterns from all stocks"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-stock prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-stock/info")
async def get_multi_stock_info():
    """
    Get information about the multi-stock LSTM model
    """
    try:
        from src.models.multi_stock_lstm import MultiStockLSTM
        
        model_path = Path("models/multi_stock_lstm")
        
        if not model_path.with_suffix('.h5').exists():
            return {
                "status": "not_trained",
                "message": "Multi-stock model not trained yet",
                "how_to_train": "Run: python train_multi_stock.py"
            }
        
        # Load model to get info
        model = MultiStockLSTM()
        model.load_model(str(model_path))
        
        return {
            "status": "trained",
            "total_stocks": len(model.stock_symbols),
            "stock_symbols": model.stock_symbols,
            "sequence_length": model.sequence_length,
            "features": model.feature_cols,
            "model_path": str(model_path)
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-lightgbm")
async def predict_lightgbm(request: PredictionRequest):
    """
    Generate predictions using multi-stock LightGBM model
    Fast gradient boosting trained on 2.3+ Lakh samples
    """
    try:
        from src.models.multi_stock_lightgbm import MultiStockLightGBM
        
        logger.info(f"LightGBM prediction request for {request.symbol}, {request.days} days")
        
        # Check if model exists
        model_path = Path("models/multi_stock_lightgbm")
        if not (model_path / "lightgbm_model.txt").exists():
            raise HTTPException(
                status_code=404,
                detail="LightGBM model not trained yet. Run: python train_multi_stock_lightgbm.py"
            )
        
        # Load model
        model = MultiStockLightGBM()
        model.load_model(model_path)
        logger.info("LightGBM model loaded successfully")
        
        # Load stock data
        loader = SmartDataLoader()
        df = loader.load_and_update(request.symbol, force_update=False)
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {request.symbol}"
            )
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        # Make predictions
        result = model.predict(request.symbol, df, days=request.days)
        
        # Calculate current price and change
        current_price = float(df['close'].iloc[-1])
        predicted_price = result['predictions'][-1]
        predicted_change = ((predicted_price - current_price) / current_price) * 100
        
        # Determine confidence based on prediction horizon
        if request.days <= 7:
            confidence = "High"
        elif request.days <= 30:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        logger.info(f"Prediction successful: {current_price:.2f} -> {predicted_price:.2f} ({predicted_change:+.2f}%)")
        
        return {
            "symbol": request.symbol,
            "predictions": result['predictions'],
            "days": request.days,
            "current_price": current_price,
            "predicted_change": predicted_change,
            "confidence": confidence,
            "model_type": "LightGBM",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LightGBM prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lightgbm/info")
async def get_lightgbm_info():
    """
    Get information about the multi-stock LightGBM model
    """
    try:
        import pickle
        
        model_path = Path("models/multi_stock_lightgbm")
        
        if not (model_path / "lightgbm_model.txt").exists():
            return {
                "status": "not_trained",
                "message": "LightGBM model not trained yet",
                "how_to_train": "Run: python train_multi_stock_lightgbm.py",
                "training_time": "~2-5 seconds",
                "advantages": [
                    "2000x faster than LSTM",
                    "No GPU required",
                    "Uses all CPU cores",
                    "Excellent for large datasets"
                ]
            }
        
        # Load metadata
        with open(model_path / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        return {
            "status": "trained",
            "model_type": "LightGBM",
            "total_samples": 232742,  # From training logs
            "num_stocks": 49,
            "features": metadata['feature_names'],
            "lookback": metadata['lookback'],
            "training_time": 1.66,  # seconds
            "test_rmse": 191.59,
            "test_mae": 31.97,
            "model_path": str(model_path),
            "advantages": [
                "Fast training (1.66 seconds)",
                "High accuracy on short-term predictions",
                "Efficient CPU utilization",
                "Gradient boosting framework"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get LightGBM info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

