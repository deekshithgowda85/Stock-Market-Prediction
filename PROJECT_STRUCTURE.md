# 📁 Project Structure Documentation

## Overview

This document explains the structure and purpose of each folder in the Stock Market Prediction project.

---

## 📂 Root Directory Structure

```
Stock-prediction/
├── .github/           # GitHub Actions CI/CD workflows
├── dataset/           # Stock market CSV data (NIFTY 50)
├── frontend/          # Next.js React dashboard
├── models/            # Trained ML models and training scripts
├── src/               # Python backend source code
├── .gitignore         # Git ignore rules
├── LICENSE            # MIT License
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
└── run.py            # Backend server launcher
```

---

## 📦 Detailed Folder Breakdown

### 1. `.github/workflows/` - CI/CD Pipelines

**Purpose**: Automated testing, building, and deployment workflows

**Contents**:

- `ci-cd.yml` - Main CI/CD pipeline with:
  - Automated testing on push/PR
  - Docker image building and pushing
  - Security vulnerability scanning
  - Code quality checks

**Usage**: Automatically triggered on git push to main/develop branches

---

### 2. `dataset/` - Stock Market Data

**Purpose**: Historical stock price data for training and analysis

**Contents**:

- **52 CSV files** - NIFTY 50 stocks (RELIANCE.csv, INFY.csv, etc.)
- `stock_metadata.csv` - Stock information and metadata
- `NIFTY50_all.csv` - Combined index data

**Data Format**:

```csv
date,open,high,low,close,volume
2020-01-01,1500.00,1520.00,1490.00,1510.00,1000000
```

**Size**: 51.84 MB total, ~235,000 records across all stocks

**Data Range**: Historical data from 2000-2021 (varies by stock)

---

### 3. `frontend/` - Web Dashboard (Next.js + React)

**Purpose**: User interface for stock analysis and predictions

**Structure**:

```
frontend/
├── app/                    # Next.js 14 App Router
│   ├── page.tsx           # Main dashboard page (single stock)
│   ├── multi-stock/       # Market-wide analysis page
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── StockSelector.tsx  # Stock picker dropdown
│   ├── StockChart.tsx     # Price chart visualization
│   ├── StockAnalysis.tsx  # Technical indicators display
│   ├── PredictionPanel.tsx # Prediction results
│   └── PreprocessingLogs.tsx # Data pipeline logs
├── public/                # Static assets
├── package.json           # Node.js dependencies
└── tsconfig.json         # TypeScript configuration
```

**Key Technologies**:

- **Framework**: Next.js 14 (React 18)
- **Styling**: Tailwind CSS
- **Charts**: Recharts library
- **API Calls**: Native fetch with localhost:8000

**Pages**:

1. **Main Dashboard** (`/`) - Single stock analysis & predictions
2. **Multi-Stock Analysis** (`/multi-stock`) - Market overview with sentiment analysis

---

### 4. `models/` - Machine Learning Models

**Purpose**: Trained models and training scripts

**Contents**:

- `multi_stock_lightgbm/` - Trained LightGBM model files
  - `lightgbm_model.txt` - Model weights (trained on 232K samples)
  - `model_metadata.json` - Training metrics and feature info
  - `scaler.pkl` - Feature scaler for normalization
- `train_multi_stock_lightgbm.py` - Training script with PySpark

**Model Details**:

- **Type**: LightGBM (Gradient Boosting)
- **Training Data**: 232,742 samples from 49 stocks
- **Features**: 32 technical indicators
- **Performance**: RMSE: 191.59, MAE: 31.97
- **Training Time**: ~3 seconds (with PySpark)

---

### 5. `src/` - Python Backend Source Code

**Purpose**: Core application logic and APIs

#### 5.1 `src/api/` - FastAPI REST Endpoints

**Purpose**: HTTP API for frontend communication

**Files**:

- `main.py` - FastAPI application entry point
- `endpoints.py` - API route handlers
  - `/data/{symbol}` - Get stock data (PySpark)
  - `/analyze/{symbol}` - Stock analysis (PySpark)
  - `/predict` - Price predictions (PySpark + ML)
  - `/predict-lightgbm` - LightGBM predictions
  - `/stocks` - List available stocks
- `auth.py` - Authentication middleware (if enabled)
- `middleware.py` - CORS, rate limiting

**Tech Stack**: FastAPI, Uvicorn, Pydantic

#### 5.2 `src/config/` - Configuration Management

**Purpose**: Application settings and environment variables

**Files**:

- `settings.py` - Configuration class with defaults
  - API settings (host, port, CORS)
  - Data paths (dataset, models)
  - Model parameters (train/test split, etc.)

#### 5.3 `src/ingestion/` - Data Loading & Fetching

**Purpose**: Load data from CSV and external APIs

**Files**:

- `csv_handler.py` - CSV validation and parsing
- `yfinance_fetcher.py` - Live data from Yahoo Finance
- `smart_loader.py` - Intelligent data loading (CSV + YFinance)
- `s3_utils.py` - AWS S3 integration (optional)

**Features**:

- Automatic CSV validation
- Smart fallback (CSV → YFinance)
- Data freshness checking
- Automatic updates for stale data

#### 5.4 `src/models/` - Machine Learning Models

**Purpose**: ML model implementations and prediction logic

**Files**:

- `arima_model.py` - ARIMA time series model
- `prophet_model.py` - Facebook Prophet model
- `lstm_model.py` - LSTM neural network (deprecated)
- `multi_stock_lightgbm.py` - **LightGBM model** (primary)
- `model_selector.py` - Auto model selection

**Primary Model**: `MultiStockLightGBM`

- Trained on all stocks simultaneously
- 32 technical features (MAs, RSI, volatility, lags)
- Iterative multi-day predictions
- Fast inference (<100ms per prediction)

#### 5.5 `src/preprocessing/` - Data Preprocessing (PySpark)

**Purpose**: Distributed data loading and cleaning

**Files**:

- `spark_loader.py` - **SparkPreprocessor class**
  - `load_and_clean_stocks()` - Load all stocks with PySpark
  - `load_single_stock()` - Load one stock distributedly
  - `_clean_spark_df()` - Dedupe, null handling, type validation
  - Automatic Spark session management
  - Falls back to pandas if PySpark fails

**Key Features**:

- Distributed CSV loading (2-3x faster for large data)
- Automatic data cleaning (duplicates, nulls, invalid prices)
- Spark → Pandas conversion (optimized without Arrow)
- Memory efficient for 200K+ records

#### 5.6 `src/processing/` - Feature Engineering

**Purpose**: Create technical indicators and features

**Files**:

- `etl_pipeline.py` - ETL workflow orchestration
- `feature_engineering.py` - Feature creation functions
- `technical_indicators.py` - RSI, MACD, Bollinger Bands
- `preprocessor.py` - Full preprocessing pipeline

**Generated Features** (35 total):

- Moving averages (MA 5, 10, 20, 50, 200)
- Exponential MAs (EMA 12, 26)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Daily returns, volatility
- Volume indicators

#### 5.7 `src/utils/` - Utility Functions

**Purpose**: Helper functions and shared utilities

**Files**:

- `logger.py` - Structured logging setup
- `config.py` - Config loading helpers
- `exceptions.py` - Custom exception classes

---

## 🔄 Data Flow Architecture

```
┌─────────────┐
│   CSV Files │ (dataset/)
│  235K rows  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  SparkPreprocessor│ (PySpark)
│ • Load & Clean    │
│ • Deduplicate     │
│ • Type Validation │
└──────┬───────────┘
       │
       ▼ toPandas()
┌──────────────────┐
│  Feature Eng.    │ (Pandas)
│ • 32 Features    │
│ • Technical Ind. │
└──────┬───────────┘
       │
       ├──────────┐
       ▼          ▼
┌──────────┐  ┌──────────┐
│ LightGBM │  │  ARIMA   │
│ Prophet  │  │  Models  │
└──────┬───┘  └─────┬────┘
       │            │
       └──────┬─────┘
              ▼
      ┌──────────────┐
      │  FastAPI     │
      │  Endpoints   │
      └──────┬───────┘
             │ HTTP/JSON
             ▼
      ┌──────────────┐
      │   Next.js    │
      │  Dashboard   │
      └──────────────┘
```

---

## 🚀 Key Workflows

### 1. Training Workflow

```bash
python models/train_multi_stock_lightgbm.py
```

1. PySpark loads 235K records from dataset/
2. Cleans and validates data distributedly
3. Converts to Pandas for feature engineering
4. Trains LightGBM on 32 features
5. Saves model to models/multi_stock_lightgbm/

### 2. Prediction Workflow (API)

```bash
python run.py  # Start backend on port 8000
```

1. Frontend sends POST to `/api/v1/predict-lightgbm`
2. Backend uses PySpark to load stock CSV
3. Creates 32 features in pandas
4. Loads trained LightGBM model
5. Generates multi-day predictions
6. Returns JSON to frontend

### 3. Dashboard Workflow

```bash
cd frontend && npm run dev  # Port 3000
```

1. User selects stock → calls `/analyze/{symbol}` (PySpark)
2. Dashboard shows analysis + chart
3. User clicks "Predict" → calls `/predict` (PySpark + ML)
4. Shows 30-day forecast with confidence

---

## 📊 Technology Stack Summary

### Backend

- **Language**: Python 3.12
- **Framework**: FastAPI
- **Big Data**: PySpark 3.5.0
- **ML**: LightGBM 4.6.0, Prophet, ARIMA
- **Data**: Pandas, NumPy
- **Server**: Uvicorn ASGI

### Frontend

- **Framework**: Next.js 14 (React 18)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Build**: Turbopack

### Infrastructure

- **CI/CD**: GitHub Actions
- **Container**: Docker (optional)
- **Version Control**: Git + GitHub

---

## 📈 Performance Metrics

- **Training Time**: 3 seconds (232K samples with PySpark)
- **Prediction Time**: <100ms per stock
- **Model RMSE**: 191.59
- **Model MAE**: 31.97
- **Dataset Size**: 51.84 MB (52 stocks)
- **API Response**: 200-500ms (with PySpark loading)
- **PySpark Speedup**: 2-3x faster than pandas for large data

---

## 🔐 Security Features

- Input validation on all endpoints
- CORS configuration for frontend
- Rate limiting (optional)
- Error handling and logging
- No sensitive data in repo

---

## 📝 Development Guidelines

1. **Adding New Models**: Create class in `src/models/`, inherit base interfaces
2. **New Endpoints**: Add to `src/api/endpoints.py`, use PySpark for data loading
3. **Frontend Components**: Add to `frontend/components/`, use TypeScript
4. **Training Scripts**: Place in `models/`, use PySpark for data loading
5. **Configuration**: Update `src/config/settings.py`

---

## 🎯 Quick Commands

```bash
# Backend
python run.py                              # Start API server

# Training
python models/train_multi_stock_lightgbm.py  # Train model

# Frontend
cd frontend && npm install && npm run dev  # Start dashboard

# Testing
python test_pyspark_api.py                 # Test PySpark integration
```

---

**Last Updated**: November 30, 2025
**Project**: Stock Market Prediction with PySpark & LightGBM
**Repository**: https://github.com/deekshithgowda85/Stock-Market-Prediction
