# 📈 Stock Market Prediction with PySpark & LightGBM

A production-ready Stock Market Forecasting Platform featuring distributed data processing with PySpark, advanced ML models, and an interactive React dashboard.

## ✨ Key Highlights

- **🔥 PySpark Integration**: Distributed data preprocessing for 235K+ records (2-3x faster)
- **🤖 LightGBM Model**: Trained on 49 stocks with 232,742 samples (RMSE: 191.59)
- **⚡ Fast Predictions**: <100ms inference time with 32 technical features
- **📊 Interactive Dashboard**: Next.js React UI with real-time analysis
- **🎯 Multi-Stock Analysis**: Market-wide sentiment and sector performance
- **🔄 Smart Data Loading**: Automatic CSV validation with YFinance fallback
- **📈 Multiple Models**: LightGBM (primary), ARIMA, Prophet with auto-selection

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+ (for frontend)
- Java 8+ (for PySpark)

### 1. Clone Repository

```bash
git clone https://github.com/deekshithgowda85/Stock-Market-Prediction.git
cd Stock-Market-Prediction
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model (Optional - Pre-trained model included)

```bash
python models/train_multi_stock_lightgbm.py
```

**Training Details**:

- Uses PySpark for distributed preprocessing
- Processes 232,742 samples from 49 stocks
- Takes ~3 seconds to train
- Model saved to `models/multi_stock_lightgbm/`

### 4. Start Backend API

```bash
python run.py
```

API runs on `http://localhost:8000`

**Note**: First request may take 40 seconds as PySpark initializes Spark session

### 5. Start Frontend Dashboard (Optional)

```bash
cd frontend
npm install
npm run dev
```

Dashboard runs on `http://localhost:3000`

### 6. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📊 API Endpoints

### Data Operations (PySpark-Powered)

**List Available Stocks**

```http
GET /api/v1/stocks
```

Returns 52 NIFTY 50 stock symbols

**Get Stock Data** (🔥 PySpark)

```http
GET /api/v1/data/{symbol}?limit=100&force_update=false
```

- Uses PySpark for distributed data loading
- Returns last 100 records by default
- Set `force_update=true` to fetch live data from YFinance

**Stock Analysis** (🔥 PySpark)

```http
GET /api/v1/analyze/{symbol}?preprocess=true&auto_update=false
```

- PySpark loads and cleans data
- Returns technical indicators (RSI, MACD, Bollinger Bands)
- Calculates volatility, returns, moving averages

### Prediction Operations

**Generate Predictions** (🔥 PySpark + ML)

```http
POST /api/v1/predict
Content-Type: application/json

{
  "symbol": "RELIANCE",
  "days": 30,
  "model_type": "auto"
}
```

- PySpark preprocesses data
- Model types: `auto`, `arima`, `prophet`
- Returns 30-day forecast with confidence intervals

**LightGBM Predictions** (Primary Model)

```http
POST /api/v1/predict-lightgbm
Content-Type: application/json

{
  "symbol": "RELIANCE",
  "days": 30
}
```

- Fast predictions (<100ms)
- Trained on 232K samples
- 32 technical features
- Best for short-term forecasts (7-30 days)

**Multi-Stock Market Analysis**

```http
POST /api/v1/analyze-market
Content-Type: application/json

{
  "symbols": ["RELIANCE", "INFY", "TCS", "HDFCBANK"],
  "days": 30
}
```

- Bulk predictions for multiple stocks
- Market sentiment analysis
- Sector performance comparison
- Top gainers/losers identification

## 🎯 Key Features

### Backend

- **🔥 PySpark Integration**: Distributed data preprocessing (2-3x faster than pandas)
- **⚡ LightGBM Model**: Primary prediction engine with 191.59 RMSE
- **📈 Multiple Models**: ARIMA, Prophet with automatic model selection
- **🔄 Smart Data Loading**: CSV-first with YFinance fallback for fresh data
- **📊 Technical Analysis**: 35 indicators (RSI, MACD, Bollinger Bands, etc.)
- **🎯 Multi-Stock Support**: Train and predict across 49 stocks simultaneously
- **⏱️ Fast Inference**: <100ms predictions with pre-trained models

### Frontend

- **📱 Interactive Dashboard**: Real-time stock analysis and visualization
- **📊 Advanced Charts**: Recharts with historical data and predictions
- **🌐 Market Overview**: Multi-stock analysis with sentiment indicators
- **📈 Sector Performance**: Compare different market sectors
- **⚡ Real-time Updates**: Live data fetching from YFinance
- **🎨 Modern UI**: Tailwind CSS with responsive design

### DevOps & Infrastructure

- **🔄 CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **🐳 Docker Support**: Containerized deployment (optional)
- **📝 Comprehensive Logging**: Structured logs for debugging and monitoring
- **🔐 API Documentation**: Auto-generated Swagger UI and ReDoc
- **⚠️ Error Handling**: Graceful fallbacks and informative error messages

## 📦 Technology Stack

### Backend

| Category  | Technology  | Version | Purpose                   |
| --------- | ----------- | ------- | ------------------------- |
| Language  | Python      | 3.12    | Core backend              |
| Framework | FastAPI     | Latest  | REST API                  |
| Big Data  | PySpark     | 3.5.0   | Distributed preprocessing |
| ML        | LightGBM    | 4.6.0   | Primary model             |
| ML        | Prophet     | Latest  | Time series forecasting   |
| ML        | Statsmodels | Latest  | ARIMA implementation      |
| Data      | Pandas      | Latest  | Data manipulation         |
| Data      | NumPy       | Latest  | Numerical operations      |
| Server    | Uvicorn     | Latest  | ASGI server               |

### Frontend

| Category  | Technology   | Version | Purpose            |
| --------- | ------------ | ------- | ------------------ |
| Framework | Next.js      | 14      | React framework    |
| Language  | TypeScript   | 5.x     | Type safety        |
| UI        | React        | 18      | Component library  |
| Styling   | Tailwind CSS | 3.x     | Utility-first CSS  |
| Charts    | Recharts     | 2.x     | Data visualization |
| Build     | Turbopack    | Latest  | Fast bundler       |

### Infrastructure

- **Version Control**: Git + GitHub
- **CI/CD**: GitHub Actions
- **Container**: Docker (optional)
- **Package Manager**: pip (Python), npm (Node.js)

## 📁 Project Structure

```
Stock-prediction/
├── dataset/                   # 52 NIFTY 50 stock CSV files (51.84 MB)
├── frontend/                  # Next.js React dashboard
│   ├── app/                  # Next.js 14 App Router
│   │   ├── page.tsx         # Main dashboard
│   │   └── multi-stock/     # Market analysis page
│   └── components/           # React components
├── models/                    # Trained ML models
│   ├── multi_stock_lightgbm/ # LightGBM model files
│   └── train_multi_stock_lightgbm.py # Training script
├── src/                       # Python backend
│   ├── api/                  # FastAPI endpoints
│   ├── config/               # Configuration
│   ├── ingestion/            # Data loading (CSV, YFinance)
│   ├── models/               # ML model implementations
│   ├── preprocessing/        # PySpark data preprocessing
│   ├── processing/           # Feature engineering
│   └── utils/                # Helper functions
├── .github/workflows/        # CI/CD pipelines
├── requirements.txt          # Python dependencies
├── run.py                    # Backend launcher
└── PROJECT_STRUCTURE.md      # Detailed folder documentation
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed folder explanations.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                        │
│  • Dashboard Page (Single Stock Analysis)                    │
│  • Multi-Stock Page (Market Overview)                        │
│  • React Components (Charts, Analysis, Predictions)          │
└────────────────────┬─────────────────────────────────────────┘
                     │ HTTP/JSON API
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                FastAPI Backend (Python 3.12)                  │
│  • /data/{symbol} - PySpark data loading                     │
│  • /analyze/{symbol} - PySpark analysis                      │
│  • /predict - PySpark + ML predictions                       │
│  • /predict-lightgbm - LightGBM predictions                  │
└────────────┬────────────────┬────────────────────────────────┘
             │                │
             ▼                ▼
┌──────────────────┐  ┌─────────────────────┐
│  SparkPreprocessor│  │   ML Models         │
│  (PySpark 3.5)   │  │                     │
│  • Load CSV      │  │  • LightGBM ⭐      │
│  • Clean Data    │  │  • ARIMA           │
│  • Deduplicate   │  │  • Prophet         │
│  • Validate      │  │  • Model Selector  │
└────────┬─────────┘  └──────────┬──────────┘
         │                       │
         ▼ toPandas()           ▼ predict()
┌──────────────────────────────────────────┐
│     Feature Engineering (Pandas)         │
│  • 32 Technical Indicators               │
│  • MAs, RSI, MACD, Bollinger Bands       │
│  • Volatility, Returns, Lag Features     │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Dataset (CSV Files)              │
│  • 52 NIFTY 50 Stocks                    │
│  • 235,192 Total Records                 │
│  • 51.84 MB Data (2000-2021)             │
└──────────────────────────────────────────┘
```

### Data Flow

1.  **Ingestion**: PySpark loads CSV → Distributed cleaning → Pandas DataFrame
2.  **Processing**: Feature engineering creates 32 technical indicators
3.  **Training**: LightGBM trains on 232K samples (one-time, 3 seconds)
4.  **Inference**: Load model → Generate predictions → Return to API
5.  **Display**: Frontend fetches predictions → Renders charts and analysis
    ┌─────────────────┐
    │ YFinance API │
    │ (NSE/BSE) │
    └────────┬────────┘
    │
    ▼
    ┌─────────────────────────────────────────┐
    │ Data Ingestion Layer │
    │ ┌──────────┐ ┌──────────────┐ │
    │ │ YFinance │ │ CSV Upload │ │
    │ │ Fetcher │ │ Handler │ │
    │ └──────────┘ └──────────────┘ │
    └────────┬────────────────────────────────┘
    │
    ▼
    ┌─────────────────────────────────────────┐
    │ AWS S3 Storage │
    │ /raw/ /processed/ /predictions/ │
    └────────┬────────────────────────────────┘
    │
    ▼
    ┌─────────────────────────────────────────┐
    │ PySpark ETL & Feature Eng. │
    │ • Data Cleaning │
    │ • Technical Indicators (RSI, MACD) │
    │ • Rolling Windows │
    └────────┬────────────────────────────────┘
    │
    ▼
    ┌─────────────────────────────────────────┐
    │ ML Model Training │
    │ ┌────────┐ ┌────────┐ ┌─────────┐ │
    │ │ ARIMA │ │ LSTM │ │ Prophet │ │
    │ └────────┘ └────────┘ └─────────┘ │
    │ Auto-selection by RMSE │
    └────────┬────────────────────────────────┘
    │
    ▼
    ┌─────────────────────────────────────────┐
    │ FastAPI Service │
    │ • Prediction Endpoints │
    │ • Authentication │
    │ • Rate Limiting │
    └────────┬────────────────────────────────┘
    │
    ▼
    ┌─────────────────────────────────────────┐
    │ React Dashboard │
    │ • Stock Charts │
    │ • Prediction Visualization │
    │ • Data Upload Interface │
    └─────────────────────────────────────────┘

           Orchestrated by Apache Airflow

```

## 📁 Project Structure

```

stock-prediction/
├── src/
│ ├── ingestion/ # Data fetching and S3 operations
│ │ ├── yfinance_fetcher.py
│ │ ├── csv_handler.py
│ │ └── s3_utils.py
│ ├── processing/ # PySpark ETL pipelines
│ │ ├── etl_pipeline.py
│ │ ├── feature_engineering.py
│ │ └── technical_indicators.py
│ ├── models/ # ML model implementations
│ │ ├── arima_model.py
│ │ ├── lstm_model.py
│ │ ├── prophet_model.py
│ │ └── model_selector.py
│ ├── api/ # FastAPI service
│ │ ├── main.py
│ │ ├── endpoints.py
│ │ ├── auth.py
│ │ └── middleware.py
│ ├── utils/ # Utilities
│ │ ├── logger.py
│ │ ├── config.py
│ │ └── exceptions.py
│ └── config/ # Configuration files
│ └── settings.py
├── infra/ # Infrastructure setup
│ ├── docker-compose.yml
│ ├── airflow/
│ │ └── dags/
│ └── spark/
│ └── spark-defaults.conf
├── dashboard/ # React frontend
│ ├── src/
│ ├── public/
│ └── package.json
├── tests/ # Unit tests
│ ├── test_ingestion.py
│ ├── test_processing.py
│ └── test_api.py
├── scripts/ # Helper scripts
│ ├── run_local.sh
│ ├── deploy.sh
│ └── setup.sh
├── .github/
│ └── workflows/
│ └── ci-cd.yml
├── requirements.txt
├── pyproject.toml
├── .env.template
├── .gitignore
└── README.md

````

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- AWS Account (for S3)
- Node.js 18+ (for dashboard)

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd stock-prediction
````

2. **Set up environment variables**

```bash
cp .env.template .env
# Edit .env with your AWS credentials and configuration
```

3. **Install Python dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Start infrastructure with Docker**

```bash
cd infra
docker-compose up -d
```

5. **Run the FastAPI service**

```bash
python -m src.api.main
```

6. **Start the dashboard**

```bash
cd dashboard
npm install
npm run dev
```

### Using Helper Scripts

**Windows (PowerShell):**

```powershell
.\scripts\setup.ps1
.\scripts\run_local.ps1
```

**Linux/Mac:**

```bash
chmod +x scripts/*.sh
./scripts/setup.sh
./scripts/run_local.sh
```

## 📊 Usage

### Fetch Stock Data

```bash
curl -X GET "http://localhost:8000/api/v1/fetch?symbol=RELIANCE.NS&start_date=2023-01-01&end_date=2024-01-01" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Get Predictions

```bash
curl -X GET "http://localhost:8000/api/v1/predict?symbol=TCS.NS&days=30" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Upload CSV Dataset

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@stock_data.csv"
```

### View Historical Data

```bash
curl -X GET "http://localhost:8000/api/v1/history?symbol=INFY.NS" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 🔐 Security Features

- **JWT Authentication**: Token-based API authentication
- **Rate Limiting**: 60 requests per minute per client
- **S3 Encryption**: Server-side encryption for all S3 objects
- **IAM Roles**: Least-privilege access policies
- **API Logging**: Comprehensive audit trail
- **Input Validation**: Request validation with Pydantic

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## 🔄 CI/CD Pipeline

GitHub Actions workflow automatically:

- Runs linting and formatting checks
- Executes unit tests
- Builds Docker images
- Deploys to staging/production

## 📈 Monitoring

- **Prometheus**: Metrics collection at `/metrics`
- **Logs**: Structured JSON logs in `logs/app.log`
- **Airflow UI**: DAG monitoring at `http://localhost:8080`
- **API Health Check**: `GET /health`

## 🎯 Scalability Improvements

### Current Architecture

- Single-node Spark processing
- Local file caching
- API rate limiting

### Recommended Enhancements

1. **Distributed Spark Cluster**

   - Set up EMR or Databricks cluster
   - Configure YARN or Kubernetes executor
   - Enable dynamic resource allocation

2. **Caching Layer**

   - Add Redis for API response caching
   - Cache frequently accessed predictions
   - Implement cache invalidation strategy

3. **Load Balancing**

   - Deploy multiple API instances
   - Use AWS ALB or NGINX
   - Implement health checks

4. **Database Optimization**

   - Use TimescaleDB for time-series data
   - Implement read replicas
   - Add connection pooling

5. **Async Processing**

   - Use Celery for background tasks
   - Implement message queue (SQS/RabbitMQ)
   - Add result backend (Redis)

6. **Model Serving**

   - Deploy models to SageMaker
   - Use TensorFlow Serving for LSTM
   - Implement A/B testing

7. **Monitoring & Alerting**

   - Set up Grafana dashboards
   - Configure PagerDuty alerts
   - Add distributed tracing (Jaeger)

8. **Data Partitioning**
   - Partition S3 data by date/symbol
   - Use Apache Hudi for incremental processing
   - Implement data versioning

## 🛠️ Configuration

Key configuration options in `.env`:

| Variable                | Description                                     | Default |
| ----------------------- | ----------------------------------------------- | ------- |
| `MODEL_TYPE`            | ML model selection (auto, arima, lstm, prophet) | `auto`  |
| `PREDICTION_DAYS`       | Number of days to forecast                      | `30`    |
| `RATE_LIMIT_PER_MINUTE` | API rate limit                                  | `60`    |
| `SPARK_DRIVER_MEMORY`   | Spark driver memory                             | `4g`    |
| `LOG_LEVEL`             | Logging level                                   | `INFO`  |

## 🐛 Troubleshooting

### Common Issues

**Spark Out of Memory**

```bash
# Increase memory in .env
SPARK_DRIVER_MEMORY=8g
SPARK_EXECUTOR_MEMORY=8g
```

**AWS Credentials Error**

```bash
# Verify credentials
aws sts get-caller-identity
```

**Airflow DAG not appearing**

```bash
# Check DAG syntax
python infra/airflow/dags/stock_pipeline.py
```

## 📝 License

MIT License - see LICENSE file for details

## 👥 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Support

For issues and questions:

- Create an issue on GitHub
- Email: support@stockforecasting.com
- Docs: https://docs.stockforecasting.com
