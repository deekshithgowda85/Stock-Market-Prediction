# 📈 Stock Market Forecasting System (Simplified)

A simple and functional Stock Market Forecasting Platform with Machine Learning models.

## 🎯 Features

- **Local CSV Data**: Works with historical stock data from CSV files
- **ML Models**: ARIMA and Prophet for time series forecasting
- **REST API**: FastAPI service with interactive documentation
- **Easy to Run**: Simple setup and execution

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run the API

```powershell
python run.py
```

The API will start on `http://localhost:8000`

### 3. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📊 API Endpoints

### List Available Stocks

```http
GET /api/v1/stocks
```

### Get Stock Data

```http
GET /api/v1/data/{symbol}?limit=100
```

### Get Latest Price

```http
GET /api/v1/latest/{symbol}
```

### Generate Predictions

```http
POST /api/v1/predict
Content-Type: application/json

{
  "symbol": "RELIANCE",
  "days": 30,
  "model_type": "auto"
}
```

Model types: `auto`, `arima`, `prophet`

## 📦 Dependencies

- pandas, numpy - Data manipulation
- yfinance - Fetch live stock data (optional)
- scikit-learn - Machine learning utilities
- statsmodels - ARIMA model
- prophet - Facebook Prophet model
- fastapi, uvicorn - Web API framework

## 🎓 Models

- **ARIMA**: Auto-regressive Integrated Moving Average
- **Prophet**: Facebook's forecasting tool
- **Auto Mode**: Trains both and selects best performer

## 📄 License

See LICENSE file for details.

## 🎯 Features

- **Real-time Data Ingestion**: Fetch live stock data from YFinance (NSE/BSE markets)
- **Distributed Processing**: PySpark-based ETL pipelines for scalable data processing
- **Advanced ML Models**: ARIMA, LSTM, and Prophet with automatic model selection
- **REST API**: FastAPI service with authentication and rate limiting
- **Interactive Dashboard**: React-based visualization with real-time predictions
- **Workflow Orchestration**: Airflow DAGs for automated data pipelines
- **Cloud Storage**: AWS S3 integration for data persistence
- **Enterprise Features**: Logging, monitoring, retry mechanisms, and security

## 🏗️ Architecture

```
┌─────────────────┐
│  YFinance API   │
│   (NSE/BSE)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        Data Ingestion Layer              │
│  ┌──────────┐      ┌──────────────┐    │
│  │ YFinance │      │ CSV Upload   │    │
│  │ Fetcher  │      │   Handler    │    │
│  └──────────┘      └──────────────┘    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         AWS S3 Storage                   │
│  /raw/  /processed/  /predictions/      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      PySpark ETL & Feature Eng.         │
│  • Data Cleaning                         │
│  • Technical Indicators (RSI, MACD)      │
│  • Rolling Windows                       │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         ML Model Training                │
│  ┌────────┐ ┌────────┐ ┌─────────┐    │
│  │ ARIMA  │ │  LSTM  │ │ Prophet │    │
│  └────────┘ └────────┘ └─────────┘    │
│       Auto-selection by RMSE            │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        FastAPI Service                   │
│  • Prediction Endpoints                  │
│  • Authentication                        │
│  • Rate Limiting                         │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│       React Dashboard                    │
│  • Stock Charts                          │
│  • Prediction Visualization              │
│  • Data Upload Interface                 │
└─────────────────────────────────────────┘

       Orchestrated by Apache Airflow
```

## 📁 Project Structure

```
stock-prediction/
├── src/
│   ├── ingestion/          # Data fetching and S3 operations
│   │   ├── yfinance_fetcher.py
│   │   ├── csv_handler.py
│   │   └── s3_utils.py
│   ├── processing/         # PySpark ETL pipelines
│   │   ├── etl_pipeline.py
│   │   ├── feature_engineering.py
│   │   └── technical_indicators.py
│   ├── models/            # ML model implementations
│   │   ├── arima_model.py
│   │   ├── lstm_model.py
│   │   ├── prophet_model.py
│   │   └── model_selector.py
│   ├── api/               # FastAPI service
│   │   ├── main.py
│   │   ├── endpoints.py
│   │   ├── auth.py
│   │   └── middleware.py
│   ├── utils/             # Utilities
│   │   ├── logger.py
│   │   ├── config.py
│   │   └── exceptions.py
│   └── config/            # Configuration files
│       └── settings.py
├── infra/                 # Infrastructure setup
│   ├── docker-compose.yml
│   ├── airflow/
│   │   └── dags/
│   └── spark/
│       └── spark-defaults.conf
├── dashboard/             # React frontend
│   ├── src/
│   ├── public/
│   └── package.json
├── tests/                 # Unit tests
│   ├── test_ingestion.py
│   ├── test_processing.py
│   └── test_api.py
├── scripts/              # Helper scripts
│   ├── run_local.sh
│   ├── deploy.sh
│   └── setup.sh
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── requirements.txt
├── pyproject.toml
├── .env.template
├── .gitignore
└── README.md
```

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
```

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
