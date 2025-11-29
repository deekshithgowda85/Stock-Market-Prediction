"""Simple run script for the Stock Prediction API."""
import uvicorn
from src.config.settings import settings

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Stock Market Forecasting API")
    print("=" * 60)
    print(f"Environment: {settings.environment}")
    print(f"Host: {settings.api_host}")
    print(f"Port: {settings.api_port}")
    print(f"Dataset Path: {settings.dataset_path}")
    print(f"API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
