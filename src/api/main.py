"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from pathlib import Path

from src.api.endpoints import router
from src.api.data_sources import router as sources_router
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Market Forecasting API",
    description="Stock market forecasting with multi-source failsafe mechanism",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)
app.include_router(sources_router, tags=["Data Sources"])

# Mount static files (dashboard)
dashboard_path = Path(__file__).parent.parent.parent / "dashboard"
if dashboard_path.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_path)), name="static")


@app.on_event("startup")
async def startup_event():
    """Actions to perform on startup."""
    logger.info("Starting Stock Market Forecasting API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"API running on {settings.api_host}:{settings.api_port}")


@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on shutdown."""
    logger.info("Shutting down Stock Market Forecasting API")


@app.get("/")
async def root():
    """Root endpoint - redirect to dashboard."""
    return FileResponse(str(dashboard_path / "index.html")) if dashboard_path.exists() else {
        "service": "Stock Market Forecasting API",
        "version": "2.0.0",
        "status": "running",
        "features": ["smart_data_loading", "auto_update", "yfinance_integration"],
        "docs": "/docs",
        "dashboard": "/static/index.html"
    }


@app.get("/dashboard")
async def dashboard():
    """Dashboard endpoint."""
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path / "index.html"))
    return {"error": "Dashboard not found"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment
    }


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=(settings.environment == "development"),
        log_level=settings.log_level.lower()
    )

