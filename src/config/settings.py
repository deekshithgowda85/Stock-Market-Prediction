"""Configuration management for the Stock Forecasting System."""
import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=('settings_',)
    )

    # Data Configuration
    dataset_path: str = Field(default="dataset", description="Path to dataset folder")

    # NSE API Configuration
    nse_api_base_url: str = Field(default="https://nse-api-khaki.vercel.app", description="NSE API base URL")
    nse_api_timeout: int = Field(default=10, description="NSE API timeout in seconds")

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # Model Configuration
    model_type: str = Field(default="auto")  # auto, arima, prophet
    train_test_split: float = Field(default=0.8)
    prediction_days: int = Field(default=30)

    # Logging
    log_level: str = Field(default="INFO")

    # Environment
    environment: str = Field(default="development")

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"


# Global settings instance
settings = Settings()

