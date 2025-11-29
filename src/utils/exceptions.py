"""Custom exceptions for the Stock Forecasting System."""


class StockForecastingException(Exception):
    """Base exception for stock forecasting system."""
    pass


class DataIngestionException(StockForecastingException):
    """Exception raised during data ingestion."""
    pass


class DataValidationException(StockForecastingException):
    """Exception raised during data validation."""
    pass


class S3Exception(StockForecastingException):
    """Exception raised during S3 operations."""
    pass


class ETLException(StockForecastingException):
    """Exception raised during ETL processing."""
    pass


class FeatureEngineeringException(StockForecastingException):
    """Exception raised during feature engineering."""
    pass


class ModelTrainingException(StockForecastingException):
    """Exception raised during model training."""
    pass


class ModelPredictionException(StockForecastingException):
    """Exception raised during model prediction."""
    pass


class APIException(StockForecastingException):
    """Exception raised in API operations."""
    pass


class AuthenticationException(APIException):
    """Exception raised during authentication."""
    pass


class RateLimitException(APIException):
    """Exception raised when rate limit is exceeded."""
    pass


class InvalidSymbolException(DataIngestionException):
    """Exception raised for invalid stock symbols."""
    pass


class NoDataException(DataIngestionException):
    """Exception raised when no data is available."""
    pass
