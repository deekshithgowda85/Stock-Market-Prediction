"""ARIMA model for stock price forecasting."""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
from src.utils.exceptions import ModelTrainingException, ModelPredictionException

logger = get_logger(__name__)


class ARIMAModel:
    """ARIMA model for time series forecasting."""

    def __init__(self, order: Tuple[int, int, int] = None):
        """
        Initialize ARIMA model.

        Args:
            order: ARIMA order (p, d, q). If None, auto-select
        """
        self.order = order
        self.model = None
        self.model_fit = None
        self.logger = logger

    def check_stationarity(self, series: pd.Series) -> bool:
        """
        Check if series is stationary using ADF test.

        Args:
            series: Time series data

        Returns:
            True if stationary, False otherwise
        """
        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            is_stationary = p_value < 0.05
            
            self.logger.debug(f"ADF test p-value: {p_value:.4f}, Stationary: {is_stationary}")
            return is_stationary
            
        except Exception as e:
            self.logger.warning(f"Stationarity check failed: {str(e)}")
            return False

    def auto_select_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Automatically select best ARIMA order.

        Args:
            series: Time series data

        Returns:
            Best ARIMA order (p, d, q)
        """
        try:
            self.logger.info("Auto-selecting ARIMA order...")
            
            model = auto_arima(
                series,
                start_p=1, start_q=1,
                max_p=5, max_q=5,
                d=None,  # Auto-select d
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            
            order = model.order
            self.logger.info(f"Selected ARIMA order: {order}")
            return order
            
        except Exception as e:
            self.logger.warning(f"Auto-selection failed: {str(e)}. Using default (1,1,1)")
            return (1, 1, 1)

    def train(
        self,
        train_data: pd.DataFrame,
        target_column: str = 'close'
    ) -> Dict:
        """
        Train ARIMA model.

        Args:
            train_data: Training data with date index
            target_column: Column to forecast

        Returns:
            Training metrics

        Raises:
            ModelTrainingException: If training fails
        """
        try:
            self.logger.info("Training ARIMA model...")
            
            # Prepare data
            if 'date' in train_data.columns:
                train_data = train_data.set_index('date')
            
            series = train_data[target_column].astype(float)
            
            # Check stationarity
            is_stationary = self.check_stationarity(series)
            self.logger.info(f"Series stationarity: {is_stationary}")
            
            # Auto-select order if not provided
            if self.order is None:
                self.order = self.auto_select_order(series)
            
            # Train model
            self.model = ARIMA(series, order=self.order)
            self.model_fit = self.model.fit()
            
            # Get training metrics
            aic = self.model_fit.aic
            bic = self.model_fit.bic
            
            # In-sample predictions
            predictions = self.model_fit.fittedvalues
            actuals = series[len(series) - len(predictions):]
            
            # Calculate metrics
            mse = np.mean((actuals - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actuals - predictions))
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            
            metrics = {
                'model': 'ARIMA',
                'order': self.order,
                'aic': float(aic),
                'bic': float(bic),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'train_size': len(series)
            }
            
            self.logger.info(f"ARIMA training complete. RMSE: {rmse:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"ARIMA training failed: {str(e)}")
            raise ModelTrainingException(f"ARIMA training failed: {str(e)}")

    def predict(self, steps: int = 30) -> pd.DataFrame:
        """
        Make future predictions.

        Args:
            steps: Number of steps to forecast

        Returns:
            DataFrame with predictions

        Raises:
            ModelPredictionException: If prediction fails
        """
        try:
            if self.model_fit is None:
                raise ModelPredictionException("Model not trained. Call train() first.")
            
            self.logger.info(f"Generating {steps} step forecast...")
            
            # Forecast
            forecast = self.model_fit.forecast(steps=steps)
            
            # Get confidence intervals
            forecast_df = self.model_fit.get_forecast(steps=steps).summary_frame()
            
            # Create result DataFrame
            result = pd.DataFrame({
                'forecast_step': range(1, steps + 1),
                'predicted_value': forecast.values,
                'lower_ci': forecast_df['mean_ci_lower'].values,
                'upper_ci': forecast_df['mean_ci_upper'].values
            })
            
            self.logger.info("Prediction complete")
            return result
            
        except ModelPredictionException:
            raise
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise ModelPredictionException(f"ARIMA prediction failed: {str(e)}")

    def evaluate(
        self,
        test_data: pd.DataFrame,
        target_column: str = 'close'
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            test_data: Test data
            target_column: Target column

        Returns:
            Evaluation metrics
        """
        try:
            if 'date' in test_data.columns:
                test_data = test_data.set_index('date')
            
            actuals = test_data[target_column].astype(float)
            n_steps = len(actuals)
            
            # Predict
            predictions = self.predict(n_steps)['predicted_value'].values
            
            # Calculate metrics
            mse = np.mean((actuals.values - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actuals.values - predictions))
            mape = np.mean(np.abs((actuals.values - predictions) / actuals.values)) * 100
            
            metrics = {
                'model': 'ARIMA',
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'test_size': n_steps
            }
            
            self.logger.info(f"ARIMA evaluation complete. RMSE: {rmse:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {'error': str(e)}

    def save_model(self, filepath: str):
        """Save model to file."""
        try:
            self.model_fit.save(filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")

    def load_model(self, filepath: str):
        """Load model from file."""
        try:
            from statsmodels.tsa.arima.model import ARIMAResults
            self.model_fit = ARIMAResults.load(filepath)
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise ModelTrainingException(f"Model loading failed: {str(e)}")
