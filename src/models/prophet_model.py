"""Prophet model for stock price forecasting."""
import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
from src.utils.exceptions import ModelTrainingException, ModelPredictionException

logger = get_logger(__name__)


class ProphetModel:
    """Prophet model for time series forecasting."""

    def __init__(
        self,
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        daily_seasonality: bool = False,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True
    ):
        """
        Initialize Prophet model.

        Args:
            growth: Growth model ('linear' or 'logistic')
            changepoint_prior_scale: Flexibility of trend
            seasonality_prior_scale: Flexibility of seasonality
            daily_seasonality: Include daily seasonality
            weekly_seasonality: Include weekly seasonality
            yearly_seasonality: Include yearly seasonality
        """
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        
        self.model = None
        self.logger = logger

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns).

        Args:
            data: Input DataFrame
            target_column: Column to forecast

        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        try:
            df = data.copy()
            
            # Ensure date column exists
            if 'date' not in df.columns:
                if df.index.name == 'date':
                    df = df.reset_index()
                else:
                    raise ValueError("'date' column not found")
            
            # Prepare Prophet format
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['date']),
                'y': df[target_column].astype(float)
            })
            
            self.logger.debug(f"Prepared data: {len(prophet_df)} rows")
            return prophet_df
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            raise ModelTrainingException(f"Data preparation failed: {str(e)}")

    def train(
        self,
        train_data: pd.DataFrame,
        target_column: str = 'close'
    ) -> Dict:
        """
        Train Prophet model.

        Args:
            train_data: Training data
            target_column: Column to forecast

        Returns:
            Training metrics

        Raises:
            ModelTrainingException: If training fails
        """
        try:
            self.logger.info("Training Prophet model...")
            
            # Prepare data
            prophet_df = self.prepare_data(train_data, target_column)
            
            # Initialize model
            self.model = Prophet(
                growth=self.growth,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                daily_seasonality=self.daily_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality
            )
            
            # Train model
            self.model.fit(prophet_df)
            
            # In-sample predictions
            forecast = self.model.predict(prophet_df)
            
            # Calculate metrics
            actuals = prophet_df['y'].values
            predictions = forecast['yhat'].values
            
            mse = np.mean((actuals - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actuals - predictions))
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            
            metrics = {
                'model': 'Prophet',
                'growth': self.growth,
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'train_size': len(prophet_df)
            }
            
            self.logger.info(f"Prophet training complete. RMSE: {rmse:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Prophet training failed: {str(e)}")
            raise ModelTrainingException(f"Prophet training failed: {str(e)}")

    def predict(self, steps: int = 30) -> pd.DataFrame:
        """
        Make future predictions.

        Args:
            steps: Number of days to forecast

        Returns:
            DataFrame with predictions

        Raises:
            ModelPredictionException: If prediction fails
        """
        try:
            if self.model is None:
                raise ModelPredictionException("Model not trained. Call train() first.")
            
            self.logger.info(f"Generating {steps} day forecast...")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=steps)
            
            # Predict
            forecast = self.model.predict(future)
            
            # Get only future predictions
            forecast = forecast.tail(steps)
            
            # Create result DataFrame
            result = pd.DataFrame({
                'date': forecast['ds'],
                'forecast_step': range(1, steps + 1),
                'predicted_value': forecast['yhat'].values,
                'lower_ci': forecast['yhat_lower'].values,
                'upper_ci': forecast['yhat_upper'].values
            })
            
            self.logger.info("Prediction complete")
            return result
            
        except ModelPredictionException:
            raise
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise ModelPredictionException(f"Prophet prediction failed: {str(e)}")

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
            # Prepare test data
            test_df = self.prepare_data(test_data, target_column)
            
            # Predict
            forecast = self.model.predict(test_df)
            
            # Calculate metrics
            actuals = test_df['y'].values
            predictions = forecast['yhat'].values
            
            mse = np.mean((actuals - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actuals - predictions))
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            
            metrics = {
                'model': 'Prophet',
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'test_size': len(test_df)
            }
            
            self.logger.info(f"Prophet evaluation complete. RMSE: {rmse:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {'error': str(e)}

    def plot_forecast(self, forecast: pd.DataFrame):
        """
        Plot forecast (requires matplotlib).

        Args:
            forecast: Forecast DataFrame from predict()
        """
        try:
            import matplotlib.pyplot as plt
            
            fig = self.model.plot(forecast)
            plt.title('Prophet Forecast')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.warning(f"Plotting failed: {str(e)}")

    def plot_components(self, forecast: pd.DataFrame):
        """
        Plot forecast components.

        Args:
            forecast: Forecast DataFrame
        """
        try:
            fig = self.model.plot_components(forecast)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.logger.warning(f"Component plotting failed: {str(e)}")
