"""Model selector for automatic model selection based on performance."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import joblib
from pathlib import Path

from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.utils.logger import get_logger
from src.utils.exceptions import ModelTrainingException
from src.config.settings import settings

logger = get_logger(__name__)


class ModelSelector:
    """Automatic model selection based on performance metrics."""

    def __init__(
        self,
        models_to_try: Optional[List[str]] = None,
        metric: str = 'rmse'
    ):
        """
        Initialize model selector.

        Args:
            models_to_try: List of models to try ('arima', 'prophet')
            metric: Metric to use for selection ('rmse', 'mae', 'mape')
        """
        self.models_to_try = models_to_try or ['arima', 'prophet']
        self.metric = metric
        self.logger = logger
        
        self.trained_models = {}
        self.metrics = {}
        self.best_model_name = None
        self.best_model = None

    def train_all_models(
        self,
        train_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
        target_column: str = 'close'
    ) -> Dict:
        """
        Train all specified models and select the best one.

        Args:
            train_data: Training data
            test_data: Optional test data for evaluation
            target_column: Column to forecast

        Returns:
            Dictionary with all model metrics
        """
        try:
            self.logger.info("Training all models for selection...")
            all_metrics = {}
            
            # Split data if test_data not provided
            if test_data is None:
                split_idx = int(len(train_data) * settings.train_test_split)
                train_subset = train_data.iloc[:split_idx]
                test_subset = train_data.iloc[split_idx:]
            else:
                train_subset = train_data
                test_subset = test_data
            
            # Train ARIMA
            if 'arima' in self.models_to_try:
                try:
                    self.logger.info("Training ARIMA...")
                    arima = ARIMAModel()
                    train_metrics = arima.train(train_subset, target_column)
                    eval_metrics = arima.evaluate(test_subset, target_column)
                    
                    self.trained_models['arima'] = arima
                    self.metrics['arima'] = eval_metrics
                    all_metrics['arima'] = eval_metrics
                    
                    self.logger.info(f"ARIMA RMSE: {eval_metrics.get('rmse', 'N/A')}")
                except Exception as e:
                    self.logger.error(f"ARIMA training failed: {str(e)}")
                    all_metrics['arima'] = {'error': str(e)}
            
            # Train Prophet
            if 'prophet' in self.models_to_try:
                try:
                    self.logger.info("Training Prophet...")
                    prophet = ProphetModel()
                    train_metrics = prophet.train(train_subset, target_column)
                    eval_metrics = prophet.evaluate(test_subset, target_column)
                    
                    self.trained_models['prophet'] = prophet
                    self.metrics['prophet'] = eval_metrics
                    all_metrics['prophet'] = eval_metrics
                    
                    self.logger.info(f"Prophet RMSE: {eval_metrics.get('rmse', 'N/A')}")
                except Exception as e:
                    self.logger.error(f"Prophet training failed: {str(e)}")
                    all_metrics['prophet'] = {'error': str(e)}
            
            # Select best model
            self._select_best_model()
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise ModelTrainingException(f"Model training failed: {str(e)}")

    def _select_best_model(self):
        """Select the best model based on the specified metric."""
        try:
            if not self.metrics:
                raise ModelTrainingException("No models trained successfully")
            
            # Filter out models with errors
            valid_metrics = {
                name: metrics for name, metrics in self.metrics.items()
                if 'error' not in metrics and self.metric in metrics
            }
            
            if not valid_metrics:
                raise ModelTrainingException("No valid models available")
            
            # Find best model (lowest metric value)
            best_name = min(
                valid_metrics.keys(),
                key=lambda x: valid_metrics[x][self.metric]
            )
            
            self.best_model_name = best_name
            self.best_model = self.trained_models[best_name]
            
            self.logger.info(
                f"Best model: {best_name} with {self.metric.upper()}: "
                f"{valid_metrics[best_name][self.metric]:.4f}"
            )
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {str(e)}")
            raise ModelTrainingException(f"Model selection failed: {str(e)}")

    def predict(self, steps: int = 30, **kwargs) -> pd.DataFrame:
        """
        Make predictions using the best model.

        Args:
            steps: Number of steps to forecast
            **kwargs: Additional arguments for specific models

        Returns:
            DataFrame with predictions
        """
        try:
            if self.best_model is None:
                raise ModelTrainingException("No model trained. Call train_all_models() first.")
            
            self.logger.info(f"Predicting with {self.best_model_name}...")
            
            predictions = self.best_model.predict(steps=steps)
            
            # Add model name to results
            predictions['model_used'] = self.best_model_name
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def get_best_model_name(self) -> str:
        """Get the name of the best model."""
        return self.best_model_name

    def get_best_model(self):
        """Get the best model object."""
        return self.best_model

    def get_all_metrics(self) -> Dict:
        """Get metrics for all trained models."""
        return self.metrics

    def save_best_model(self, filepath: str):
        """
        Save the best model to disk.

        Args:
            filepath: Path to save the model
        """
        try:
            if self.best_model is None:
                raise ModelTrainingException("No model trained")
            
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model based on type
            if self.best_model_name == 'arima':
                self.best_model.save_model(filepath)
            elif self.best_model_name == 'prophet':
                # Prophet doesn't have a built-in save method, use joblib
                joblib.dump(self.best_model.model, filepath)
            
            # Save metadata
            metadata = {
                'model_name': self.best_model_name,
                'metrics': self.metrics[self.best_model_name]
            }
            metadata_path = path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Best model ({self.best_model_name}) saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")

    def load_best_model(self, filepath: str, model_name: str):
        """
        Load a saved model.

        Args:
            filepath: Path to the saved model
            model_name: Name of the model ('arima', 'lstm', 'prophet')
        """
        try:
            if model_name == 'arima':
                model = ARIMAModel()
                model.load_model(filepath)
            elif model_name == 'prophet':
                from prophet import Prophet
                model_obj = joblib.load(filepath)
                model = ProphetModel()
                model.model = model_obj
            else:
                raise ValueError(f"Unknown model name: {model_name}")
            
            self.best_model = model
            self.best_model_name = model_name
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise ModelTrainingException(f"Model loading failed: {str(e)}")
