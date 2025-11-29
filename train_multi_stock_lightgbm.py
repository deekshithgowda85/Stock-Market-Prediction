"""
Train Multi-Stock LightGBM Model
Fast training (2-5 minutes) on all stocks
"""

import sys

# Check dependencies
print("Checking dependencies...")
try:
    import pandas as pd
    print("✓ pandas installed")
except ImportError as e:
    print(f"✗ pandas error: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy installed")
except ImportError as e:
    print(f"✗ numpy not installed: {e}")
    sys.exit(1)

try:
    import lightgbm as lgb
    print(f"✓ lightgbm {lgb.__version__} installed")
except ImportError as e:
    print(f"✗ lightgbm not installed: {e}")
    print("Please run: pip install lightgbm")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
    print("✓ scikit-learn installed")
except ImportError as e:
    print(f"✗ scikit-learn not installed: {e}")
    sys.exit(1)

print("\nAll dependencies OK! Loading modules...\n")

from pathlib import Path
from src.models.multi_stock_lightgbm import MultiStockLightGBM
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_lightgbm_model():
    """Train LightGBM on all stocks"""
    
    dataset_path = Path("dataset")
    model_save_path = Path("models/multi_stock_lightgbm")
    
    # Initialize model
    model = MultiStockLightGBM(lookback=60)
    
    # Train on all stocks
    model.train(dataset_path, test_size=0.2)
    
    # Save model
    model.save_model(model_save_path)
    logger.info(f"\n✅ Model saved to {model_save_path}")
    
    # Test prediction on one stock
    logger.info("\n" + "="*80)
    logger.info("Testing prediction on RELIANCE...")
    logger.info("="*80)
    
    import pandas as pd
    reliance_df = pd.read_csv(dataset_path / "RELIANCE.csv")
    reliance_df.columns = reliance_df.columns.str.lower()  # Normalize column names
    predictions = model.predict("RELIANCE", reliance_df, days=30)
    
    logger.info(f"\nNext 30 days predictions for RELIANCE:")
    for i, pred in enumerate(predictions['predictions'][:10], 1):
        logger.info(f"  Day {i}: ₹{pred:.2f}")
    logger.info("  ...")
    
    logger.info("\n✅ Training and testing completed!")
    logger.info(f"Model ready for predictions via API")


if __name__ == "__main__":
    train_lightgbm_model()
