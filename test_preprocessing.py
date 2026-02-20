"""
Test script for preprocessing functionality
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.preprocessor import DataPreprocessor

def test_preprocessing():
    """Test the preprocessing module"""
    print("=" * 80)
    print("Testing Data Preprocessing Module")
    print("=" * 80)
    
    # Create sample data with missing values
    np.random.seed(42)
    
    data = {
        'numeric_feature_1': np.random.randn(100),
        'numeric_feature_2': [np.random.randn() if np.random.random() > 0.2 else np.nan for _ in range(100)],
        'numeric_feature_3': [np.random.randn() if np.random.random() > 0.7 else np.nan for _ in range(100)],
        'categorical_feature_1': np.random.choice(['A', 'B', 'C', None], 100),
        'categorical_feature_2': [np.random.choice(['X', 'Y', 'Z', None]) if np.random.random() > 0.3 else np.nan for _ in range(100)],
        'target': np.random.randint(0, 2, 100)
    }
    
    X = pd.DataFrame({k: v for k, v in data.items() if k != 'target'})
    y = pd.Series(data['target'], name='target')
    
    print("\nOriginal Data:")
    print(f"Shape: {X.shape}")
    print(f"Missing values per column:\n{X.isna().sum()}")
    print(f"Missing ratio per column:\n{X.isna().mean()}")
    
    # Test with 'auto' strategy
    print("\n" + "=" * 80)
    print("Test 1: Auto Imputation Strategy")
    print("=" * 80)
    
    preprocessor = DataPreprocessor(missing_threshold=0.6, imputation_strategy='auto')
    X_processed, y_processed = preprocessor.fit(X.copy(), y.copy())
    
    print(f"\nProcessed Data Shape: {X_processed.shape}")
    print(f"Remaining samples: {len(X_processed)}")
    print(f"Missing values after preprocessing:\n{X_processed.isna().sum()}")
    
    report = preprocessor.get_preprocessing_report(X)
    print(f"\nPreprocessing Report:")
    print(f"  Total missing values: {report['total_missing_values']}")
    print(f"  Features with missing: {report['features_with_missing']}")
    print(f"  Removed features: {report['removed_features']}")
    print(f"  Imputation strategy: {report['imputation_strategy']}")
    print(f"  Imputed values: {report['imputation_values']}")
    
    # Test with 'median' strategy
    print("\n" + "=" * 80)
    print("Test 2: Median Imputation Strategy")
    print("=" * 80)
    
    preprocessor2 = DataPreprocessor(missing_threshold=0.6, imputation_strategy='median')
    X_processed2, y_processed2 = preprocessor2.fit(X.copy(), y.copy())
    
    print(f"\nProcessed Data Shape: {X_processed2.shape}")
    print(f"Missing values after preprocessing:\n{X_processed2.isna().sum()}")
    
    # Test with 'drop' strategy
    print("\n" + "=" * 80)
    print("Test 3: Drop Strategy (Remove rows with missing)")
    print("=" * 80)
    
    preprocessor3 = DataPreprocessor(missing_threshold=0.6, imputation_strategy='drop')
    X_processed3, y_processed3 = preprocessor3.fit(X.copy(), y.copy())
    
    print(f"\nProcessed Data Shape: {X_processed3.shape}")
    print(f"Samples dropped: {len(X) - len(X_processed3)}")
    print(f"Missing values after preprocessing:\n{X_processed3.isna().sum()}")
    
    print("\n" + "=" * 80)
    print("✅ All preprocessing tests completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    test_preprocessing()
