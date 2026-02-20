"""
Integration test for preprocessing with model training
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.model.trainer import ModelTrainer
from src.detection.task_detector import TaskType
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def test_integration():
    """Test preprocessing integration with training"""
    print("=" * 80)
    print("Integration Test: Preprocessing + Model Training")
    print("=" * 80)
    
    # Create sample data with missing values (numeric only for sklearn compatibility)
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'feature_1': np.random.randn(n_samples),
        'feature_2': [np.random.randn() if np.random.random() > 0.2 else np.nan for _ in range(n_samples)],
        'feature_3': [np.random.randn() if np.random.random() > 0.1 else np.nan for _ in range(n_samples)],
        'feature_4': np.random.randn(n_samples),
        'feature_5': [np.random.randn() if np.random.random() > 0.3 else np.nan for _ in range(n_samples)],
        'target': np.random.randint(0, 2, n_samples)
    }
    
    X = pd.DataFrame({k: v for k, v in data.items() if k != 'target'})
    y = pd.Series(data['target'], name='target')
    
    print("\n1. Original Data:")
    print(f"   Shape: {X.shape}")
    print(f"   Missing values:\n{X.isna().sum()}")
    
    # Create a simple pipeline
    sklearn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    pipeline_dict = {
        'id': 'test_pipeline_1',
        'name': 'Test Random Forest',
        'pipeline': sklearn_pipeline,
        'hyperparameters': {}
    }
    
    # Test training with preprocessing
    print("\n2. Training ModelTrainer (with preprocessing):")
    trainer = ModelTrainer(TaskType.CLASSIFICATION)
    
    try:
        result = trainer.train_pipeline(pipeline_dict, X, y)
        print(f"   ✅ Training successful!")
        print(f"   CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        print(f"   Train size: {len(result['X_train'])}")
        print(f"   Test size: {len(result['X_test'])}")
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check preprocessing report
    print("\n3. Preprocessing Report:")
    prep_report = trainer.get_preprocessing_report()
    print(f"   Removed features: {prep_report['removed_features']}")
    print(f"   Imputation strategy: {prep_report['imputation_strategy']}")
    print(f"   Imputed values:")
    for feature, value in prep_report['imputation_values'].items():
        print(f"     - {feature}: {value}")
    
    # Verify no missing values in training data
    print("\n4. Verification:")
    missing_in_train = result['X_train'].isna().sum().sum()
    missing_in_test = result['X_test'].isna().sum().sum()
    print(f"   Missing values in X_train: {missing_in_train}")
    print(f"   Missing values in X_test: {missing_in_test}")
    
    if missing_in_train == 0 and missing_in_test == 0:
        print("   ✅ All missing values successfully handled!")
    else:
        print("   ❌ Some missing values remain!")
        return False
    
    print("\n" + "=" * 80)
    print("✅ Integration test completed successfully!")
    print("=" * 80)
    return True

if __name__ == '__main__':
    success = test_integration()
    exit(0 if success else 1)
