
import pandas as pd
import numpy as np
from src.data.metadata import MetadataExtractor
from src.pipeline.model_selector import RuleBasedModelSelector
from src.detection.task_detector import TaskType

def test_enhanced_metadata():
    print("Testing Metadata Extraction with Skewness, Outliers, etc.")
    
    # Create synthetic dataset with high skewness and outliers
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame({
        'skewed': np.random.exponential(scale=2.0, size=n),  # Highly skewed
        'outliers': np.random.normal(0, 1, n),
        'high_card': [f"cat_{i % 50}" for i in range(n)],    # High cardinality
        'normal': np.random.normal(size=n)
    })
    # Add manual outliers
    X.loc[0:49, 'outliers'] = 100 
    
    # Regression target - skewed
    y_reg = pd.Series(np.random.exponential(scale=5.0, size=n), name='price')
    
    # Classification target - imbalanced
    y_cls = pd.Series([0] * 990 + [1] * 10, name='fraud') # 1% minority
    
    extractor = MetadataExtractor()
    
    # 1. Test Regression Metadata
    print("\n--- Regression Metadata Analysis ---")
    reg_meta = extractor.extract(X, y_reg)
    print(f"Target Skew: {reg_meta['statistics']['target_skew']:.4f}")
    print(f"Feature Skew (skewed): {reg_meta['statistics']['skewness']['skewed']:.4f}")
    print(f"Outlier Density: {reg_meta['outliers']['overall_density']:.2%}")
    print(f"High Cardinality Features: {reg_meta['cardinality']['high_cardinality_features']}")
    
    selector = RuleBasedModelSelector()
    recommendations = selector.select_models(TaskType.REGRESSION, reg_meta)
    
    print("\nRegression Recommendations:")
    for rec in recommendations:
        print(f" - {rec.name} (Priority {rec.priority}): {rec.reason}")

    # 2. Test Classification Metadata
    print("\n--- Classification Metadata Analysis ---")
    cls_meta = extractor.extract(X, y_cls)
    print(f"Is Imbalanced: {cls_meta['imbalance']['is_highly_imbalanced']}")
    print(f"Min Class Ratio: {cls_meta['imbalance']['min_class_ratio']:.2%}")
    
    recommendations = selector.select_models(TaskType.CLASSIFICATION, cls_meta)
    
    print("\nClassification Recommendations:")
    for rec in recommendations:
        if 'class_weight' in str(rec.hyperparameters) or 'balanced' in rec.reason:
            print(f" - {rec.name} (Priority {rec.priority}): {rec.reason}")
        else:
            print(f" - {rec.name} (Priority {rec.priority})")

if __name__ == "__main__":
    test_enhanced_metadata()
