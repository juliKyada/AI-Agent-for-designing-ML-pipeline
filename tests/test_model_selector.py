"""
Test the Rule-Based Model Selector
"""
import pandas as pd
import numpy as np
from src.pipeline.model_selector import RuleBasedModelSelector, DatasetSize, DatasetComplexity
from src.detection.task_detector import TaskType


def create_test_metadata(n_samples, n_features, n_categorical=5, n_unique_target=2, is_classification=True):
    """Create test metadata for different scenarios"""
    numerical_features = [f'num_{i}' for i in range(n_features - n_categorical)]
    categorical_features = [f'cat_{i}' for i in range(n_categorical)]
    
    metadata = {
        'dataset': {
            'n_samples': n_samples,
            'n_features': n_features,
            'feature_names': numerical_features + categorical_features,
            'target_name': 'target',
            'memory_usage_mb': (n_samples * n_features * 8) / 1024**2
        },
        'features': {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'datetime': [],
            'n_numerical': len(numerical_features),
            'n_categorical': len(categorical_features),
            'n_datetime': 0
        },
        'target': {
            'name': 'target',
            'dtype': 'int64' if is_classification else 'float64',
            'n_unique': n_unique_target,
            'missing_values': 0,
            'missing_ratio': 0.0
        },
        'quality': {
            'overall_missing_ratio': 0.05,
            'columns_with_missing': [],
            'high_missing_columns': [],
            'duplicate_rows': 0,
            'constant_columns': [],
            'target_missing': 0,
            'issues': []
        },
        'statistics': {}
    }
    
    # Add class balance for classification
    if is_classification:
        if n_unique_target == 2:
            metadata['target']['class_balance'] = {0: 0.6, 1: 0.4}
        else:
            metadata['target']['class_balance'] = {i: 1.0/n_unique_target for i in range(n_unique_target)}
    
    return metadata


def test_tiny_dataset_classification():
    """Test model selection for tiny classification dataset"""
    print("\n" + "="*80)
    print("TEST 1: Tiny Dataset Classification (500 samples, 10 features)")
    print("="*80)
    
    metadata = create_test_metadata(n_samples=500, n_features=10, n_categorical=3)
    selector = RuleBasedModelSelector()
    
    recommendations = selector.select_models(
        task_type=TaskType.CLASSIFICATION,
        metadata=metadata,
        max_models=5
    )
    
    print(f"\nRecommended {len(recommendations)} models:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.name} (Priority: {rec.priority})")
        print(f"   Reason: {rec.reason}")
        print(f"   Description: {rec.description}")
    
    # Assertions
    assert len(recommendations) > 0, "Should recommend at least one model"
    assert selector.dataset_size == DatasetSize.TINY, "Should detect TINY dataset"
    assert any("Logistic Regression" in rec.name for rec in recommendations), "Should recommend simple models for tiny data"
    print("\n✓ Test passed!")
    return recommendations


def test_large_dataset_classification():
    """Test model selection for large classification dataset"""
    print("\n" + "="*80)
    print("TEST 2: Large Dataset Classification (200,000 samples, 30 features)")
    print("="*80)
    
    metadata = create_test_metadata(n_samples=200000, n_features=30, n_categorical=5, n_unique_target=3)
    selector = RuleBasedModelSelector()
    
    recommendations = selector.select_models(
        task_type=TaskType.CLASSIFICATION,
        metadata=metadata,
        max_models=5
    )
    
    print(f"\nRecommended {len(recommendations)} models:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.name} (Priority: {rec.priority})")
        print(f"   Reason: {rec.reason}")
        print(f"   Description: {rec.description}")
    
    # Assertions
    assert selector.dataset_size == DatasetSize.LARGE, "Should detect LARGE dataset"
    assert any("LightGBM" in rec.name for rec in recommendations), "Should recommend LightGBM for large data"
    print("\n✓ Test passed!")
    return recommendations


def test_high_dimensional_classification():
    """Test model selection for high-dimensional classification"""
    print("\n" + "="*80)
    print("TEST 3: High-Dimensional Classification (1,000 samples, 200 features)")
    print("="*80)
    
    metadata = create_test_metadata(n_samples=1000, n_features=200, n_categorical=10)
    selector = RuleBasedModelSelector()
    
    recommendations = selector.select_models(
        task_type=TaskType.CLASSIFICATION,
        metadata=metadata,
        max_models=5
    )
    
    print(f"\nRecommended {len(recommendations)} models:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.name} (Priority: {rec.priority})")
        print(f"   Reason: {rec.reason}")
        print(f"   Description: {rec.description}")
    
    # Assertions
    assert selector.dataset_complexity in [DatasetComplexity.MODERATE, DatasetComplexity.COMPLEX], \
        "Should detect high complexity"
    assert any("Logistic Regression" in rec.name for rec in recommendations), \
        "Should recommend regularized models for high-dimensional data"
    print("\n✓ Test passed!")
    return recommendations


def test_small_regression():
    """Test model selection for small regression dataset"""
    print("\n" + "="*80)
    print("TEST 4: Small Regression Dataset (5,000 samples, 15 features)")
    print("="*80)
    
    metadata = create_test_metadata(
        n_samples=5000, 
        n_features=15, 
        n_categorical=3,
        n_unique_target=4500,  # Continuous target
        is_classification=False
    )
    selector = RuleBasedModelSelector()
    
    recommendations = selector.select_models(
        task_type=TaskType.REGRESSION,
        metadata=metadata,
        max_models=5
    )
    
    print(f"\nRecommended {len(recommendations)} models:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.name} (Priority: {rec.priority})")
        print(f"   Reason: {rec.reason}")
        print(f"   Description: {rec.description}")
    
    # Assertions
    assert selector.dataset_size == DatasetSize.SMALL, "Should detect SMALL dataset"
    assert any("Random Forest" in rec.name or "XGBoost" in rec.name for rec in recommendations), \
        "Should recommend ensemble methods for small-medium regression"
    print("\n✓ Test passed!")
    return recommendations


def test_imbalanced_classification():
    """Test model selection for imbalanced classification"""
    print("\n" + "="*80)
    print("TEST 5: Imbalanced Classification (10,000 samples, 20 features)")
    print("="*80)
    
    metadata = create_test_metadata(n_samples=10000, n_features=20, n_categorical=5, n_unique_target=2)
    # Make it imbalanced
    metadata['target']['class_balance'] = {0: 0.95, 1: 0.05}
    
    selector = RuleBasedModelSelector()
    
    recommendations = selector.select_models(
        task_type=TaskType.CLASSIFICATION,
        metadata=metadata,
        max_models=5
    )
    
    print(f"\nRecommended {len(recommendations)} models:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.name} (Priority: {rec.priority})")
        print(f"   Reason: {rec.reason}")
        print(f"   Has class_weight param: {'class_weight' in rec.hyperparameters.get('model__class_weight', [])}")
    
    # Check if class balancing is applied
    has_class_weight = any('model__class_weight' in rec.hyperparameters for rec in recommendations)
    print(f"\n✓ Class balancing applied: {has_class_weight}")
    print("✓ Test passed!")
    return recommendations


def test_selection_summary():
    """Test selection summary functionality"""
    print("\n" + "="*80)
    print("TEST 6: Selection Summary")
    print("="*80)
    
    metadata = create_test_metadata(n_samples=50000, n_features=25, n_categorical=8)
    selector = RuleBasedModelSelector()
    
    recommendations = selector.select_models(
        task_type=TaskType.CLASSIFICATION,
        metadata=metadata,
        max_models=3
    )
    
    summary = selector.get_selection_summary()
    
    print("\nSelection Summary:")
    print(f"Task Type: {summary['task_type']}")
    print(f"Dataset Size Category: {summary['dataset_size_category']}")
    print(f"Dataset Complexity: {summary['dataset_complexity']}")
    print(f"Samples: {summary['n_samples']:,}")
    print(f"Features: {summary['n_features']}")
    print(f"\nSelected Models ({len(summary['selected_models'])}):")
    for model_info in summary['selected_models']:
        print(f"  - {model_info['name']} (Priority: {model_info['priority']})")
        print(f"    {model_info['reason']}")
    
    # Assertions
    assert summary['task_type'] is not None
    assert summary['dataset_size_category'] is not None
    assert len(summary['selected_models']) == 3
    print("\n✓ Test passed!")
    return summary


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("RULE-BASED MODEL SELECTOR - TEST SUITE")
    print("="*80)
    
    try:
        test_tiny_dataset_classification()
        test_large_dataset_classification()
        test_high_dimensional_classification()
        test_small_regression()
        test_imbalanced_classification()
        test_selection_summary()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
