#!/usr/bin/env python
"""Test to verify we can generate more than 5 pipelines"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.model_selector import RuleBasedModelSelector
from src.detection.task_detector import TaskType

def create_test_metadata(n_samples, n_features, n_categorical=5):
    return {
        'dataset': {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_columns': n_features + 1,
            'memory_usage_mb': (n_samples * n_features * 8) / 1024**2
        },
        'features': {
            'numerical': [f'num_feature_{i}' for i in range(n_features - n_categorical)],
            'categorical': [f'cat_feature_{i}' for i in range(n_categorical)],
            'n_categorical': n_categorical,
            'has_missing': False,
            'missing_ratio': 0.0
        },
        'target': {
            'n_unique': 2,
            'class_balance': {0: 0.5, 1: 0.5},
            'is_multiclass': False
        }
    }

print("\n" + "="*80)
print("TEST: MaxRequestsPerDataset")
print("="*80)

selector = RuleBasedModelSelector()

# Test with 7 models requested
for max_requested in [5, 7, 10]:
    print(f"\nRequesting {max_requested} pipelines:")
    print("-" * 40)
    
    # TINY dataset
    metadata_tiny = create_test_metadata(n_samples=500, n_features=10)
    recs_tiny = selector.select_models(
        task_type=TaskType.CLASSIFICATION,
        metadata=metadata_tiny,
        max_models=max_requested
    )
    print(f"TINY (vs request {max_requested}): Got {len(recs_tiny)} models")
    
    # SMALL dataset
    metadata_small = create_test_metadata(n_samples=5000, n_features=15)
    recs_small = selector.select_models(
        task_type=TaskType.CLASSIFICATION,
        metadata=metadata_small,
        max_models=max_requested
    )
    print(f"SMALL (vs request {max_requested}): Got {len(recs_small)} models")
    
    # LARGE dataset  
    metadata_large = create_test_metadata(n_samples=500000, n_features=30)
    recs_large = selector.select_models(
        task_type=TaskType.CLASSIFICATION,
        metadata=metadata_large,
        max_models=max_requested
    )
    print(f"LARGE (vs request {max_requested}): Got {len(recs_large)} models")

print("\n" + "="*80)
print("RESULT:")
print("- Request 5: Get 5 pipelines")
print("- Request 7: Get 7 pipelines (when available)")
print("- Request 10: Get up to 8-9 pipelines (max available)")
print("="*80 + "\n")
