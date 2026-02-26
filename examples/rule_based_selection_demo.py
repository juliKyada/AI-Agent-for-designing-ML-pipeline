"""
Example: Using the Rule-Based Model Selector

This example demonstrates how the rule-based model selection works
and how it automatically chooses appropriate models based on dataset characteristics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import PipelineAgent
from src.pipeline import RuleBasedModelSelector
from src.data import MetadataExtractor
from src.detection import TaskDetector


def example_1_small_dataset():
    """Example 1: Small dataset - demonstrates selection of simpler models"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Small Dataset (Credit Card Default)")
    print("="*80)
    
    # Create a small synthetic dataset
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_ratio': np.random.uniform(0, 1, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    })
    
    print(f"\nDataset Info:")
    print(f"  Samples: {len(data)}")
    print(f"  Features: {len(data.columns) - 1}")
    print(f"  Target: default (binary)")
    
    # Run pipeline with rule-based selection
    agent = PipelineAgent()
    results = agent.run(dataframe=data, target_column='default', n_pipelines=3)
    
    print(f"\n{'─'*80}")
    print("Selected Models (Rule-Based):")
    print(f"{'─'*80}")
    for i, pipeline in enumerate(results['all_pipelines'], 1):
        print(f"\n{i}. {pipeline['pipeline_name']}")
        if 'selection_reason' in pipeline:
            print(f"   Selection Reason: {pipeline['selection_reason']}")
        print(f"   Performance: {list(pipeline['metrics'].items())[0]}")
    
    print(f"\nBest Model: {results['best_pipeline']['name']}")
    

def example_2_large_dataset():
    """Example 2: Large dataset - demonstrates selection of scalable models"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Large Dataset (E-commerce Transactions)")
    print("="*80)
    
    # Create a large synthetic dataset
    np.random.seed(42)
    n_samples = 150000
    
    print(f"Generating {n_samples:,} samples... (this may take a moment)")
    
    data = pd.DataFrame({
        'order_value': np.random.uniform(10, 500, n_samples),
        'customer_age': np.random.randint(18, 80, n_samples),
        'items_count': np.random.randint(1, 20, n_samples),
        'days_since_last_order': np.random.randint(0, 365, n_samples),
        'customer_segment': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'shipping_method': np.random.choice(['Standard', 'Express', 'Prime'], n_samples),
        'payment_method': np.random.choice(['Card', 'PayPal', 'Crypto'], n_samples),
        'product_category': np.random.choice(['Electronics', 'Fashion', 'Home', 'Sports'], n_samples),
        'will_return': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    print(f"\nDataset Info:")
    print(f"  Samples: {len(data):,}")
    print(f"  Features: {len(data.columns) - 1}")
    print(f"  Target: will_return (binary)")
    print(f"  Categorical Features: 4")
    
    # Extract metadata and show rule-based selection
    from src.data import DataLoader
    loader = DataLoader()
    X, y = loader.load_from_dataframe(data, 'will_return')
    
    metadata_extractor = MetadataExtractor()
    metadata = metadata_extractor.extract(X, y)
    
    task_detector = TaskDetector()
    task_type, _, _ = task_detector.detect(y)
    
    # Use rule-based selector directly
    selector = RuleBasedModelSelector()
    recommendations = selector.select_models(task_type, metadata, max_models=3)
    
    print(f"\n{'─'*80}")
    print("Rule-Based Model Selection Results:")
    print(f"{'─'*80}")
    
    summary = selector.get_selection_summary()
    print(f"\nDataset Classification:")
    print(f"  Size Category: {summary['dataset_size_category'].upper()}")
    print(f"  Complexity: {summary['dataset_complexity'].upper()}")
    
    print(f"\nRecommended Models:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.name}")
        print(f"   Priority: {rec.priority}")
        print(f"   Reason: {rec.reason}")
        print(f"   Description: {rec.description}")
    

def example_3_high_dimensional():
    """Example 3: High-dimensional dataset - demonstrates regularization focus"""
    print("\n" + "="*80)
    print("EXAMPLE 3: High-Dimensional Dataset (Gene Expression)")
    print("="*80)
    
    # Create a high-dimensional synthetic dataset
    np.random.seed(42)
    n_samples = 800
    n_features = 150  # More features than typical
    
    print(f"Generating high-dimensional data...")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Feature/Sample Ratio: {n_features/n_samples:.2f}")
    
    # Generate random feature data
    feature_data = np.random.randn(n_samples, n_features)
    feature_names = [f'gene_{i}' for i in range(n_features)]
    
    data = pd.DataFrame(feature_data, columns=feature_names)
    
    # Create target with some relationship to features
    weights = np.random.randn(n_features)
    weights[50:] = 0  # Only first 50 features are important
    linear_combo = np.dot(feature_data, weights)
    data['disease'] = (linear_combo > np.median(linear_combo)).astype(int)
    
    # Extract metadata and show selection
    loader = DataLoader()
    X, y = loader.load_from_dataframe(data, 'disease')
    
    metadata_extractor = MetadataExtractor()
    metadata = metadata_extractor.extract(X, y)
    
    task_detector = TaskDetector()
    task_type, _, _ = task_detector.detect(y)
    
    selector = RuleBasedModelSelector()
    recommendations = selector.select_models(task_type, metadata, max_models=3)
    
    print(f"\n{'─'*80}")
    print("Rule-Based Selection for High-Dimensional Data:")
    print(f"{'─'*80}")
    
    summary = selector.get_selection_summary()
    print(f"\nDataset Analysis:")
    print(f"  Complexity: {summary['dataset_complexity'].upper()}")
    print(f"  Reason: High feature-to-sample ratio ({n_features}/{n_samples} = {n_features/n_samples:.2f})")
    
    print(f"\nRecommended Models (Focus on Regularization):")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.name}")
        print(f"   Reason: {rec.reason}")
        has_regularization = 'L1' in rec.description or 'L2' in rec.description or 'regularization' in rec.description.lower()
        print(f"   Has Regularization: {'Yes ✓' if has_regularization else 'No'}")
    

def example_4_regression():
    """Example 4: Regression task - demonstrates regression model selection"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Regression Task (House Price Prediction)")
    print("="*80)
    
    # Create regression dataset
    np.random.seed(42)
    n_samples = 5000
    
    data = pd.DataFrame({
        'sqft': np.random.randint(500, 5000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.uniform(1, 4, n_samples),
        'age_years': np.random.randint(0, 100, n_samples),
        'location_score': np.random.uniform(1, 10, n_samples),
        'school_rating': np.random.uniform(1, 10, n_samples),
        'crime_rate': np.random.uniform(0, 100, n_samples),
        'neighborhood': np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples),
    })
    
    # Generate price with some relationship to features
    data['price'] = (
        data['sqft'] * 200 + 
        data['bedrooms'] * 10000 + 
        data['bathrooms'] * 15000 + 
        data['location_score'] * 50000 +
        data['school_rating'] * 30000 -
        data['age_years'] * 500 -
        data['crime_rate'] * 1000 +
        np.random.normal(0, 50000, n_samples)
    )
    
    print(f"\nDataset Info:")
    print(f"  Samples: {len(data):,}")
    print(f"  Features: {len(data.columns) - 1}")
    print(f"  Target: price (continuous)")
    print(f"  Price Range: ${data['price'].min():,.0f} - ${data['price'].max():,.0f}")
    
    # Extract metadata and show selection
    loader = DataLoader()
    X, y = loader.load_from_dataframe(data, 'price')
    
    metadata_extractor = MetadataExtractor()
    metadata = metadata_extractor.extract(X, y)
    
    task_detector = TaskDetector()
    task_type, confidence, reason = task_detector.detect(y)
    
    print(f"\nTask Detection:")
    print(f"  Type: {task_type.value.upper()}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Reason: {reason}")
    
    selector = RuleBasedModelSelector()
    recommendations = selector.select_models(task_type, metadata, max_models=4)
    
    print(f"\n{'─'*80}")
    print("Regression Model Recommendations:")
    print(f"{'─'*80}")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.name}")
        print(f"   Priority: {rec.priority}")
        print(f"   Reason: {rec.reason}")
        print(f"   Hyperparameters to tune: {len(rec.hyperparameters)}")


def comparison_with_legacy():
    """Compare rule-based selection vs. legacy approach"""
    print("\n" + "="*80)
    print("COMPARISON: Rule-Based vs. Legacy Model Selection")
    print("="*80)
    
    # Create a medium dataset
    np.random.seed(42)
    n_samples = 8000
    
    data = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'feature_4': np.random.randn(n_samples),
        'feature_5': np.random.randn(n_samples),
        'category_A': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'category_B': np.random.choice(['P', 'Q'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    loader = DataLoader()
    X, y = loader.load_from_dataframe(data, 'target')
    
    metadata_extractor = MetadataExtractor()
    metadata = metadata_extractor.extract(X, y)
    
    task_detector = TaskDetector()
    task_type, _, _ = task_detector.detect(y)
    
    print(f"\nDataset: {n_samples} samples, {len(data.columns)-1} features")
    
    # Rule-based selection
    print(f"\n{'─'*80}")
    print("Rule-Based Selection:")
    print(f"{'─'*80}")
    
    selector = RuleBasedModelSelector()
    rule_based_models = selector.select_models(task_type, metadata, max_models=5)
    
    for i, rec in enumerate(rule_based_models, 1):
        print(f"{i}. {rec.name} - {rec.reason}")
    
    # Legacy would try all models
    print(f"\n{'─'*80}")
    print("Legacy Selection (would try all models):")
    print(f"{'─'*80}")
    
    all_models = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Decision Tree",
        "SVM",
        "Naive Bayes",
        "KNN",
        "... (potentially more)"
    ]
    
    for i, model in enumerate(all_models, 1):
        print(f"{i}. {model}")
    
    print(f"\n{'─'*80}")
    print("Efficiency Comparison:")
    print(f"{'─'*80}")
    print(f"Rule-Based: {len(rule_based_models)} models (targeted)")
    print(f"Legacy: {len(all_models)}+ models (exhaustive)")
    print(f"Time Savings: ~{(1 - len(rule_based_models)/len(all_models))*100:.0f}%")
    print(f"Focus: Intelligent selection based on data characteristics")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RULE-BASED MODEL SELECTION - EXAMPLES")
    print("="*80)
    
    print("\nThese examples demonstrate how the rule-based model selector")
    print("intelligently chooses models based on dataset characteristics.")
    
    # Run examples
    example_2_large_dataset()
    example_3_high_dimensional()
    example_4_regression()
    comparison_with_legacy()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nTo run the full pipeline with rule-based selection:")
    print("  python examples/sample_usage.py")
    print("\nTo test the model selector:")
    print("  python tests/test_model_selector.py")
