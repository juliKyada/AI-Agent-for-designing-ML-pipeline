"""
Test categorical feature encoding and feature name handling
"""
import pandas as pd
import numpy as np
from src.model.trainer import ModelTrainer
from src.detection.task_detector import TaskType
from src.pipeline.generator import PipelineGenerator
from src.data.metadata import MetadataExtractor

# Create test data with categorical features
np.random.seed(42)
X = pd.DataFrame({
    'numeric_1': np.random.randn(50),
    'numeric_2': np.random.randn(50),
    'category_1': np.random.choice(['A', 'B', 'C'], 50),
    'category_2': np.random.choice(['X', 'Y'], 50),
})
y = pd.Series(np.random.randint(0, 2, 50), name='target')

print('Test Data:')
print(f'  X shape: {X.shape}')
print(f'  Columns: {list(X.columns)}')
print(f'  Data types:')
for col in X.columns:
    print(f'    {col}: {X[col].dtype}')

# Extract metadata
extractor = MetadataExtractor()
metadata = extractor.extract(X, y)

print(f'\nMetadata:')
print(f'  Numerical features: {metadata["features"]["numerical"]}')
print(f'  Categorical features: {metadata["features"]["categorical"]}')

# Generate pipelines
generator = PipelineGenerator()
pipelines = generator.generate(TaskType.CLASSIFICATION, metadata, n_pipelines=1)

print(f'\nGenerated {len(pipelines)} pipeline')
print(f'  Pipeline: {pipelines[0]["name"]}')

# Train pipeline
trainer = ModelTrainer(TaskType.CLASSIFICATION)
result = trainer.train_pipeline(pipelines[0], X, y)

print(f'\nTraining successful!')
print(f'  CV Score: {result["cv_mean"]:.4f}')
print(f'  Test predictions shape: {result["y_test_pred"].shape}')
print(f'  No NaN in predictions: {not np.isnan(result["y_test_pred"]).any()}')
print('\nFeature encoding test PASSED!')
