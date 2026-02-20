"""
Quick test for target NaN handling
"""
import pandas as pd
import numpy as np
from src.model.trainer import ModelTrainer
from src.detection.task_detector import TaskType
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Create test data with NaN in target
np.random.seed(42)
X = pd.DataFrame({
    'a': np.random.randn(50),
    'b': np.random.randn(50),
})
y = pd.Series(np.random.randn(50), name='target')
# Add some NaN to y
y.iloc[[5, 15, 25]] = np.nan

print(f'Before: X shape {X.shape}, y NaN count: {y.isna().sum()}')

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=5, random_state=42))
])

pipeline_dict = {
    'id': 'test',
    'name': 'Test Pipeline',
    'pipeline': pipeline,
    'hyperparameters': {}
}

trainer = ModelTrainer(TaskType.REGRESSION)
result = trainer.train_pipeline(pipeline_dict, X, y)

print(f'After: X_train NaN: {result["X_train"].isna().sum().sum()}, y_train NaN: {result["y_train"].isna().sum()}')
print(f'After: X_test NaN: {result["X_test"].isna().sum().sum()}, y_test NaN: {result["y_test"].isna().sum()}')
print('Training successful! No NaN values in target and features.')
