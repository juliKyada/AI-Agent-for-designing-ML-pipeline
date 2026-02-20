"""
Pipeline generator for creating candidate ML pipelines
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from typing import List, Dict, Any
from src.detection.task_detector import TaskType
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()
config = get_config()


def _gpu_kwargs_xgboost():
    """Return kwargs for XGBoost to use GPU if enabled in config."""
    if not config.get("training.use_gpu", False):
        return {}
    device = config.get("training.device", "cuda")
    try:
        return {"device": device, "tree_method": "hist"}
    except Exception:
        return {}


def _gpu_kwargs_lightgbm():
    """Return kwargs for LightGBM to use GPU if enabled in config."""
    if not config.get("training.use_gpu", False):
        return {}
    device = config.get("training.device", "cuda")
    try:
        return {"device": device}
    except Exception:
        return {}


class PipelineGenerator:
    """Generates candidate ML pipelines based on task type and data characteristics"""
    
    def __init__(self):
        """Initialize PipelineGenerator"""
        self.pipelines = []
        self.preprocessors = []
    
    def generate(self, task_type: TaskType, metadata: Dict[str, Any], n_pipelines: int = None) -> List[Dict]:
        """
        Generate candidate pipelines
        
        Args:
            task_type: Type of ML task (classification or regression)
            metadata: Dataset metadata from MetadataExtractor
            n_pipelines: Number of pipelines to generate (default from config)
            
        Returns:
            List of pipeline configurations
        """
        if n_pipelines is None:
            n_pipelines = config.get('pipeline.n_candidate_pipelines', 5)
        
        logger.info(f"Generating {n_pipelines} candidate pipelines for {task_type.value}")
        if config.get("training.use_gpu", False):
            device = config.get("training.device", "cuda")
            logger.info(f"  GPU enabled for XGBoost and LightGBM (device={device})")
        
        # Get feature information
        numerical_features = metadata['features']['numerical']
        categorical_features = metadata['features']['categorical']
        
        # Create preprocessor
        preprocessor = self._create_preprocessor(numerical_features, categorical_features)
        
        # Get model configurations
        if task_type == TaskType.CLASSIFICATION:
            model_configs = self._get_classification_models()
        else:
            model_configs = self._get_regression_models()
        
        # Generate pipelines
        self.pipelines = []
        for i, model_config in enumerate(model_configs[:n_pipelines]):
            pipeline_dict = {
                'id': i,
                'name': model_config['name'],
                'preprocessor': preprocessor,
                'model': model_config['model'],
                'hyperparameters': model_config['hyperparameters'],
                'description': model_config['description'],
                'pipeline': None  # Will be set when pipeline is built
            }
            
            # Build sklearn pipeline
            pipeline_dict['pipeline'] = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model_config['model'])
            ])
            
            self.pipelines.append(pipeline_dict)
            logger.info(f"  Pipeline {i}: {model_config['name']}")
        
        logger.info(f"Generated {len(self.pipelines)} pipelines")
        return self.pipelines
    
    def _create_preprocessor(self, numerical_features: List[str], categorical_features: List[str]):
        """
        Create a preprocessor for the pipeline
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            
        Returns:
            ColumnTransformer for preprocessing
        """
        transformers = []
        
        # Numerical features preprocessing
        if numerical_features:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, numerical_features))
        
        # Categorical features preprocessing
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        if not transformers:
            # No features to transform
            return None
        
        # Create preprocessor with get_feature_names_out for proper feature naming
        preprocessor = ColumnTransformer(
            transformers=transformers,
            verbose_feature_names_out=True
        )
        return preprocessor
    
    def _get_classification_models(self) -> List[Dict]:
        """Get classification model configurations"""
        models = [
            {
                'name': 'Logistic Regression',
                'model': LogisticRegression(random_state=config.get('random_seed', 42), max_iter=1000),
                'hyperparameters': {
                    'model__C': [0.01, 0.1, 1.0, 10.0],
                    'model__penalty': ['l2'],
                    'model__solver': ['lbfgs']
                },
                'description': 'Simple linear model for binary and multiclass classification'
            },
            {
                'name': 'Random Forest',
                'model': RandomForestClassifier(random_state=config.get('random_seed', 42), n_jobs=-1),
                'hyperparameters': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5],
                    'model__min_samples_leaf': [1, 2]
                },
                'description': 'Ensemble of decision trees with bagging'
            },
            {
                'name': 'XGBoost',
                'model': XGBClassifier(
                    random_state=config.get('random_seed', 42),
                    n_jobs=-1,
                    verbosity=0,
                    **_gpu_kwargs_xgboost(),
                ),
                'hyperparameters': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1, 0.3],
                    'model__subsample': [0.8, 1.0]
                },
                'description': 'Gradient boosting with regularization'
            },
            {
                'name': 'LightGBM',
                'model': LGBMClassifier(
                    random_state=config.get('random_seed', 42),
                    n_jobs=-1,
                    verbose=-1,
                    **_gpu_kwargs_lightgbm(),
                ),
                'hyperparameters': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [5, 10, -1],
                    'model__learning_rate': [0.01, 0.1],
                    'model__num_leaves': [31, 50]
                },
                'description': 'Fast gradient boosting framework'
            },
            {
                'name': 'Decision Tree',
                'model': DecisionTreeClassifier(random_state=config.get('random_seed', 42)),
                'hyperparameters': {
                    'model__max_depth': [5, 10, 15, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__criterion': ['gini', 'entropy']
                },
                'description': 'Single decision tree (baseline model)'
            }
        ]
        
        return models
    
    def _get_regression_models(self) -> List[Dict]:
        """Get regression model configurations"""
        models = [
            {
                'name': 'Linear Regression',
                'model': LinearRegression(n_jobs=-1),
                'hyperparameters': {},
                'description': 'Simple linear regression (baseline)'
            },
            {
                'name': 'Ridge Regression',
                'model': Ridge(random_state=config.get('random_seed', 42)),
                'hyperparameters': {
                    'model__alpha': [0.1, 1.0, 10.0, 100.0]
                },
                'description': 'Linear regression with L2 regularization'
            },
            {
                'name': 'Random Forest',
                'model': RandomForestRegressor(random_state=config.get('random_seed', 42), n_jobs=-1),
                'hyperparameters': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5],
                    'model__min_samples_leaf': [1, 2]
                },
                'description': 'Ensemble of decision trees with bagging'
            },
            {
                'name': 'XGBoost',
                'model': XGBRegressor(
                    random_state=config.get('random_seed', 42),
                    n_jobs=-1,
                    verbosity=0,
                    **_gpu_kwargs_xgboost(),
                ),
                'hyperparameters': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1, 0.3],
                    'model__subsample': [0.8, 1.0]
                },
                'description': 'Gradient boosting with regularization'
            },
            {
                'name': 'LightGBM',
                'model': LGBMRegressor(
                    random_state=config.get('random_seed', 42),
                    n_jobs=-1,
                    verbose=-1,
                    **_gpu_kwargs_lightgbm(),
                ),
                'hyperparameters': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [5, 10, -1],
                    'model__learning_rate': [0.01, 0.1],
                    'model__num_leaves': [31, 50]
                },
                'description': 'Fast gradient boosting framework'
            }
        ]
        
        return models
    
    def get_pipelines(self) -> List[Dict]:
        """Get generated pipelines"""
        return self.pipelines
