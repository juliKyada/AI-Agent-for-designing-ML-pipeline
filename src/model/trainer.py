"""
Model trainer for training ML pipelines
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.base import clone
from typing import Dict, List, Any, Tuple
from src.detection.task_detector import TaskType
from src.utils.logger import get_logger
from src.utils.config import get_config
from src.data.preprocessor import DataPreprocessor

# Suppress sklearn feature name warnings (not an error, just informational)
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

logger = get_logger()
config = get_config()


class ModelTrainer:
    """Handles model training and hyperparameter tuning"""
    
    def __init__(self, task_type: TaskType):
        """
        Initialize ModelTrainer
        
        Args:
            task_type: Type of ML task
        """
        self.task_type = task_type
        self.trained_models = []
        self.preprocessor = DataPreprocessor(
            missing_threshold=config.get('data.max_missing_ratio', 0.5),
            imputation_strategy=config.get('data.imputation_strategy', 'auto')
        )
    
    def train_pipeline(self, pipeline_dict: Dict, X: pd.DataFrame, y: pd.Series,
                      tune_hyperparameters: bool = True) -> Dict:
        """
        Train a single pipeline (with preprocessing if not already done)
        
        Args:
            pipeline_dict: Pipeline configuration dictionary
            X: Features DataFrame
            y: Target Series
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training pipeline: {pipeline_dict['name']}")
        
        # Preprocess data if not already done (for standalone usage)
        logger.info("  Preprocessing data - handling missing values...")
        X_processed, y_processed = self.preprocessor.fit(X, y)
        
        logger.info(f"  Data shape after preprocessing: {X_processed.shape}")
        logger.info(f"  Remaining samples: {len(X_processed)}")
        
        return self._train_single_pipeline(pipeline_dict, X_processed, y_processed)
    
    def train_all_pipelines(self, pipelines: List[Dict], X: pd.DataFrame, y: pd.Series) -> List[Dict]:
        """
        Train all pipelines
        
        Args:
            pipelines: List of pipeline configurations
            X: Features DataFrame
            y: Target Series
            
        Returns:
            List of training results for all pipelines
        """
        logger.info(f"Training {len(pipelines)} pipelines...")
        
        # Preprocess data once for all pipelines (ensures consistency)
        logger.info("Preprocessing data once for all pipelines...")
        X_processed, y_processed = self.preprocessor.fit(X, y)
        logger.info(f"Data preprocessed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        
        self.trained_models = []
        
        for i, pipeline_dict in enumerate(pipelines):
            logger.info(f"Pipeline {i+1}/{len(pipelines)}")
            try:
                result = self._train_single_pipeline(pipeline_dict, X_processed, y_processed)
            except Exception as e:
                logger.error(f"Error training {pipeline_dict['name']}: {str(e)}")
                continue
        
        logger.info(f"Successfully trained {len(self.trained_models)} pipelines")
        return self.trained_models
    
    def _train_single_pipeline(self, pipeline_dict: Dict, X_processed: pd.DataFrame, y_processed: pd.Series) -> Dict:
        """
        Train a single pipeline with already-preprocessed data
        
        Args:
            pipeline_dict: Pipeline configuration dictionary
            X_processed: Preprocessed features DataFrame
            y_processed: Preprocessed target Series
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training pipeline: {pipeline_dict['name']}")
        
        pipeline = pipeline_dict['pipeline']
        
        # Ensure X_processed is a DataFrame (preserve column names)
        if not isinstance(X_processed, pd.DataFrame):
            logger.warning("X_processed is not a DataFrame, converting...")
            X_processed = pd.DataFrame(X_processed)
        
        # Split data (train_test_split preserves DataFrames)
        test_size = config.get('data.test_size', 0.2)
        random_seed = config.get('random_seed', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, random_state=random_seed, stratify=y_processed if self._is_classification() else None
        )
        
        # Ensure train/test data maintain DataFrame structure for proper feature names
        X_train = pd.DataFrame(X_train, columns=X_processed.columns) if not isinstance(X_train, pd.DataFrame) else X_train
        X_test = pd.DataFrame(X_test, columns=X_processed.columns) if not isinstance(X_test, pd.DataFrame) else X_test
        
        logger.info(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"  Features: {list(X_train.columns)}")
        
        # Hyperparameter tuning
        if pipeline_dict['hyperparameters']:
            logger.info("  Performing hyperparameter tuning...")
            pipeline = self._tune_hyperparameters(
                pipeline, pipeline_dict['hyperparameters'], X_train, y_train, pipeline_dict['name']
            )
        else:
            logger.info("  Training with default parameters...")
            pipeline.fit(X_train, y_train)
        
        # Get predictions (ensure X_train and X_test are DataFrames)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate cross-validation score (use n_jobs=1 for tree models to avoid hang/deadlock)
        cv_folds = config.get('training.cv_folds', 5)
        scoring = self._get_scoring_metric()
        cv_n_jobs = self._safe_n_jobs_for_cv(pipeline_dict['name'])
        logger.info(f"  Computing {cv_folds}-fold cross-validation score...")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=cv_n_jobs)
        
        result = {
            'pipeline_id': pipeline_dict['id'],
            'pipeline_name': pipeline_dict['name'],
            'pipeline': pipeline,
            'trained_model': pipeline,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': pipeline.named_steps['model'].get_params() if hasattr(pipeline, 'named_steps') else {}
        }
        
        logger.info(f"  Training complete. CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        
        self.trained_models.append(result)
        return result
    
    def _safe_n_jobs_for_cv(self, pipeline_name: str) -> int:
        """Use n_jobs=1 for tree/boosting models to avoid multiprocessing + multithreading deadlocks (e.g. stuck on last pipeline)."""
        if pipeline_name in ('LightGBM', 'XGBoost', 'Random Forest'):
            return 1
        return -1

    def _tune_hyperparameters(self, pipeline, param_grid: Dict, X_train, y_train, pipeline_name: str = ""):
        """
        Tune hyperparameters by iterating over the grid and logging each combination
        so progress is visible in Execution Logs (avoids appearing stuck).
        """
        cv_folds = config.get('training.cv_folds', 5)
        scoring = self._get_scoring_metric()
        n_jobs = self._safe_n_jobs_for_cv(pipeline_name)
        param_list = list(ParameterGrid(param_grid))
        n_combinations = len(param_list)
        if n_combinations == 0:
            pipeline.fit(X_train, y_train)
            return pipeline
        total_fits = n_combinations * cv_folds
        logger.info(f"    Grid search: {cv_folds} folds × {n_combinations} candidates = {total_fits} fits")
        best_score = -np.inf
        best_params = None
        best_estimator = None
        for i, params in enumerate(param_list):
            logger.info(f"    [{i+1}/{n_combinations}] Testing {params}...")
            candidate = clone(pipeline)
            candidate.set_params(**params)
            scores = cross_val_score(
                candidate, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=n_jobs
            )
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            logger.info(f"        -> CV score: {mean_score:.4f} (+/- {std_score:.4f})")
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_estimator = candidate
        logger.info(f"    Best parameters: {best_params}")
        logger.info(f"    Best CV score: {best_score:.4f}")
        best_estimator.fit(X_train, y_train)
        return best_estimator
    
    def _get_scoring_metric(self) -> str:
        """Get appropriate scoring metric for the task"""
        if self._is_classification():
            return 'accuracy'
        else:
            return 'r2'
    
    def _is_classification(self) -> bool:
        """Check if task is classification"""
        return self.task_type == TaskType.CLASSIFICATION
    
    def get_trained_models(self) -> List[Dict]:
        """Get all trained models"""
        return self.trained_models
    
    def get_preprocessing_report(self) -> Dict:
        """Get preprocessing report from last training"""
        if self.preprocessor:
            return {
                'removed_features': self.preprocessor.dropped_features,
                'imputation_values': self.preprocessor.imputation_values,
                'imputation_strategy': self.preprocessor.imputation_strategy,
            }
        return {}
