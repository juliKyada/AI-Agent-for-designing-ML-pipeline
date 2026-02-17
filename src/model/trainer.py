"""
Model trainer for training ML pipelines
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from typing import Dict, List, Any, Tuple
from src.detection.task_detector import TaskType
from src.utils.logger import get_logger
from src.utils.config import get_config

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
    
    def train_pipeline(self, pipeline_dict: Dict, X: pd.DataFrame, y: pd.Series,
                      tune_hyperparameters: bool = True) -> Dict:
        """
        Train a single pipeline
        
        Args:
            pipeline_dict: Pipeline configuration dictionary
            X: Features DataFrame
            y: Target Series
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training pipeline: {pipeline_dict['name']}")
        
        pipeline = pipeline_dict['pipeline']
        
        # Split data
        test_size = config.get('data.test_size', 0.2)
        random_seed = config.get('random_seed', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y if self._is_classification() else None
        )
        
        logger.info(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Hyperparameter tuning
        if tune_hyperparameters and pipeline_dict['hyperparameters']:
            logger.info("  Performing hyperparameter tuning...")
            pipeline = self._tune_hyperparameters(pipeline, pipeline_dict['hyperparameters'], X_train, y_train)
        else:
            logger.info("  Training with default parameters...")
            pipeline.fit(X_train, y_train)
        
        # Get predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate cross-validation score
        cv_folds = config.get('training.cv_folds', 5)
        scoring = self._get_scoring_metric()
        
        logger.info(f"  Computing {cv_folds}-fold cross-validation score...")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1)
        
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
        
        self.trained_models = []
        
        for i, pipeline_dict in enumerate(pipelines):
            logger.info(f"Pipeline {i+1}/{len(pipelines)}")
            try:
                result = self.train_pipeline(pipeline_dict, X, y)
            except Exception as e:
                logger.error(f"Error training {pipeline_dict['name']}: {str(e)}")
                continue
        
        logger.info(f"Successfully trained {len(self.trained_models)} pipelines")
        return self.trained_models
    
    def _tune_hyperparameters(self, pipeline, param_grid: Dict, X_train, y_train):
        """
        Tune hyperparameters using GridSearchCV
        
        Args:
            pipeline: sklearn Pipeline
            param_grid: Hyperparameter grid
            X_train: Training features
            y_train: Training target
            
        Returns:
            Best estimator
        """
        cv_folds = config.get('training.cv_folds', 5)
        scoring = self._get_scoring_metric()
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"    Best parameters: {grid_search.best_params_}")
        logger.info(f"    Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
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
