"""
Rule-based optimized model selection based on dataset metadata
This module implements intelligent model selection rules to choose the most
appropriate models for a given dataset based on its characteristics.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from src.detection.task_detector import TaskType
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()
config = get_config()


class DatasetSize(Enum):
    """Dataset size categories"""
    TINY = "tiny"           # < 1000 samples
    SMALL = "small"         # 1000 - 10000
    MEDIUM = "medium"       # 10000 - 100000
    LARGE = "large"         # 100000 - 1000000
    HUGE = "huge"           # > 1000000


class DatasetComplexity(Enum):
    """Dataset complexity levels"""
    SIMPLE = "simple"       # Few features, linear patterns
    MODERATE = "moderate"   # Moderate features, some non-linearity
    COMPLEX = "complex"     # Many features, high non-linearity


@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning"""
    name: str
    model_class: Any
    priority: int  # Lower is higher priority
    reason: str
    hyperparameters: Dict[str, Any]
    description: str


class RuleBasedModelSelector:
    """
    Intelligent model selector that uses dataset metadata to recommend
    the most appropriate models for a given problem.
    """
    
    def __init__(self):
        """Initialize the Rule-Based Model Selector"""
        self.metadata = None
        self.task_type = None
        self.dataset_size = None
        self.dataset_complexity = None
        self.recommendations = []
    
    def select_models(self, task_type: TaskType, metadata: Dict[str, Any], 
                     max_models: int = 5) -> List[ModelRecommendation]:
        """
        Select the most appropriate models based on metadata
        
        Args:
            task_type: Type of ML task (classification or regression)
            metadata: Dataset metadata from MetadataExtractor
            max_models: Maximum number of models to recommend
            
        Returns:
            List of ModelRecommendation objects, sorted by priority
        """
        logger.info("=" * 80)
        logger.info("Rule-Based Model Selection")
        logger.info("=" * 80)
        
        self.metadata = metadata
        self.task_type = task_type
        
        # Analyze dataset characteristics
        self.dataset_size = self._categorize_dataset_size()
        self.dataset_complexity = self._assess_complexity()
        
        logger.info(f"Task Type: {task_type.value}")
        logger.info(f"Dataset Size: {self.dataset_size.value} ({metadata['dataset']['n_samples']:,} samples)")
        logger.info(f"Feature Count: {metadata['dataset']['n_features']}")
        logger.info(f"Dataset Complexity: {self.dataset_complexity.value}")
        
        # Generate recommendations based on rules
        if task_type == TaskType.CLASSIFICATION:
            self.recommendations = self._recommend_classification_models()
        elif task_type == TaskType.REGRESSION:
            self.recommendations = self._recommend_regression_models()
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return []
        
        # Sort by priority and limit to max_models
        self.recommendations.sort(key=lambda x: x.priority)
        selected = self.recommendations[:max_models]
        
        logger.info(f"\nSelected {len(selected)} models:")
        for i, rec in enumerate(selected, 1):
            logger.info(f"{i}. {rec.name} (Priority: {rec.priority})")
            logger.info(f"   Reason: {rec.reason}")
        
        return selected
    
    def _categorize_dataset_size(self) -> DatasetSize:
        """Categorize dataset size"""
        n_samples = self.metadata['dataset']['n_samples']
        
        if n_samples < 1000:
            return DatasetSize.TINY
        elif n_samples < 10000:
            return DatasetSize.SMALL
        elif n_samples < 100000:
            return DatasetSize.MEDIUM
        elif n_samples < 1000000:
            return DatasetSize.LARGE
        else:
            return DatasetSize.HUGE
    
    def _assess_complexity(self) -> DatasetComplexity:
        """Assess dataset complexity"""
        n_features = self.metadata['dataset']['n_features']
        n_samples = self.metadata['dataset']['n_samples']
        n_categorical = self.metadata['features']['n_categorical']
        
        # Feature-to-sample ratio
        feature_ratio = n_features / n_samples if n_samples > 0 else 1
        
        # High dimensionality
        is_high_dim = n_features > 50
        
        # Many categorical features can increase complexity
        high_categorical_ratio = n_categorical / n_features > 0.5 if n_features > 0 else False
        
        # Complexity scoring
        complexity_score = 0
        
        if is_high_dim:
            complexity_score += 2
        if feature_ratio > 0.1:
            complexity_score += 2
        if high_categorical_ratio:
            complexity_score += 1
        if n_features > 20:
            complexity_score += 1
        
        # Classify complexity
        if complexity_score <= 2:
            return DatasetComplexity.SIMPLE
        elif complexity_score <= 4:
            return DatasetComplexity.MODERATE
        else:
            return DatasetComplexity.COMPLEX
    
    def _recommend_classification_models(self) -> List[ModelRecommendation]:
        """Recommend classification models based on dataset characteristics"""
        recommendations = []
        n_samples = self.metadata['dataset']['n_samples']
        n_features = self.metadata['dataset']['n_features']
        n_classes = self.metadata['target']['n_unique']
        
        # Check for class imbalance
        class_balance = self.metadata['target'].get('class_balance', {})
        is_imbalanced = False
        if class_balance:
            min_class_ratio = min(class_balance.values())
            is_imbalanced = min_class_ratio < 0.1
        
        # Rule 1: For tiny datasets, prefer simple models
        if self.dataset_size == DatasetSize.TINY:
            recommendations.append(ModelRecommendation(
                name="Logistic Regression",
                model_class=LogisticRegression,
                priority=1,
                reason="Small dataset - simple model to avoid overfitting",
                hyperparameters={
                    'model__C': [0.01, 0.1, 1.0, 10.0],
                    'model__penalty': ['l2'],
                    'model__max_iter': [1000]
                },
                description="Linear model with regularization"
            ))
            
            recommendations.append(ModelRecommendation(
                name="Decision Tree",
                model_class=DecisionTreeClassifier,
                priority=2,
                reason="Interpretable model for small datasets",
                hyperparameters={
                    'model__max_depth': [3, 5, 7],
                    'model__min_samples_split': [5, 10],
                    'model__min_samples_leaf': [2, 4]
                },
                description="Single decision tree with constraints"
            ))
        
        # Rule 2: For small to medium datasets, use ensemble methods
        elif self.dataset_size in [DatasetSize.SMALL, DatasetSize.MEDIUM]:
            recommendations.append(ModelRecommendation(
                name="Random Forest",
                model_class=RandomForestClassifier,
                priority=1,
                reason="Robust ensemble method for small-medium datasets",
                hyperparameters={
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5],
                    'model__min_samples_leaf': [1, 2]
                },
                description="Bagging ensemble of decision trees"
            ))
            
            recommendations.append(ModelRecommendation(
                name="XGBoost",
                model_class=XGBClassifier,
                priority=2,
                reason="Powerful gradient boosting for structured data",
                hyperparameters={
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1, 0.3],
                    'model__subsample': [0.8, 1.0]
                },
                description="Gradient boosting with regularization"
            ))
            
            # Add logistic regression as baseline
            recommendations.append(ModelRecommendation(
                name="Logistic Regression",
                model_class=LogisticRegression,
                priority=4,
                reason="Baseline linear model for comparison",
                hyperparameters={
                    'model__C': [0.01, 0.1, 1.0, 10.0],
                    'model__penalty': ['l2'],
                    'model__max_iter': [1000]
                },
                description="Linear baseline model"
            ))
        
        # Rule 3: For large datasets, use scalable algorithms
        elif self.dataset_size in [DatasetSize.LARGE, DatasetSize.HUGE]:
            recommendations.append(ModelRecommendation(
                name="LightGBM",
                model_class=LGBMClassifier,
                priority=1,
                reason="Fast and efficient for large datasets",
                hyperparameters={
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [5, 10, -1],
                    'model__learning_rate': [0.01, 0.1],
                    'model__num_leaves': [31, 50],
                    'model__verbosity': [-1]
                },
                description="Highly optimized gradient boosting"
            ))
            
            recommendations.append(ModelRecommendation(
                name="Logistic Regression",
                model_class=LogisticRegression,
                priority=2,
                reason="Scalable linear model for large data",
                hyperparameters={
                    'model__C': [0.1, 1.0, 10.0],
                    'model__penalty': ['l2'],
                    'model__solver': ['saga'],
                    'model__max_iter': [500]
                },
                description="Linear model with SAG solver for large data"
            ))
        
        # Rule 4: Binary vs Multiclass considerations
        if n_classes == 2:
            # Binary classification - SVM can work well
            if self.dataset_size in [DatasetSize.SMALL, DatasetSize.MEDIUM]:
                recommendations.append(ModelRecommendation(
                    name="SVM (RBF kernel)",
                    model_class=SVC,
                    priority=3,
                    reason="Binary classification with moderate data size",
                    hyperparameters={
                        'model__C': [0.1, 1.0, 10.0],
                        'model__kernel': ['rbf'],
                        'model__gamma': ['scale', 'auto']
                    },
                    description="Support Vector Machine with RBF kernel"
                ))
        
        # Rule 5: High-dimensional data
        if n_features > 50 or n_features / n_samples > 0.1:
            # Remove complex models, add regularized models
            recommendations = [r for r in recommendations 
                             if r.name not in ["Random Forest", "SVM (RBF kernel)"]]
            
            if not any(r.name == "Logistic Regression" for r in recommendations):
                recommendations.append(ModelRecommendation(
                    name="Logistic Regression",
                    model_class=LogisticRegression,
                    priority=1,
                    reason="High-dimensional data - regularized linear model",
                    hyperparameters={
                        'model__C': [0.001, 0.01, 0.1, 1.0],
                        'model__penalty': ['l1', 'l2'],
                        'model__solver': ['saga'],
                        'model__max_iter': [1000]
                    },
                    description="Linear model with L1/L2 regularization"
                ))
        
        # Rule 6: Imbalanced classes
        if is_imbalanced:
            logger.info("Class imbalance detected - adjusting recommendations")
            for rec in recommendations:
                if rec.name in ["Random Forest", "XGBoost", "LightGBM"]:
                    rec.hyperparameters['model__class_weight'] = ['balanced']
                    rec.reason += " (with class balancing)"
        
        # Rule 7: Simple linear patterns
        if self.dataset_complexity == DatasetComplexity.SIMPLE:
            # Boost priority of linear models
            for rec in recommendations:
                if rec.name == "Logistic Regression":
                    rec.priority = max(1, rec.priority - 1)
                    rec.reason += " (dataset shows simple patterns)"
        
        return recommendations
    
    def _recommend_regression_models(self) -> List[ModelRecommendation]:
        """Recommend regression models based on dataset characteristics"""
        recommendations = []
        n_samples = self.metadata['dataset']['n_samples']
        n_features = self.metadata['dataset']['n_features']
        
        # Rule 1: For tiny datasets, prefer simple models
        if self.dataset_size == DatasetSize.TINY:
            recommendations.append(ModelRecommendation(
                name="Linear Regression",
                model_class=LinearRegression,
                priority=1,
                reason="Small dataset - simple linear model",
                hyperparameters={},
                description="Ordinary least squares regression"
            ))
            
            recommendations.append(ModelRecommendation(
                name="Ridge Regression",
                model_class=Ridge,
                priority=2,
                reason="Regularization to prevent overfitting on small data",
                hyperparameters={
                    'model__alpha': [0.1, 1.0, 10.0, 100.0]
                },
                description="Linear regression with L2 regularization"
            ))
        
        # Rule 2: For small to medium datasets, use ensemble methods
        elif self.dataset_size in [DatasetSize.SMALL, DatasetSize.MEDIUM]:
            recommendations.append(ModelRecommendation(
                name="Random Forest",
                model_class=RandomForestRegressor,
                priority=1,
                reason="Robust ensemble method for small-medium datasets",
                hyperparameters={
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5],
                    'model__min_samples_leaf': [1, 2]
                },
                description="Bagging ensemble of decision trees"
            ))
            
            recommendations.append(ModelRecommendation(
                name="XGBoost",
                model_class=XGBRegressor,
                priority=2,
                reason="Powerful gradient boosting for structured data",
                hyperparameters={
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1, 0.3],
                    'model__subsample': [0.8, 1.0]
                },
                description="Gradient boosting with regularization"
            ))
            
            # Add linear baseline
            recommendations.append(ModelRecommendation(
                name="Ridge Regression",
                model_class=Ridge,
                priority=4,
                reason="Baseline linear model for comparison",
                hyperparameters={
                    'model__alpha': [0.1, 1.0, 10.0, 100.0]
                },
                description="Linear baseline with regularization"
            ))
        
        # Rule 3: For large datasets, use scalable algorithms
        elif self.dataset_size in [DatasetSize.LARGE, DatasetSize.HUGE]:
            recommendations.append(ModelRecommendation(
                name="LightGBM",
                model_class=LGBMRegressor,
                priority=1,
                reason="Fast and efficient for large datasets",
                hyperparameters={
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [5, 10, -1],
                    'model__learning_rate': [0.01, 0.1],
                    'model__num_leaves': [31, 50],
                    'model__verbosity': [-1]
                },
                description="Highly optimized gradient boosting"
            ))
            
            recommendations.append(ModelRecommendation(
                name="Ridge Regression",
                model_class=Ridge,
                priority=2,
                reason="Scalable linear model for large data",
                hyperparameters={
                    'model__alpha': [0.1, 1.0, 10.0],
                    'model__solver': ['saga']
                },
                description="Linear model with SAG solver"
            ))
        
        # Rule 4: High-dimensional data
        if n_features > 50 or n_features / n_samples > 0.1:
            recommendations = [r for r in recommendations 
                             if r.name not in ["Random Forest"]]
            
            # Add Lasso for feature selection
            recommendations.append(ModelRecommendation(
                name="Lasso Regression",
                model_class=Lasso,
                priority=1,
                reason="High-dimensional data - L1 regularization for feature selection",
                hyperparameters={
                    'model__alpha': [0.001, 0.01, 0.1, 1.0]
                },
                description="Linear regression with L1 regularization"
            ))
        
        # Rule 5: Complex non-linear patterns
        if self.dataset_complexity == DatasetComplexity.COMPLEX:
            # Boost priority of non-linear models
            for rec in recommendations:
                if rec.name in ["XGBoost", "LightGBM", "Random Forest"]:
                    rec.priority = max(1, rec.priority - 1)
                    rec.reason += " (handles complex patterns well)"
        
        # Rule 6: Simple linear patterns
        if self.dataset_complexity == DatasetComplexity.SIMPLE:
            # Boost priority of linear models
            for rec in recommendations:
                if "Regression" in rec.name and rec.name != "Decision Tree":
                    rec.priority = max(1, rec.priority - 1)
                    rec.reason += " (dataset shows linear patterns)"
        
        return recommendations
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the selection process
        
        Returns:
            Dictionary with selection details
        """
        return {
            'task_type': self.task_type.value if self.task_type else None,
            'dataset_size_category': self.dataset_size.value if self.dataset_size else None,
            'dataset_complexity': self.dataset_complexity.value if self.dataset_complexity else None,
            'n_samples': self.metadata['dataset']['n_samples'] if self.metadata else None,
            'n_features': self.metadata['dataset']['n_features'] if self.metadata else None,
            'selected_models': [
                {
                    'name': rec.name,
                    'priority': rec.priority,
                    'reason': rec.reason
                }
                for rec in self.recommendations
            ]
        }
