"""
Task detector for identifying ML task type (classification or regression)
"""
import pandas as pd
import numpy as np
from enum import Enum
from typing import Tuple
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()
config = get_config()


class TaskType(Enum):
    """Enumeration of ML task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class TaskDetector:
    """Detects whether a problem is classification or regression"""
    
    def __init__(self):
        """Initialize TaskDetector"""
        self.task_type = None
        self.confidence = 0.0
        self.reason = ""
    
    def detect(self, y: pd.Series, metadata: dict = None) -> Tuple[TaskType, float, str]:
        """
        Detect the task type based on target variable
        
        Args:
            y: Target variable
            metadata: Optional metadata from MetadataExtractor
            
        Returns:
            Tuple of (TaskType, confidence, reason)
        """
        logger.info("Detecting task type...")
        
        # Get configuration thresholds
        classification_threshold = config.get('detection.classification_threshold', 20)
        min_samples_per_class = config.get('detection.min_samples_per_class', 10)
        logger.info(f"  Thresholds: classification_threshold={classification_threshold}, min_samples_per_class={min_samples_per_class}")
        
        # Remove NaN values for analysis
        y_clean = y.dropna()
        
        if len(y_clean) == 0:
            self.task_type = TaskType.UNKNOWN
            self.confidence = 0.0
            self.reason = "Target variable is entirely missing"
            logger.warning(self.reason)
            return self.task_type, self.confidence, self.reason
        
        n_unique = y_clean.nunique()
        n_samples = len(y_clean)
        dtype = y_clean.dtype
        logger.info(f"  Target: dtype={dtype}, n_unique={n_unique}, n_samples={n_samples}")
        
        # Decision logic
        if pd.api.types.is_numeric_dtype(dtype):
            # Numeric target - could be either
            
            # Check if values are all integers
            is_all_integers = np.all(y_clean == y_clean.astype(int))
            
            if n_unique == 2:
                # Binary classification
                self.task_type = TaskType.CLASSIFICATION
                self.confidence = 0.95
                self.reason = f"Binary target with {n_unique} unique values"
            
            elif n_unique < classification_threshold and is_all_integers:
                # Multiclass classification
                value_counts = y_clean.value_counts()
                min_class_samples = value_counts.min()
                
                if min_class_samples >= min_samples_per_class:
                    self.task_type = TaskType.CLASSIFICATION
                    self.confidence = 0.90
                    self.reason = f"Integer target with {n_unique} unique values (multiclass)"
                else:
                    self.task_type = TaskType.CLASSIFICATION
                    self.confidence = 0.70
                    self.reason = f"Integer target with {n_unique} classes, but some classes have few samples"
            
            elif n_unique > n_samples * 0.5:
                # Many unique continuous values - regression
                self.task_type = TaskType.REGRESSION
                self.confidence = 0.95
                self.reason = f"Continuous numeric target with {n_unique} unique values ({n_unique/n_samples:.1%} of samples)"
            
            else:
                # Ambiguous case - use heuristic
                if is_all_integers and n_unique < 50:
                    self.task_type = TaskType.CLASSIFICATION
                    self.confidence = 0.60
                    self.reason = f"Integer target with {n_unique} unique values (borderline case)"
                else:
                    self.task_type = TaskType.REGRESSION
                    self.confidence = 0.70
                    self.reason = f"Numeric target with {n_unique} unique values (likely continuous)"
        
        else:
            # Non-numeric target - classification
            if n_unique == 2:
                self.task_type = TaskType.CLASSIFICATION
                self.confidence = 1.0
                self.reason = f"Binary classification with categories: {list(y_clean.unique())}"
            else:
                self.task_type = TaskType.CLASSIFICATION
                self.confidence = 1.0
                self.reason = f"Multiclass classification with {n_unique} classes"
        
        logger.info("Task detection result:")
        logger.info(f"  -> Task type: {self.task_type.value.upper()}")
        logger.info(f"  -> Confidence: {self.confidence:.2%}")
        logger.info(f"  -> Reason: {self.reason}")
        
        return self.task_type, self.confidence, self.reason
    
    def get_task_info(self) -> dict:
        """
        Get detailed information about the detected task
        
        Returns:
            Dictionary with task information
        """
        return {
            'task_type': self.task_type.value if self.task_type else None,
            'confidence': self.confidence,
            'reason': self.reason
        }
    
    def is_classification(self) -> bool:
        """Check if task is classification"""
        return self.task_type == TaskType.CLASSIFICATION
    
    def is_regression(self) -> bool:
        """Check if task is regression"""
        return self.task_type == TaskType.REGRESSION
