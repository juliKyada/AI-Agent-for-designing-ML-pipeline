"""
Model evaluator for assessing pipeline performance
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report
)
from typing import Dict, List, Any
from src.detection.task_detector import TaskType
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()
config = get_config()


class ModelEvaluator:
    """Evaluates model performance and detects issues"""
    
    def __init__(self, task_type: TaskType):
        """
        Initialize ModelEvaluator
        
        Args:
            task_type: Type of ML task
        """
        self.task_type = task_type
        self.evaluations = []
    
    def evaluate(self, trained_model_result: Dict) -> Dict:
        """
        Evaluate a trained model
        
        Args:
            trained_model_result: Result dictionary from ModelTrainer
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating pipeline: {trained_model_result['pipeline_name']}")
        
        y_train = trained_model_result['y_train']
        y_test = trained_model_result['y_test']
        y_train_pred = trained_model_result['y_train_pred']
        y_test_pred = trained_model_result['y_test_pred']
        
        # Calculate metrics
        if self._is_classification():
            metrics = self._evaluate_classification(y_train, y_test, y_train_pred, y_test_pred)
        else:
            metrics = self._evaluate_regression(y_train, y_test, y_train_pred, y_test_pred)
        
        # Add cross-validation scores
        metrics['cv_mean'] = trained_model_result['cv_mean']
        metrics['cv_std'] = trained_model_result['cv_std']
        
        # Detect issues
        issues = self._detect_issues(metrics, y_train, y_test)
        
        evaluation = {
            'pipeline_id': trained_model_result['pipeline_id'],
            'pipeline_name': trained_model_result['pipeline_name'],
            'metrics': metrics,
            'issues': issues,
            'trained_model_result': trained_model_result
        }
        
        self.evaluations.append(evaluation)
        
        # Log summary
        self._log_evaluation_summary(evaluation)
        
        return evaluation
    
    def evaluate_all(self, trained_models: List[Dict]) -> List[Dict]:
        """
        Evaluate all trained models
        
        Args:
            trained_models: List of trained model results
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Evaluating {len(trained_models)} models...")
        
        self.evaluations = []
        for result in trained_models:
            self.evaluate(result)
        
        return self.evaluations
    
    def _evaluate_classification(self, y_train, y_test, y_train_pred, y_test_pred) -> Dict:
        """Evaluate classification model"""
        
        # Determine if binary or multiclass
        n_classes = len(np.unique(y_train))

        # Use binary averaging only for standard numeric labels where
        # sklearn's default pos_label=1 is valid; otherwise fall back to
        # weighted averaging which works for arbitrary labels (e.g. 'Y'/'N').
        unique_labels = np.unique(y_train)
        numeric_labels = np.issubdtype(unique_labels.dtype, np.number)

        if n_classes == 2 and numeric_labels:
            sorted_labels = np.sort(unique_labels)
            if np.array_equal(sorted_labels, np.array([0, 1])) or np.array_equal(sorted_labels, np.array([-1, 1])):
                average = 'binary'
            else:
                average = 'weighted'
        else:
            average = 'weighted'
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_precision': precision_score(y_train, y_train_pred, average=average, zero_division=0),
            'test_precision': precision_score(y_test, y_test_pred, average=average, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, average=average, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, average=average, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, average=average, zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, average=average, zero_division=0),
        }
        
        # Add ROC AUC only for binary classification with probability predictions
        try:
            if n_classes == 2:
                # For binary, we could add roc_auc if we have predict_proba
                pass
        except:
            pass
        
        return metrics
    
    def _evaluate_regression(self, y_train, y_test, y_train_pred, y_test_pred) -> Dict:
        """Evaluate regression model"""
        
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
        }
        
        # Add MAPE (Mean Absolute Percentage Error) if no zeros in y
        if not (y_train == 0).any() and not (y_test == 0).any():
            metrics['train_mape'] = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
            metrics['test_mape'] = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        
        return metrics
    
    def _detect_issues(self, metrics: Dict, y_train, y_test) -> List[str]:
        """
        Detect performance issues
        
        Args:
            metrics: Evaluation metrics
            y_train: Training target
            y_test: Test target
            
        Returns:
            List of detected issues
        """
        issues = []
        
        overfitting_threshold = config.get('thresholds.overfitting_threshold', 0.10)
        
        if self._is_classification():
            min_score = config.get('thresholds.min_score.classification', 0.70)
            primary_metric = 'test_accuracy'
            train_metric = 'train_accuracy'
            test_metric = 'test_accuracy'
        else:
            min_score = config.get('thresholds.min_score.regression', 0.60)
            primary_metric = 'test_r2'
            train_metric = 'train_r2'
            test_metric = 'test_r2'
        
        # Check for low performance
        if metrics[primary_metric] < min_score:
            issues.append(f"Low performance: {primary_metric}={metrics[primary_metric]:.4f} < {min_score}")
        
        # Check for overfitting
        train_score = metrics[train_metric]
        test_score = metrics[test_metric]
        gap = train_score - test_score
        
        if gap > overfitting_threshold:
            issues.append(f"Overfitting detected: train={train_score:.4f}, test={test_score:.4f}, gap={gap:.4f}")
        
        # Check for underfitting
        if train_score < min_score:
            issues.append(f"Underfitting detected: train score {train_score:.4f} < {min_score}")
        
        # Check for high variance in CV scores
        if 'cv_std' in metrics and metrics['cv_std'] > 0.15:
            issues.append(f"High variance in cross-validation: std={metrics['cv_std']:.4f}")
        
        return issues
    
    def _log_evaluation_summary(self, evaluation: Dict):
        """Log evaluation summary"""
        metrics = evaluation['metrics']
        issues = evaluation['issues']
        
        logger.info(f"  Evaluation results for {evaluation['pipeline_name']}:")
        
        if self._is_classification():
            logger.info(f"    Accuracy: train={metrics['train_accuracy']:.4f}, test={metrics['test_accuracy']:.4f}")
            logger.info(f"    F1 Score: train={metrics['train_f1']:.4f}, test={metrics['test_f1']:.4f}")
        else:
            logger.info(f"    R² Score: train={metrics['train_r2']:.4f}, test={metrics['test_r2']:.4f}")
            logger.info(f"    RMSE: train={metrics['train_rmse']:.4f}, test={metrics['test_rmse']:.4f}")
        
        logger.info(f"    CV Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        
        if issues:
            logger.warning(f"    Issues detected: {len(issues)}")
            for issue in issues:
                logger.warning(f"      - {issue}")
        else:
            logger.info("    No issues detected")
    
    def get_best_pipeline(self) -> Dict:
        """
        Get the best performing pipeline
        
        Returns:
            Best evaluation result
        """
        if not self.evaluations:
            raise ValueError("No evaluations available")
        
        # Sort by primary metric
        if self._is_classification():
            metric_key = 'test_accuracy'
        else:
            metric_key = 'test_r2'
        
        best = max(self.evaluations, key=lambda x: x['metrics'][metric_key])
        
        logger.info(f"Best pipeline: {best['pipeline_name']} ({metric_key}={best['metrics'][metric_key]:.4f})")
        
        return best
    
    def get_pipelines_needing_improvement(self) -> List[Dict]:
        """
        Get pipelines that need improvement
        
        Returns:
            List of evaluations with issues
        """
        return [e for e in self.evaluations if e['issues']]
    
    def _is_classification(self) -> bool:
        """Check if task is classification"""
        return self.task_type == TaskType.CLASSIFICATION
    
    def get_evaluations(self) -> List[Dict]:
        """Get all evaluations"""
        return self.evaluations
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive evaluation report
        
        Returns:
            Formatted report string
        """
        if not self.evaluations:
            return "No evaluations available"
        
        report = ["=" * 80]
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Task Type: {self.task_type.value.upper()}")
        report.append(f"Number of Pipelines: {len(self.evaluations)}")
        report.append("")
        
        # Sort by performance
        if self._is_classification():
            sorted_evals = sorted(self.evaluations, 
                                key=lambda x: x['metrics']['test_accuracy'], 
                                reverse=True)
            metric_name = "Accuracy"
        else:
            sorted_evals = sorted(self.evaluations, 
                                key=lambda x: x['metrics']['test_r2'], 
                                reverse=True)
            metric_name = "R²"
        
        for i, eval_result in enumerate(sorted_evals, 1):
            name = eval_result['pipeline_name']
            metrics = eval_result['metrics']
            issues = eval_result['issues']
            
            report.append(f"{i}. {name}")
            report.append("-" * 80)
            
            if self._is_classification():
                report.append(f"   Accuracy:  Train={metrics['train_accuracy']:.4f}, Test={metrics['test_accuracy']:.4f}")
                report.append(f"   F1 Score:  Train={metrics['train_f1']:.4f}, Test={metrics['test_f1']:.4f}")
                report.append(f"   Precision: Train={metrics['train_precision']:.4f}, Test={metrics['test_precision']:.4f}")
                report.append(f"   Recall:    Train={metrics['train_recall']:.4f}, Test={metrics['test_recall']:.4f}")
            else:
                report.append(f"   R² Score:  Train={metrics['train_r2']:.4f}, Test={metrics['test_r2']:.4f}")
                report.append(f"   RMSE:      Train={metrics['train_rmse']:.4f}, Test={metrics['test_rmse']:.4f}")
                report.append(f"   MAE:       Train={metrics['train_mae']:.4f}, Test={metrics['test_mae']:.4f}")
            
            report.append(f"   CV Score:  {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
            
            if issues:
                report.append(f"   ⚠ Issues ({len(issues)}):")
                for issue in issues:
                    report.append(f"     - {issue}")
            else:
                report.append("   ✓ No issues detected")
            
            report.append("")
        
        report.append("=" * 80)
        best = self.get_best_pipeline()
        report.append(f"BEST PIPELINE: {best['pipeline_name']}")
        report.append("=" * 80)
        
        return "\n".join(report)
