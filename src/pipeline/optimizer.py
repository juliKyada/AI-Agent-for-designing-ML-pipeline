"""
Pipeline optimizer for improving ML pipelines iteratively
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.detection.task_detector import TaskType
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()
config = get_config()


class PipelineOptimizer:
    """Optimizes pipelines by addressing detected issues"""
    
    def __init__(self, task_type: TaskType):
        """
        Initialize PipelineOptimizer
        
        Args:
            task_type: Type of ML task
        """
        self.task_type = task_type
        self.optimization_history = []
    
    def optimize(self, evaluation: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Optimize a pipeline based on evaluation results
        
        Args:
            evaluation: Evaluation result with detected issues
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with optimization suggestions
        """
        logger.info(f"Analyzing pipeline for optimization: {evaluation['pipeline_name']}")
        
        issues = evaluation['issues']
        metrics = evaluation['metrics']
        
        if not issues:
            logger.info("  No issues to optimize")
            return {
                'needs_optimization': False,
                'suggestions': [],
                'evaluation': evaluation
            }
        
        logger.info(f"  Found {len(issues)} issues to address")
        
        # Generate optimization suggestions
        suggestions = []
        
        for issue in issues:
            if 'overfitting' in issue.lower():
                suggestions.extend(self._suggest_overfitting_fixes(evaluation, X, y))
            elif 'underfitting' in issue.lower():
                suggestions.extend(self._suggest_underfitting_fixes(evaluation, X, y))
            elif 'low performance' in issue.lower():
                suggestions.extend(self._suggest_performance_improvements(evaluation, X, y))
            elif 'high variance' in issue.lower():
                suggestions.extend(self._suggest_variance_reduction(evaluation, X, y))
        
        # Remove duplicates
        suggestions = list(dict.fromkeys(suggestions))
        
        logger.info(f"  Generated {len(suggestions)} optimization suggestions")
        for i, suggestion in enumerate(suggestions, 1):
            logger.info(f"    {i}. {suggestion}")
        
        optimization_result = {
            'needs_optimization': True,
            'issues': issues,
            'suggestions': suggestions,
            'evaluation': evaluation
        }
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def _suggest_overfitting_fixes(self, evaluation: Dict, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Generate suggestions to fix overfitting"""
        suggestions = [
            "Increase regularization strength (L1/L2)",
            "Reduce model complexity (decrease max_depth for trees)",
            "Add dropout or early stopping",
            "Increase training data through data augmentation",
            "Reduce number of features using feature selection",
            "Use cross-validation to tune hyperparameters",
            "Apply ensemble methods with bagging"
        ]
        
        return suggestions[:3]  # Return top 3
    
    def _suggest_underfitting_fixes(self, evaluation: Dict, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Generate suggestions to fix underfitting"""
        suggestions = [
            "Increase model complexity (more layers, deeper trees)",
            "Add polynomial features or interaction terms",
            "Reduce regularization strength",
            "Train for more epochs/iterations",
            "Try more complex models (e.g., ensemble methods)",
            "Engineer additional features from domain knowledge",
            "Remove or reduce feature selection constraints"
        ]
        
        return suggestions[:3]  # Return top 3
    
    def _suggest_performance_improvements(self, evaluation: Dict, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Generate general performance improvement suggestions"""
        suggestions = [
            "Try different model algorithms (ensemble methods often perform better)",
            "Perform comprehensive hyperparameter tuning with more iterations",
            "Address class imbalance using SMOTE or class weights",
            "Improve feature engineering (create interaction terms, polynomial features)",
            "Handle outliers in the data",
            "Try different feature scaling methods (StandardScaler, MinMaxScaler, RobustScaler)",
            "Increase cross-validation folds for better validation"
        ]
        
        return suggestions[:4]  # Return top 4
    
    def _suggest_variance_reduction(self, evaluation: Dict, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Generate suggestions to reduce variance"""
        suggestions = [
            "Increase cross-validation folds for more stable estimates",
            "Use stratified sampling to ensure balanced splits",
            "Increase training data size",
            "Apply ensemble methods (bagging, boosting)",
            "Use more robust evaluation metrics"
        ]
        
        return suggestions[:2]  # Return top 2
    
    def should_continue_optimization(self, iteration: int, recent_improvements: List[float]) -> bool:
        """
        Determine if optimization should continue
        
        Args:
            iteration: Current iteration number
            recent_improvements: List of recent performance improvements
            
        Returns:
            Boolean indicating whether to continue
        """
        max_iterations = config.get('pipeline.max_iterations', 10)
        early_stopping = config.get('pipeline.early_stopping_rounds', 3)
        min_improvement = config.get('pipeline.min_improvement', 0.01)
        
        # Check max iterations
        if iteration >= max_iterations:
            logger.info(f"Reached maximum iterations ({max_iterations})")
            return False
        
        # Check early stopping
        if len(recent_improvements) >= early_stopping:
            recent = recent_improvements[-early_stopping:]
            if all(imp < min_improvement for imp in recent):
                logger.info(f"Early stopping: No significant improvement in last {early_stopping} rounds")
                return False
        
        return True
    
    def get_optimization_summary(self) -> str:
        """
        Get a summary of optimization history
        
        Returns:
            Formatted summary string
        """
        if not self.optimization_history:
            return "No optimization history available"
        
        summary = ["=" * 80]
        summary.append("OPTIMIZATION SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Total Optimizations: {len(self.optimization_history)}")
        summary.append("")
        
        for i, opt in enumerate(self.optimization_history, 1):
            summary.append(f"{i}. {opt['evaluation']['pipeline_name']}")
            summary.append(f"   Issues: {len(opt['issues'])}")
            summary.append(f"   Suggestions: {len(opt['suggestions'])}")
            summary.append("")
        
        return "\n".join(summary)
    
    def generate_improvement_plan(self, evaluations: List[Dict]) -> Dict:
        """
        Generate a comprehensive improvement plan for all pipelines
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Improvement plan dictionary
        """
        logger.info("Generating comprehensive improvement plan...")
        
        # Find pipelines needing improvement
        pipelines_with_issues = [e for e in evaluations if e['issues']]
        
        if not pipelines_with_issues:
            logger.info("All pipelines are performing well!")
            best = max(evaluations, key=lambda x: x['metrics'].get('test_accuracy', x['metrics'].get('test_r2', 0))) if evaluations else None
            return {
                'needs_improvement': False,
                'message': 'All pipelines meet performance criteria',
                'best_pipeline': best,
            }
        
        logger.info(f"Found {len(pipelines_with_issues)} pipelines needing improvement")
        
        # Generate optimization for each problematic pipeline
        improvement_plan = {
            'needs_improvement': True,
            'pipelines_to_improve': [],
            'overall_recommendations': []
        }
        
        for eval_result in pipelines_with_issues:
            # Skip actual optimization for now, just document
            improvement_plan['pipelines_to_improve'].append({
                'pipeline_name': eval_result['pipeline_name'],
                'issues': eval_result['issues'],
                'current_score': eval_result['metrics'].get('test_accuracy', 
                                                            eval_result['metrics'].get('test_r2', 0))
            })
        
        # Generate overall recommendations
        all_issues = [issue for e in pipelines_with_issues for issue in e['issues']]
        
        if any('overfitting' in issue.lower() for issue in all_issues):
            improvement_plan['overall_recommendations'].append(
                "Multiple models show overfitting - consider global regularization strategy"
            )
        
        if any('low performance' in issue.lower() for issue in all_issues):
            improvement_plan['overall_recommendations'].append(
                "Low overall performance - consider feature engineering or data quality improvements"
            )
        
        logger.info(f"Generated improvement plan with {len(improvement_plan['pipelines_to_improve'])} pipelines to improve")
        
        return improvement_plan
