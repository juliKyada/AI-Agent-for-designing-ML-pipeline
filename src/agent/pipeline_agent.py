"""
AI Agent for orchestrating the entire ML pipeline automation process
"""
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any, Callable, List
from src.data import DataLoader, MetadataExtractor
from src.detection import TaskDetector, TaskType
from src.pipeline import PipelineGenerator, PipelineOptimizer
from src.model import ModelTrainer, ModelEvaluator
from src.utils.logger import get_logger, setup_logger
from src.utils.config import get_config

logger = get_logger()
config = get_config()


class PipelineAgent:
    """
    AI Agent that automatically designs, trains, and optimizes ML pipelines
    """
    
    def __init__(self):
        """Initialize the Pipeline Agent"""
        setup_logger()
        logger.info("=" * 80)
        logger.info("MetaFlow Pipeline Agent Initialized")
        logger.info("=" * 80)
        
        self.data_loader = DataLoader()
        self.metadata_extractor = MetadataExtractor()
        self.task_detector = TaskDetector()
        
        self.X = None
        self.y = None
        self.metadata = None
        self.task_type = None
        
        self.pipeline_generator = None
        self.model_trainer = None
        self.model_evaluator = None
        self.pipeline_optimizer = None
        
        self.results = {}

    def run(
        self,
        dataset_path: Union[str, Path] = None,
        dataframe: pd.DataFrame = None,
        target_column: str = None,
        max_iterations: int = None,
        n_pipelines: int = None,
        stop_callback: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete automated pipeline design process
        
        Args:
            dataset_path: Path to dataset file (CSV, Excel, Parquet)
            dataframe: Pandas DataFrame (alternative to dataset_path)
            target_column: Name of target column
            max_iterations: Maximum optimization iterations (overrides config)
            n_pipelines: Number of candidate pipelines to generate (overrides config)
            stop_callback: Optional callable that returns True when execution should stop
            
        Returns:
            Dictionary with final results including best pipeline and explanations
        """
        logger.info("Starting MetaFlow automated pipeline design...")
        
        try:
            # Step 1: Load Data
            self._load_data(dataset_path, dataframe, target_column)
            
            # Step 2: Extract Metadata
            self._extract_metadata()
            
            # Step 3: Detect Task Type
            self._detect_task()
            
            # Step 4: Generate Candidate Pipelines
            self._generate_pipelines(n_pipelines=n_pipelines)
            
            # Step 5: Train Models
            self._train_models(stop_callback=stop_callback)
            
            # Step 6: Evaluate Performance
            self._evaluate_models()
            
            # Step 7: Check for Issues and Optimize
            self.max_iterations_param = max_iterations  # Store for optimizer
            improvement_plan = self._check_and_plan_improvements()
            
            # Step 8: Get Best Pipeline
            best_pipeline = self._get_best_pipeline()
            
            # Step 9: Generate Explanation
            explanation = self._generate_explanation(best_pipeline, improvement_plan)
            
            # Step 10: Compile Final Results
            self.results = {
                'success': True,
                'task_type': self.task_type.value,
                'metadata': self.metadata,
                'preprocessing': self.model_trainer.get_preprocessing_report(),
                'training_warnings': self.model_trainer.get_training_warnings(),
                'best_pipeline': {
                    'name': best_pipeline['pipeline_name'],
                    'model': best_pipeline['trained_model_result']['trained_model'],
                    'metrics': best_pipeline['metrics'],
                    'issues': best_pipeline['issues']
                },
                'all_pipelines': self.model_evaluator.get_evaluations(),
                'improvement_plan': improvement_plan,
                'explanation': explanation,
                'evaluation_report': self.model_evaluator.generate_report()
            }
            
            logger.info("=" * 80)
            logger.info("MetaFlow pipeline design completed successfully!")
            logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            raise
    
    def _load_data(self, dataset_path, dataframe, target_column):
        """Step 1: Load dataset"""
        logger.info("")
        logger.info("STEP 1: Loading Dataset")
        logger.info("-" * 80)
        
        if dataset_path is not None:
            self.X, self.y = self.data_loader.load(dataset_path, target_column)
        elif dataframe is not None:
            if target_column is None:
                raise ValueError("target_column must be specified when using dataframe")
            self.X, self.y = self.data_loader.load_from_dataframe(dataframe, target_column)
        else:
            raise ValueError("Either dataset_path or dataframe must be provided")
        
        logger.info(f"[OK] Data loaded: {len(self.X)} samples, {len(self.X.columns)} features")
    
    def _extract_metadata(self):
        """Step 2: Extract metadata"""
        logger.info("")
        logger.info("STEP 2: Extracting Metadata")
        logger.info("-" * 80)
        
        self.metadata = self.metadata_extractor.extract(self.X, self.y)
        logger.info("[OK] Metadata extracted")
        logger.info(self.metadata_extractor.get_summary())
    
    def _detect_task(self):
        """Step 3: Detect task type"""
        logger.info("")
        logger.info("STEP 3: Detecting Task Type")
        logger.info("-" * 80)
        
        self.task_type, confidence, reason = self.task_detector.detect(self.y, self.metadata)
        logger.info(f"[OK] Task detected: {self.task_type.value.upper()}")
        logger.info(f"  Confidence: {confidence:.2%}")
        logger.info(f"  Reason: {reason}")
    
    def _generate_pipelines(self, n_pipelines: int = None):
        """Step 4: Generate candidate pipelines"""
        logger.info("")
        logger.info("STEP 4: Generating Candidate Pipelines")
        logger.info("-" * 80)
        
        self.pipeline_generator = PipelineGenerator()
        self.pipelines = self.pipeline_generator.generate(self.task_type, self.metadata, n_pipelines=n_pipelines)
        logger.info(f"[OK] Generated {len(self.pipelines)} candidate pipelines")
    
    def _train_models(self, stop_callback: Optional[Callable[[], bool]] = None):
        """Step 5: Train all models

        Args:
            stop_callback: Optional callable that returns True when training should stop
        """
        logger.info("")
        logger.info("STEP 5: Training Models")
        logger.info("-" * 80)
        
        self.model_trainer = ModelTrainer(self.task_type)
        self.trained_models = self.model_trainer.train_all_pipelines(
            self.pipelines,
            self.X,
            self.y,
            stop_callback=stop_callback,
        )
        logger.info(f"[OK] Trained {len(self.trained_models)} models")
    
    def _evaluate_models(self):
        """Step 6: Evaluate all models"""
        logger.info("")
        logger.info("STEP 6: Evaluating Performance")
        logger.info("-" * 80)
        
        self.model_evaluator = ModelEvaluator(self.task_type)
        self.evaluations = self.model_evaluator.evaluate_all(self.trained_models)
        logger.info(f"[OK] Evaluated {len(self.evaluations)} pipelines")
    
    def _check_and_plan_improvements(self):
        """Step 7: Check for issues and plan improvements with iterative optimization"""
        logger.info("")
        logger.info("STEP 7: Checking for Issues & Planning Improvements")
        logger.info("-" * 80)
        
        # Store max_iterations for use in optimization
        self.max_iterations = getattr(self, 'max_iterations_param', None) or 3
        self.pipeline_optimizer = PipelineOptimizer(self.task_type, max_iterations=self.max_iterations)
        
        # Get initial best score
        initial_best_score = max(
            e['metrics'].get('test_accuracy', e['metrics'].get('test_r2', 0)) 
            for e in self.evaluations
        ) if self.evaluations else 0
        
        recent_improvements = []
        improvement_plan = None
        iteration = 0
        
        # Iterative optimization loop
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"")
            logger.info(f"{'=' * 80}")
            logger.info(f"OPTIMIZATION ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'=' * 80}")
            
            # Generate improvement plan for current state
            improvement_plan = self.pipeline_optimizer.generate_improvement_plan(self.evaluations)
            
            if not improvement_plan['needs_improvement']:
                logger.info("[OK] All pipelines performing well - optimization complete")
                break
            
            # Apply improvements to problematic pipelines
            improved_pipeline_names = self._apply_improvements(improvement_plan)
            
            if not improved_pipeline_names:
                logger.info("No improvements could be applied")
                break
            
            # Re-train improved pipelines
            logger.info(f"Re-training {len(improved_pipeline_names)} improved pipelines...")
            self.model_trainer.retrain_pipelines(improved_pipeline_names, self.trained_models, self.X, self.y)
            
            # Re-evaluate all pipelines
            logger.info("Re-evaluating pipelines...")
            self.evaluations = self.model_evaluator.evaluate_all(self.trained_models)
            
            # Calculate improvement
            current_best_score = max(
                e['metrics'].get('test_accuracy', e['metrics'].get('test_r2', 0)) 
                for e in self.evaluations
            ) if self.evaluations else 0
            
            improvement = current_best_score - initial_best_score
            recent_improvements.append(improvement)
            
            logger.info(f"Iteration {iteration} Results:")
            logger.info(f"  Previous best score: {initial_best_score:.4f}")
            logger.info(f"  Current best score: {current_best_score:.4f}")
            logger.info(f"  Improvement: {improvement:+.4f}")
            
            # Check if should continue
            if not self.pipeline_optimizer.should_continue_optimization(iteration, recent_improvements):
                logger.info(f"Stopping optimization")
                break
            
            initial_best_score = current_best_score
        
        logger.info(f"")
        logger.info(f"Optimization completed after {iteration} iterations")
        
        return improvement_plan or self.pipeline_optimizer.generate_improvement_plan(self.evaluations)
    
    def _apply_improvements(self, improvement_plan: Dict) -> List[str]:
        """Apply improvements to problematic pipelines by adjusting hyperparameters"""
        improved_names = []
        
        if not improvement_plan['needs_improvement']:
            return improved_names
        
        for pipeline_improvement in improvement_plan['pipelines_to_improve']:
            pipeline_name = pipeline_improvement['pipeline_name']
            issues = pipeline_improvement['issues']
            
            # Find the trained model for this pipeline
            trained_model = None
            for tm in self.trained_models:
                if tm['pipeline_name'] == pipeline_name:
                    trained_model = tm
                    break
            
            if not trained_model:
                continue
            
            # Get model instance
            model = trained_model['trained_model']
            improvement_made = False
            
            # Apply improvements based on issues detected
            for issue in issues:
                if 'overfitting' in issue.lower():
                    improvement_made = self._adjust_for_overfitting(model, trained_model) or improvement_made
                elif 'underfitting' in issue.lower():
                    improvement_made = self._adjust_for_underfitting(model, trained_model) or improvement_made
                elif 'low performance' in issue.lower():
                    improvement_made = self._adjust_for_low_performance(model, trained_model) or improvement_made
            
            if improvement_made:
                improved_names.append(pipeline_name)
                logger.info(f"  Applied improvements to: {pipeline_name}")
        
        return improved_names
    
    def _adjust_for_overfitting(self, model, trained_model: Dict) -> bool:
        """Adjust hyperparameters to reduce overfitting"""
        try:
            # Try to access model pipeline
            if hasattr(model, 'named_steps'):
                ml_model = model.named_steps.get('model')
            else:
                ml_model = model
            
            if ml_model is None:
                return False
            
            # Adjust regularization based on model type
            if hasattr(ml_model, 'C'):  # Logistic Regression
                ml_model.C = max(0.001, ml_model.C * 0.5)  # Increase regularization
                logger.info(f"    Reduced C (increased regularization) to {ml_model.C}")
                return True
            elif hasattr(ml_model, 'max_depth') and hasattr(ml_model, 'min_samples_leaf'):  # Tree-based
                original_depth = ml_model.max_depth
                ml_model.max_depth = max(3, (ml_model.max_depth or 10) - 2) if ml_model.max_depth else 5
                ml_model.min_samples_leaf = min(ml_model.min_samples_leaf + 2, 10) if hasattr(ml_model, 'min_samples_leaf') else 5
                logger.info(f"    Reduced max_depth from {original_depth} to {ml_model.max_depth}")
                return True
            elif hasattr(ml_model, 'alpha'):  # Ridge/Lasso
                ml_model.alpha = ml_model.alpha * 2  # Increase regularization
                logger.info(f"    Increased alpha to {ml_model.alpha}")
                return True
        except Exception as e:
            logger.info(f"    Could not adjust hyperparameters: {str(e)}")
        
        return False
    
    def _adjust_for_underfitting(self, model, trained_model: Dict) -> bool:
        """Adjust hyperparameters to improve underfitting"""
        try:
            if hasattr(model, 'named_steps'):
                ml_model = model.named_steps.get('model')
            else:
                ml_model = model
            
            if ml_model is None:
                return False
            
            # Reduce regularization or increase complexity
            if hasattr(ml_model, 'C'):  # Logistic Regression
                ml_model.C = ml_model.C * 2  # Decrease regularization
                logger.info(f"    Increased C (decreased regularization) to {ml_model.C}")
                return True
            elif hasattr(ml_model, 'max_depth'):  # Tree-based
                original_depth = ml_model.max_depth
                ml_model.max_depth = (ml_model.max_depth or 10) + 2
                if hasattr(ml_model, 'min_samples_leaf'):
                    ml_model.min_samples_leaf = max(1, ml_model.min_samples_leaf - 1)
                logger.info(f"    Increased max_depth from {original_depth} to {ml_model.max_depth}")
                return True
            elif hasattr(ml_model, 'alpha'):  # Ridge/Lasso
                ml_model.alpha = max(0.0001, ml_model.alpha * 0.5)  # Decrease regularization
                logger.info(f"    Decreased alpha to {ml_model.alpha}")
                return True
        except Exception as e:
            logger.info(f"    Could not adjust hyperparameters: {str(e)}")
        
        return False
    
    def _adjust_for_low_performance(self, model, trained_model: Dict) -> bool:
        """Adjust hyperparameters to improve low performance"""
        try:
            if hasattr(model, 'named_steps'):
                ml_model = model.named_steps.get('model')
            else:
                ml_model = model
            
            if ml_model is None:
                return False
            
            # Try to improve performance by tuning key parameters
            if hasattr(ml_model, 'n_estimators'):  # Ensemble methods
                ml_model.n_estimators = min(ml_model.n_estimators + 50, 500)
                logger.info(f"    Increased n_estimators to {ml_model.n_estimators}")
                return True
            elif hasattr(ml_model, 'C'):  # Logistic Regression (try wider search)
                # Try a balanced C value
                ml_model.C = 1.0
                logger.info(f"    Reset C to balanced value {ml_model.C}")
                return True
        except Exception as e:
            logger.info(f"    Could not adjust hyperparameters: {str(e)}")
        
        return False
    
    def _get_best_pipeline(self):
        """Step 8: Get the best pipeline"""
        logger.info("")
        logger.info("STEP 8: Selecting Best Pipeline")
        logger.info("-" * 80)
        
        best = self.model_evaluator.get_best_pipeline()
        logger.info(f"[OK] Best pipeline: {best['pipeline_name']}")
        
        return best
    
    def _generate_explanation(self, best_pipeline, improvement_plan):
        """Step 9: Generate human-readable explanation"""
        logger.info("")
        logger.info("STEP 9: Generating Explanation")
        logger.info("-" * 80)
        
        explanation = self._create_detailed_explanation(best_pipeline, improvement_plan)
        logger.info("[OK] Explanation generated")
        
        return explanation
    
    def _create_detailed_explanation(self, best_pipeline, improvement_plan) -> str:
        """Create a detailed explanation of the results"""
        lines = []
        
        lines.append("=" * 80)
        lines.append("METAFLOW AUTOMATED ML PIPELINE - FINAL REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        def sep():
            lines.append("")
            lines.append("---")
            lines.append("")

        # 1. Dataset Overview: question on one line, answer on next line(s)
        lines.append("**1. DATASET OVERVIEW:**  ")
        ds_info = self.metadata['dataset']
        lines.append(f"   • Samples: {ds_info['n_samples']:,}")
        lines.append(f"   • Features: {ds_info['n_features']}")
        lines.append(f"   • Target: {ds_info['target_name']}")
        sep()
        
        # 2. Task Detection
        lines.append("**2. TASK DETECTION:**  ")
        lines.append(f"   • Task Type: {self.task_type.value.upper()}")
        lines.append(f"   • Detection Reason: {self.task_detector.reason}")
        sep()
        
        # 3. Pipelines Evaluated
        lines.append("**3. PIPELINES EVALUATED:**  ")
        lines.append(f"   • Total Pipelines: {len(self.evaluations)}")
        for eval_result in self.evaluations:
            name = eval_result['pipeline_name']
            if self.task_type == TaskType.CLASSIFICATION:
                score = eval_result['metrics']['test_accuracy']
                metric = "Accuracy"
            else:
                score = eval_result['metrics']['test_r2']
                metric = "R²"
            lines.append(f"   • {name}: {metric} = {score:.4f}")
        sep()
        
        # 4. Best Pipeline
        lines.append("**4. BEST PIPELINE:**  ")
        lines.append(f"   • Name: {best_pipeline['pipeline_name']}")
        metrics = best_pipeline['metrics']
        if self.task_type == TaskType.CLASSIFICATION:
            lines.append(f"   • Test Accuracy: {metrics['test_accuracy']:.4f}")
            lines.append(f"   • Test F1 Score: {metrics['test_f1']:.4f}")
            lines.append(f"   • Test Precision: {metrics['test_precision']:.4f}")
            lines.append(f"   • Test Recall: {metrics['test_recall']:.4f}")
        else:
            lines.append(f"   • Test R² Score: {metrics['test_r2']:.4f}")
            lines.append(f"   • Test RMSE: {metrics['test_rmse']:.4f}")
            lines.append(f"   • Test MAE: {metrics['test_mae']:.4f}")
        lines.append(f"   • CV Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        sep()
        
        # 5 & 6. Issues and Recommendations, or Status
        if best_pipeline['issues']:
            lines.append("**5. DETECTED ISSUES:**  ")
            for issue in best_pipeline['issues']:
                lines.append(f"   [!] {issue}")
            sep()
            
            lines.append("**6. RECOMMENDATIONS:**  ")
            if improvement_plan['needs_improvement']:
                if improvement_plan.get('overall_recommendations'):
                    for rec in improvement_plan['overall_recommendations']:
                        lines.append(f"   • {rec}")
                else:
                    lines.append("   • Consider hyperparameter tuning for further optimization")
                    lines.append("   • Review feature engineering opportunities")
            sep()
        else:
            lines.append("**5. STATUS:**  ")
            lines.append("   [OK] No significant issues detected")
            lines.append("   [OK] Pipeline meets performance criteria")
            sep()
        
        # 7. Conclusion
        lines.append("**7. CONCLUSION:**  ")
        lines.append(f"   The best performing pipeline is '{best_pipeline['pipeline_name']}'")
        lines.append(f"   for this {self.task_type.value} task.")
        if not best_pipeline['issues']:
            lines.append("   The model shows good performance without major issues.")
        else:
            lines.append("   Some improvements are recommended (see above).")
        sep()
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def get_results(self) -> Dict:
        """Get the complete results"""
        return self.results
    
    def save_best_model(self, output_path: Union[str, Path]):
        """
        Save the best model to disk
        
        Args:
            output_path: Path to save the model
        """
        import joblib
        
        if not self.results:
            raise ValueError("No results available. Run the agent first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.results['best_pipeline']['model'], output_path)
        logger.info(f"Best model saved to: {output_path}")
