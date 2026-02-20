"""
MetaFlow - Main Entry Point
AI-Powered ML Pipeline Automation
"""
import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent import PipelineAgent
from src.utils import setup_logger, get_logger

# Setup logger on import
setup_logger()
logger = get_logger()


class MetaFlowAgent:
    """
    MetaFlow AI Agent - Automatic ML Pipeline Designer
    
    This is the main interface for using MetaFlow.
    """
    
    def __init__(self):
        """Initialize MetaFlow Agent"""
        self.agent = PipelineAgent()
    
    def run(self, dataset_path=None, dataframe=None, target_column=None, max_iterations=None, n_pipelines=None):
        """
        Run automated ML pipeline design
        
        Args:
            dataset_path: Path to dataset file (CSV, Excel, Parquet)
            dataframe: Pandas DataFrame (alternative to dataset_path)
            target_column: Name of target column
            max_iterations: Maximum optimization iterations
            n_pipelines: Number of candidate pipelines to generate (overrides config)
            
        Returns:
            Dictionary with results including best pipeline and explanation
        """
        return self.agent.run(
            dataset_path=dataset_path,
            dataframe=dataframe,
            target_column=target_column,
            max_iterations=max_iterations,
            n_pipelines=n_pipelines
        )
    
    def get_results(self):
        """Get the complete results"""
        return self.agent.get_results()
    
    def save_best_model(self, output_path):
        """Save the best model to disk"""
        self.agent.save_best_model(output_path)
    
    def print_explanation(self):
        """Print the explanation of the final pipeline"""
        results = self.agent.get_results()
        if results and 'explanation' in results:
            print(results['explanation'])
        else:
            print("No results available yet. Run the agent first.")
    
    def print_report(self):
        """Print the full evaluation report"""
        results = self.agent.get_results()
        if results and 'evaluation_report' in results:
            print(results['evaluation_report'])
        else:
            print("No results available yet. Run the agent first.")


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MetaFlow - Automated ML Pipeline Designer')
    parser.add_argument('dataset', type=str, help='Path to dataset file')
    parser.add_argument('--target', type=str, default=None, help='Target column name (default: last column)')
    parser.add_argument('--output', type=str, default='models/best_model.pkl', help='Output path for best model')
    parser.add_argument('--max-iterations', type=int, default=None, help='Maximum optimization iterations')
    
    args = parser.parse_args()
    
    # Run MetaFlow
    agent = MetaFlowAgent()
    results = agent.run(
        dataset_path=args.dataset,
        target_column=args.target,
        max_iterations=args.max_iterations
    )
    
    # Print explanation
    print("\n")
    agent.print_explanation()
    
    # Save model
    agent.save_best_model(args.output)
    
    print(f"\n✓ Best model saved to: {args.output}")


if __name__ == '__main__':
    main()
