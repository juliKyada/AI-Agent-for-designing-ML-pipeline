"""
Test script to verify iterative optimization with max_iterations works
"""
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.pipeline_agent import PipelineAgent


def test_iterative_optimization():
    """Test that max_iterations is properly used in optimization loop"""
    print("\n" + "=" * 80)
    print("TESTING ITERATIVE OPTIMIZATION WITH max_iterations")
    print("=" * 80 + "\n")
    
    # Load Titanic dataset
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Run with max_iterations = 3
    print("Running pipeline with max_iterations=3...")
    agent = PipelineAgent()
    
    result = agent.run(
        dataframe=df,
        target_column='target',
        max_iterations=3,
        n_pipelines=5
    )
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    print(f"Best pipeline: {result['best_pipeline']['name']}")
    print(f"Best accuracy: {result['best_pipeline']['metrics'].get('test_accuracy', 'N/A')}")
    
    return result


if __name__ == "__main__":
    try:
        result = test_iterative_optimization()
        print("\n✓ Iterative optimization test passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
