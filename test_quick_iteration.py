"""
Quick test to verify iterative optimization loop runs
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agent.pipeline_agent import PipelineAgent


def test_quick_iteration():
    """Quick test with small dataset"""
    print("\n" + "=" * 80)
    print("QUICK TEST: Iterative Optimization with max_iterations=2")
    print("=" * 80 + "\n")
    
    # Create a tiny dataset
    np.random.seed(42)
    X = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'f3': np.random.randn(100)
    })
    y = (X['f1'] + X['f2'] > 0).astype(int)
    df = X.copy()
    df['target'] = y
    
    # Run with max_iterations = 2
    print("Running pipeline with max_iterations=2 and 3 pipelines...")
    agent = PipelineAgent()
    
    result = agent.run(
        dataframe=df,
        target_column='target',
        max_iterations=2,
        n_pipelines=3
    )
    
    print("\n" + "=" * 80)
    print("✓ TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Best pipeline: {result['best_pipeline']['name']}")
    
    return True


if __name__ == "__main__":
    try:
        test_quick_iteration()
        print("\n[OK] Iterative optimization is working!")
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
