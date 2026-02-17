"""
MetaFlow Demo - Quick demonstration without requiring a dataset file
Run this to see MetaFlow in action!
"""
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.main import MetaFlowAgent
import pandas as pd
from sklearn.datasets import make_classification
import os

def run_demo():
    """Run a quick demo with synthetic data"""
    
    print("=" * 80)
    print("                    METAFLOW QUICK DEMO")
    print("         AI-Powered ML Pipeline Automation")
    print("=" * 80)
    print()
    print("Creating synthetic classification dataset...")
    
    # Create synthetic dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.1  # Add some noise
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"✓ Dataset created: {len(df)} samples, {len(feature_names)} features")
    print()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Save dataset for reference
    df.to_csv('data/demo_dataset.csv', index=False)
    print("✓ Dataset saved to: data/demo_dataset.csv")
    print()
    
    # Initialize MetaFlow
    print("Initializing MetaFlow AI Agent...")
    print()
    agent = MetaFlowAgent()
    
    # Run automated pipeline design
    print("Starting automated ML pipeline design...")
    print("(This will take 1-2 minutes)")
    print()
    
    results = agent.run(
        dataframe=df,
        target_column='target'
    )
    
    # Print results
    print()
    print("=" * 80)
    print("                         RESULTS")
    print("=" * 80)
    print()
    
    agent.print_explanation()
    
    # Save best model
    model_path = 'models/demo_best_model.pkl'
    agent.save_best_model(model_path)
    
    print()
    print("=" * 80)
    print(f"✓ Demo completed successfully!")
    print(f"✓ Best model saved to: {model_path}")
    print(f"✓ Dataset saved to: data/demo_dataset.csv")
    print(f"✓ Logs saved to: logs/metaflow.log")
    print()
    print("To use MetaFlow with your own data:")
    print("  python -m src.main your_data.csv --target your_target_column")
    print()
    print("Or in Python:")
    print("  from src.main import MetaFlowAgent")
    print("  agent = MetaFlowAgent()")
    print("  results = agent.run(dataset_path='your_data.csv')")
    print("=" * 80)

if __name__ == '__main__':
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
