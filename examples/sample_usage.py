"""
MetaFlow - Example Usage Script

This script demonstrates how to use MetaFlow for automated ML pipeline design.
"""
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, make_classification
from src.main import MetaFlowAgent


def example_1_iris_classification():
    """Example 1: Iris Classification Dataset"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Iris Classification Dataset")
    print("=" * 80)
    
    # Load iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Save to CSV for demonstration
    df.to_csv('data/iris.csv', index=False)
    
    # Initialize MetaFlow agent
    agent = MetaFlowAgent()
    
    # Run automated pipeline design
    results = agent.run(
        dataset_path='data/iris.csv',
        target_column='target'
    )
    
    # Print explanation
    print("\n")
    agent.print_explanation()
    
    # Save best model
    agent.save_best_model('models/iris_best_model.pkl')
    
    print("\n✓ Example 1 completed!")
    
    return results


def example_2_diabetes_regression():
    """Example 2: Diabetes Regression Dataset"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Diabetes Regression Dataset")
    print("=" * 80)
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    
    # Save to CSV
    df.to_csv('data/diabetes.csv', index=False)
    
    # Initialize MetaFlow agent
    agent = MetaFlowAgent()
    
    # Run automated pipeline design
    results = agent.run(
        dataset_path='data/diabetes.csv',
        target_column='target'
    )
    
    # Print explanation
    print("\n")
    agent.print_explanation()
    
    # Save best model
    agent.save_best_model('models/diabetes_best_model.pkl')
    
    print("\n✓ Example 2 completed!")
    
    return results


def example_3_custom_dataframe():
    """Example 3: Custom DataFrame (Synthetic Classification)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Synthetic Classification Dataset")
    print("=" * 80)
    
    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Initialize MetaFlow agent
    agent = MetaFlowAgent()
    
    # Run with DataFrame directly (no file needed)
    results = agent.run(
        dataframe=df,
        target_column='target'
    )
    
    # Print explanation
    print("\n")
    agent.print_explanation()
    
    # Print full report
    print("\n")
    agent.print_report()
    
    # Save best model
    agent.save_best_model('models/synthetic_best_model.pkl')
    
    print("\n✓ Example 3 completed!")
    
    return results


def example_4_with_missing_values():
    """Example 4: Dataset with Missing Values"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Dataset with Missing Values")
    print("=" * 80)
    
    # Create dataset with missing values
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
    
    # Introduce missing values
    import numpy as np
    mask = np.random.random(df.shape) < 0.1  # 10% missing
    df = df.mask(mask)
    df['target'] = y
    
    # Initialize MetaFlow agent
    agent = MetaFlowAgent()
    
    # Run automated pipeline design
    results = agent.run(
        dataframe=df,
        target_column='target'
    )
    
    # Print explanation
    print("\n")
    agent.print_explanation()
    
    print("\n✓ Example 4 completed!")
    
    return results


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "METAFLOW - EXAMPLE USAGE" + " " * 34 + "║")
    print("║" + " " * 18 + "AI-Powered ML Pipeline Automation" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Create data directory
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        # Run examples
        print("\nRunning examples... (this may take a few minutes)\n")
        
        # Example 1: Classification
        example_1_iris_classification()
        
        # Example 2: Regression
        example_2_diabetes_regression()
        
        # Example 3: DataFrame input
        example_3_custom_dataframe()
        
        # Example 4: Missing values
        example_4_with_missing_values()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nCheck the 'models/' directory for saved models.")
        print("Check the 'logs/' directory for detailed logs.")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
