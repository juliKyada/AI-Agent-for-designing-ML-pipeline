"""
Simple test to verify MetaFlow installation
"""
import sys

def test_imports():
    """Test all core imports"""
    print("Testing imports...")
    
    try:
        from src.main import MetaFlowAgent
        from src.data import DataLoader, MetadataExtractor
        from src.detection import TaskDetector, TaskType
        from src.pipeline import PipelineGenerator, PipelineOptimizer
        from src.model import ModelTrainer, ModelEvaluator
        from src.agent import PipelineAgent
        from src.utils import get_config, setup_logger
        
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from sklearn.datasets import make_classification
        import pandas as pd
        from src.main import MetaFlowAgent
        
        # Create test data
        X, y = make_classification(n_samples=50, n_features=3, n_informative=2,
                                   n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
        df['target'] = y
        
        # Test MetaFlow
        agent = MetaFlowAgent()
        results = agent.run(dataframe=df, target_column='target')
        
        # Verify results
        assert results['success'] == True
        assert 'best_pipeline' in results
        assert 'explanation' in results
        
        print("✓ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("MetaFlow Basic Test")
    print("=" * 60)
    
    test1 = test_imports()
    test2 = test_basic_functionality()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("✓ All tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)
