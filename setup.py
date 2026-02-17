# MetaFlow Setup Script
# Run this to verify your installation and setup

import sys
import subprocess

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required packages"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    import os
    
    dirs = ['data', 'models', 'logs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"  ✓ {dir_name}/")
    
    print("✓ Directories created")
    return True

def test_imports():
    """Test if all modules can be imported"""
    print("\nTesting imports...")
    
    try:
        from src.data import DataLoader, MetadataExtractor
        print("  ✓ Data module")
        
        from src.detection import TaskDetector, TaskType
        print("  ✓ Detection module")
        
        from src.pipeline import PipelineGenerator, PipelineOptimizer
        print("  ✓ Pipeline module")
        
        from src.model import ModelTrainer, ModelEvaluator
        print("  ✓ Model module")
        
        from src.agent import PipelineAgent
        print("  ✓ Agent module")
        
        from src.main import MetaFlowAgent
        print("  ✓ Main module")
        
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def run_quick_test():
    """Run a quick test with synthetic data"""
    print("\nRunning quick test...")
    
    try:
        from sklearn.datasets import make_classification
        import pandas as pd
        from src.main import MetaFlowAgent
        
        # Create small synthetic dataset
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, 
                                   n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        # Run MetaFlow
        print("  Running MetaFlow on synthetic data...")
        agent = MetaFlowAgent()
        results = agent.run(dataframe=df, target_column='target')
        
        if results['success']:
            print("  ✓ Test successful!")
            print(f"  Best model: {results['best_pipeline']['name']}")
            return True
        else:
            print("  ❌ Test failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Test error: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("           MetaFlow Setup & Verification")
    print("=" * 60)
    
    steps = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("Directories", create_directories),
        ("Imports", test_imports),
        ("Quick Test", run_quick_test)
    ]
    
    results = []
    for step_name, step_func in steps:
        print()
        success = step_func()
        results.append((step_name, success))
        if not success and step_name in ["Python Version", "Dependencies"]:
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please resolve the issue and try again.")
            return
    
    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print("=" * 60)
    
    all_passed = True
    for step_name, success in results:
        status = "✓" if success else "❌"
        print(f"  {status} {step_name}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 MetaFlow is ready to use!")
        print("\nNext steps:")
        print("  1. Read QUICKSTART.md for usage guide")
        print("  2. Run: python examples/sample_usage.py")
        print("  3. Try it on your own data!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print()

if __name__ == '__main__':
    main()
