# MetaFlow Quick Start Guide

## Installation

1. **Clone or download the project**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Basic Usage

### Option 1: Python Script

```python
from src.main import MetaFlowAgent

# Initialize the agent
agent = MetaFlowAgent()

# Run on your dataset
results = agent.run(dataset_path="your_data.csv")

# Print explanation
agent.print_explanation()

# Save the best model
agent.save_best_model("models/my_model.pkl")
```

### Option 2: Command Line

```bash
python -m src.main your_dataset.csv --target target_column --output models/output.pkl
```

### Option 3: Using DataFrame

```python
import pandas as pd
from src.main import MetaFlowAgent

# Load your data
df = pd.read_csv("your_data.csv")

# Run MetaFlow
agent = MetaFlowAgent()
results = agent.run(dataframe=df, target_column="target")

# Get results
print(results['explanation'])
```

## Example Datasets

Run the examples to see MetaFlow in action:

```bash
python examples/sample_usage.py
```

This will demonstrate:
- ✅ Classification (Iris dataset)
- ✅ Regression (Diabetes dataset)
- ✅ Synthetic data
- ✅ Handling missing values

## What MetaFlow Does Automatically

1. **Loads your data** - Supports CSV, Excel, Parquet
2. **Analyzes metadata** - Features, types, missing values, statistics
3. **Detects task type** - Classification or Regression
4. **Generates pipelines** - Creates 5+ candidate ML pipelines
5. **Trains models** - Trains all pipelines with hyperparameter tuning
6. **Evaluates performance** - Comprehensive metrics and cross-validation
7. **Detects issues** - Identifies overfitting, underfitting, low scores
8. **Plans improvements** - Suggests optimizations
9. **Selects best pipeline** - Based on performance metrics
10. **Explains results** - Human-readable report

## Results Structure

```python
results = {
    'success': True,
    'task_type': 'classification',  # or 'regression'
    'metadata': {...},  # Dataset information
    'best_pipeline': {
        'name': 'XGBoost',
        'model': <trained_model>,
        'metrics': {...},
        'issues': [...]
    },
    'all_pipelines': [...],  # All evaluated pipelines
    'improvement_plan': {...},
    'explanation': "...",  # Human-readable explanation
    'evaluation_report': "..."  # Detailed report
}
```

## Configuration

Edit `config/config.yaml` to customize:

- Maximum iterations
- Cross-validation folds
- Performance thresholds
- Hyperparameter search space
- Logging settings

## Output

MetaFlow generates:
- **Best trained model** - Ready to use for predictions
- **Performance metrics** - Accuracy, F1, R², RMSE, etc.
- **Evaluation report** - Detailed comparison of all pipelines
- **Improvement suggestions** - How to enhance performance
- **Logs** - Stored in `logs/` directory

## Making Predictions

```python
# After training
results = agent.run(dataset_path="train.csv")
best_model = results['best_pipeline']['model']

# Load new data
import pandas as pd
new_data = pd.read_csv("new_data.csv")

# Make predictions
predictions = best_model.predict(new_data)
```

## Troubleshooting

**Issue:** Module not found errors
**Solution:** Make sure you're running from the project root directory

**Issue:** Missing dependencies
**Solution:** Run `pip install -r requirements.txt`

**Issue:** Out of memory
**Solution:** Reduce `n_candidate_pipelines` in config.yaml

**Issue:** Slow training
**Solution:** Reduce `hyperparameter_tuning.n_trials` in config.yaml

## Next Steps

1. Try it on your own data
2. Adjust configuration for your needs
3. Review the generated report
4. Fine-tune based on suggestions
5. Deploy your best model

## Need Help?

- Check the examples in `examples/`
- Review the configuration in `config/config.yaml`
- Read the code documentation
- Check logs in `logs/metaflow.log`
