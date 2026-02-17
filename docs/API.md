# MetaFlow - API Reference

## Main Interface

### MetaFlowAgent

The primary interface for MetaFlow.

```python
from src.main import MetaFlowAgent

agent = MetaFlowAgent()
```

#### Methods

##### `run(dataset_path=None, dataframe=None, target_column=None, max_iterations=None)`

Run the complete automated ML pipeline design process.

**Parameters:**
- `dataset_path` (str, optional): Path to dataset file (CSV, Excel, Parquet)
- `dataframe` (pd.DataFrame, optional): Pandas DataFrame (alternative to dataset_path)
- `target_column` (str, optional): Name of target column (defaults to last column)
- `max_iterations` (int, optional): Maximum optimization iterations (overrides config)

**Returns:**
- `dict`: Results dictionary containing:
  - `success` (bool): Whether execution was successful
  - `task_type` (str): Detected task type ('classification' or 'regression')
  - `metadata` (dict): Dataset metadata
  - `best_pipeline` (dict): Best performing pipeline
  - `all_pipelines` (list): All evaluated pipelines
  - `improvement_plan` (dict): Optimization suggestions
  - `explanation` (str): Human-readable explanation
  - `evaluation_report` (str): Detailed evaluation report

**Example:**
```python
results = agent.run(dataset_path="data.csv", target_column="target")
```

##### `get_results()`

Get the complete results from the last run.

**Returns:**
- `dict`: Results dictionary

**Example:**
```python
results = agent.get_results()
print(results['explanation'])
```

##### `save_best_model(output_path)`

Save the best trained model to disk.

**Parameters:**
- `output_path` (str): Path where model should be saved

**Example:**
```python
agent.save_best_model("models/my_model.pkl")
```

##### `print_explanation()`

Print the human-readable explanation of results.

**Example:**
```python
agent.print_explanation()
```

##### `print_report()`

Print the detailed evaluation report.

**Example:**
```python
agent.print_report()
```

## Core Components

### DataLoader

Handles loading datasets from various formats.

```python
from src.data import DataLoader

loader = DataLoader()
X, y = loader.load("data.csv", target_column="target")
```

#### Methods

##### `load(file_path, target_column=None)`

Load dataset from file.

**Parameters:**
- `file_path` (str): Path to dataset file
- `target_column` (str, optional): Target column name

**Returns:**
- `tuple`: (X, y) - Features DataFrame and target Series

##### `load_from_dataframe(df, target_column)`

Load dataset from DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `target_column` (str): Target column name

**Returns:**
- `tuple`: (X, y) - Features DataFrame and target Series

##### `get_basic_info()`

Get basic dataset information.

**Returns:**
- `dict`: Dictionary with dataset info

### MetadataExtractor

Extracts and analyzes dataset metadata.

```python
from src.data import MetadataExtractor

extractor = MetadataExtractor()
metadata = extractor.extract(X, y)
```

#### Methods

##### `extract(X, y)`

Extract comprehensive metadata from dataset.

**Parameters:**
- `X` (pd.DataFrame): Features
- `y` (pd.Series): Target

**Returns:**
- `dict`: Metadata dictionary containing:
  - `dataset`: Basic dataset info
  - `features`: Feature information
  - `target`: Target variable info
  - `quality`: Data quality assessment
  - `statistics`: Statistical summaries

##### `get_summary()`

Get human-readable metadata summary.

**Returns:**
- `str`: Formatted summary string

### TaskDetector

Automatically detects ML task type.

```python
from src.detection import TaskDetector, TaskType

detector = TaskDetector()
task_type, confidence, reason = detector.detect(y)
```

#### Methods

##### `detect(y, metadata=None)`

Detect task type from target variable.

**Parameters:**
- `y` (pd.Series): Target variable
- `metadata` (dict, optional): Dataset metadata

**Returns:**
- `tuple`: (TaskType, confidence, reason)
  - `TaskType`: Enum value (CLASSIFICATION or REGRESSION)
  - `confidence` (float): Confidence score (0-1)
  - `reason` (str): Explanation for detection

##### `get_task_info()`

Get detailed task information.

**Returns:**
- `dict`: Task information dictionary

##### `is_classification()`

Check if detected task is classification.

**Returns:**
- `bool`: True if classification

##### `is_regression()`

Check if detected task is regression.

**Returns:**
- `bool`: True if regression

### PipelineGenerator

Generates candidate ML pipelines.

```python
from src.pipeline import PipelineGenerator
from src.detection import TaskType

generator = PipelineGenerator()
pipelines = generator.generate(TaskType.CLASSIFICATION, metadata)
```

#### Methods

##### `generate(task_type, metadata, n_pipelines=None)`

Generate candidate pipelines.

**Parameters:**
- `task_type` (TaskType): Type of ML task
- `metadata` (dict): Dataset metadata
- `n_pipelines` (int, optional): Number of pipelines to generate

**Returns:**
- `list`: List of pipeline configuration dictionaries

##### `get_pipelines()`

Get generated pipelines.

**Returns:**
- `list`: List of pipeline dictionaries

### ModelTrainer

Trains ML models with hyperparameter tuning.

```python
from src.model import ModelTrainer
from src.detection import TaskType

trainer = ModelTrainer(TaskType.CLASSIFICATION)
results = trainer.train_all_pipelines(pipelines, X, y)
```

#### Methods

##### `train_pipeline(pipeline_dict, X, y, tune_hyperparameters=True)`

Train a single pipeline.

**Parameters:**
- `pipeline_dict` (dict): Pipeline configuration
- `X` (pd.DataFrame): Features
- `y` (pd.Series): Target
- `tune_hyperparameters` (bool): Whether to tune hyperparameters

**Returns:**
- `dict`: Training results

##### `train_all_pipelines(pipelines, X, y)`

Train all pipelines.

**Parameters:**
- `pipelines` (list): List of pipeline configurations
- `X` (pd.DataFrame): Features
- `y` (pd.Series): Target

**Returns:**
- `list`: List of training results

### ModelEvaluator

Evaluates model performance.

```python
from src.model import ModelEvaluator
from src.detection import TaskType

evaluator = ModelEvaluator(TaskType.CLASSIFICATION)
evaluations = evaluator.evaluate_all(trained_models)
```

#### Methods

##### `evaluate(trained_model_result)`

Evaluate a trained model.

**Parameters:**
- `trained_model_result` (dict): Result from ModelTrainer

**Returns:**
- `dict`: Evaluation results

##### `evaluate_all(trained_models)`

Evaluate all trained models.

**Parameters:**
- `trained_models` (list): List of trained model results

**Returns:**
- `list`: List of evaluation results

##### `get_best_pipeline()`

Get the best performing pipeline.

**Returns:**
- `dict`: Best evaluation result

##### `get_pipelines_needing_improvement()`

Get pipelines with detected issues.

**Returns:**
- `list`: List of evaluations with issues

##### `generate_report()`

Generate comprehensive evaluation report.

**Returns:**
- `str`: Formatted report string

### PipelineOptimizer

Optimizes pipelines and suggests improvements.

```python
from src.pipeline import PipelineOptimizer
from src.detection import TaskType

optimizer = PipelineOptimizer(TaskType.CLASSIFICATION)
plan = optimizer.generate_improvement_plan(evaluations)
```

#### Methods

##### `optimize(evaluation, X, y)`

Optimize a single pipeline.

**Parameters:**
- `evaluation` (dict): Evaluation result
- `X` (pd.DataFrame): Features
- `y` (pd.Series): Target

**Returns:**
- `dict`: Optimization suggestions

##### `generate_improvement_plan(evaluations)`

Generate comprehensive improvement plan.

**Parameters:**
- `evaluations` (list): List of evaluation results

**Returns:**
- `dict`: Improvement plan

##### `should_continue_optimization(iteration, recent_improvements)`

Check if optimization should continue.

**Parameters:**
- `iteration` (int): Current iteration
- `recent_improvements` (list): Recent performance improvements

**Returns:**
- `bool`: Whether to continue

## Configuration

### get_config()

Get global configuration instance.

```python
from src.utils import get_config

config = get_config()
value = config.get('pipeline.max_iterations')
```

#### Methods

##### `get(key, default=None)`

Get configuration value by key.

**Parameters:**
- `key` (str): Configuration key (supports dot notation)
- `default`: Default value if key not found

**Returns:**
- Any: Configuration value

##### `to_dict()`

Get full configuration as dictionary.

**Returns:**
- `dict`: Complete configuration

## Logging

### setup_logger()

Configure the logger.

```python
from src.utils import setup_logger

setup_logger()
```

### get_logger()

Get configured logger instance.

```python
from src.utils import get_logger

logger = get_logger()
logger.info("Message")
```

## Enums

### TaskType

Enumeration of ML task types.

**Values:**
- `TaskType.CLASSIFICATION`: Classification task
- `TaskType.REGRESSION`: Regression task
- `TaskType.UNKNOWN`: Unknown task type

**Example:**
```python
from src.detection import TaskType

if task_type == TaskType.CLASSIFICATION:
    print("This is a classification problem")
```
