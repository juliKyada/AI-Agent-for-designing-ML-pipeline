# MetaFlow - Architecture Documentation

## System Architecture

MetaFlow follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                    (MetaFlowAgent / CLI)                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      PIPELINE AGENT                          │
│                   (Orchestration Layer)                      │
└─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬────────────┘
      │     │     │     │     │     │     │     │
      ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
    ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
    │ 1 │ │ 2 │ │ 3 │ │ 4 │ │ 5 │ │ 6 │ │ 7 │ │ 8 │
    └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
     │     │     │     │     │     │     │     │
     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
   Load Extract Detect Gen  Train Eval Check Best
   Data Meta  Task  Pipe Model Perf Issues Pick
```

## Core Modules

### 1. Data Module (`src/data/`)

**Purpose:** Handle data loading and metadata extraction

**Components:**
- `DataLoader`: Loads datasets from various formats (CSV, Excel, Parquet)
- `MetadataExtractor`: Analyzes dataset characteristics

**Key Features:**
- Support for multiple file formats
- Automatic feature type detection
- Missing value analysis
- Statistical profiling
- Data quality assessment

### 2. Detection Module (`src/detection/`)

**Purpose:** Automatically detect the ML task type

**Components:**
- `TaskDetector`: Determines if task is classification or regression
- `TaskType`: Enumeration of task types

**Logic:**
- Analyzes target variable characteristics
- Considers number of unique values
- Checks data types
- Provides confidence score and reasoning

### 3. Pipeline Module (`src/pipeline/`)

**Purpose:** Generate and optimize ML pipelines

**Components:**
- `PipelineGenerator`: Creates candidate ML pipelines
- `PipelineOptimizer`: Suggests improvements

**Features:**
- Automatic preprocessing pipeline construction
- Multiple algorithm candidates
- Hyperparameter search spaces
- Issue detection and optimization suggestions

### 4. Model Module (`src/model/`)

**Purpose:** Train and evaluate ML models

**Components:**
- `ModelTrainer`: Handles model training and hyperparameter tuning
- `ModelEvaluator`: Evaluates performance and detects issues

**Features:**
- Cross-validation
- Hyperparameter optimization (GridSearchCV)
- Comprehensive metrics
- Overfitting/underfitting detection
- Performance reporting

### 5. Agent Module (`src/agent/`)

**Purpose:** Orchestrate the entire pipeline

**Components:**
- `PipelineAgent`: Main AI agent that coordinates all steps

**Workflow:**
1. Load data
2. Extract metadata
3. Detect task type
4. Generate candidate pipelines
5. Train models
6. Evaluate performance
7. Detect issues
8. Plan improvements
9. Select best pipeline
10. Generate explanation

### 6. Utils Module (`src/utils/`)

**Purpose:** Shared utilities and configuration

**Components:**
- `Config`: Configuration management
- `Logger`: Logging utilities

## Data Flow

```
Input Dataset
      │
      ▼
┌──────────────┐
│ Data Loader  │ ──> Features (X), Target (y)
└──────────────┘
      │
      ▼
┌──────────────────┐
│ Metadata Extract │ ──> Dataset characteristics
└──────────────────┘
      │
      ▼
┌──────────────┐
│ Task Detect  │ ──> Classification / Regression
└──────────────┘
      │
      ▼
┌──────────────────┐
│ Pipeline Gen     │ ──> 5 candidate pipelines
└──────────────────┘
      │
      ▼
┌──────────────┐
│ Model Train  │ ──> Trained models
└──────────────┘
      │
      ▼
┌──────────────┐
│ Evaluate     │ ──> Metrics, Issues
└──────────────┘
      │
      ▼
┌──────────────┐
│ Optimize     │ ──> Improvement plan
└──────────────┘
      │
      ▼
┌──────────────┐
│ Best Pipeline│ ──> Final model + Explanation
└──────────────┘
```

## Configuration System

Configuration is managed through `config/config.yaml`:

- **Pipeline Settings:** Iterations, stopping criteria
- **Data Processing:** Train/test split, missing value handling
- **Training:** Cross-validation, hyperparameters
- **Evaluation:** Metrics, thresholds
- **Logging:** Verbosity, file output

## Extensibility

### Adding New Models

1. Import model in `src/pipeline/generator.py`
2. Add configuration to `_get_classification_models()` or `_get_regression_models()`
3. Define hyperparameter search space

### Adding New Metrics

1. Update `src/model/evaluator.py`
2. Add metric calculation in `_evaluate_classification()` or `_evaluate_regression()`
3. Update logging and reporting

### Custom Preprocessing

1. Modify `_create_preprocessor()` in `PipelineGenerator`
2. Add custom transformers
3. Update pipeline construction logic

## Error Handling

- Try-catch blocks around model training
- Validation of input data
- Graceful degradation for missing values
- Detailed logging for debugging

## Performance Considerations

- Parallel training with `n_jobs=-1`
- Efficient cross-validation
- Configurable hyperparameter search
- Early stopping for optimization

## Best Practices

1. Always validate input data
2. Use appropriate metrics for task type
3. Monitor for overfitting
4. Log all major operations
5. Provide clear explanations
6. Handle edge cases gracefully

## Future Enhancements

- Neural network support
- Time series detection
- Feature engineering automation
- Model interpretability (SHAP, LIME)
- Multi-objective optimization
- AutoML integration
- Web UI
- API endpoints
