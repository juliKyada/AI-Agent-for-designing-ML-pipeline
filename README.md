# MetaFlow - AI-Powered ML Pipeline Automation

MetaFlow is an intelligent AI agent that automatically designs, optimizes, and evaluates machine learning pipelines for your datasets.

## 🚀 Features

- **Automatic Task Detection**: Identifies whether your problem is classification or regression
- **Metadata Extraction**: Analyzes dataset characteristics automatically
- **Pipeline Generation**: Creates multiple candidate ML pipelines
- **Smart Training**: Trains models with optimal hyperparameters
- **Performance Evaluation**: Comprehensive metrics and validation
- **Auto-Optimization**: Detects overfitting and iteratively improves pipelines
- **Explainable Results**: Provides clear explanations of the final pipeline
- **🎨 Web UI**: Beautiful Streamlit interface for easy dataset upload and visualization

## 🌐 Web Interface (NEW!)

MetaFlow now includes a beautiful web interface! No coding required.

### Quick Start:
```bash
# Install Streamlit (if not already installed)
pip install streamlit plotly

# Launch the web UI
streamlit run app.py
```

Or simply **double-click `start_ui.bat`** (Windows) and your browser will open automatically!

**Features:**
- 📤 Drag-and-drop dataset upload
- 🎯 Interactive target column selection
- 📊 Real-time performance visualizations
- 📈 Pipeline comparison charts
- 💾 One-click model download
- 🎨 Professional, responsive design

See [UI_GUIDE.md](UI_GUIDE.md) for detailed instructions.

## 📋 Pipeline Flow

```
User uploads dataset
        ↓
Extract metadata (features, target, data types, missing values)
        ↓
Detect task (classification / regression)
        ↓
Generate candidate pipelines
        ↓
Train models
        ↓
Evaluate performance
        ↓
Check for overfitting / low score
        ↓
Improve pipeline
        ↓
Repeat until good
        ↓
Display final pipeline + explanation
```

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📖 Quick Start

```python
from src.main import MetaFlowAgent

# Initialize the agent
agent = MetaFlowAgent()

# Run automated pipeline design
results = agent.run(dataset_path="data/your_dataset.csv")

# Get the best pipeline
best_pipeline = results['best_pipeline']
explanation = results['explanation']
metrics = results['metrics']
```

## 📁 Project Structure

```
MetaFlow/
├── src/                    # Source code
│   ├── data/              # Data loading and metadata extraction
│   ├── detection/         # Task type detection
│   ├── pipeline/          # Pipeline generation and optimization
│   ├── model/             # Model training and evaluation
│   ├── agent/             # AI agent orchestration
│   └── utils/             # Utilities and configuration
├── config/                # Configuration files
├── tests/                 # Unit tests
├── examples/              # Example scripts
└── requirements.txt       # Dependencies
```

## 🎯 Usage Examples

See `examples/sample_usage.py` for detailed examples.

## 📊 Supported Algorithms

- Linear Models (Linear/Logistic Regression, Ridge, Lasso)
- Tree-based Models (Decision Trees, Random Forest, XGBoost, LightGBM)
- Support Vector Machines
- Neural Networks
- Ensemble Methods

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Max iterations for optimization
- Evaluation metrics
- Cross-validation settings
- Model search space
- Performance thresholds

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
