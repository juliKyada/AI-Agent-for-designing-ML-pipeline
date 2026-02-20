# MetaFlow - AI-Powered ML Pipeline Automation

MetaFlow is an intelligent AI agent that automatically designs, optimizes, and evaluates machine learning pipelines for your datasets.

## 🚀 Features

- **Automatic Task Detection**: Identifies whether your problem is classification or regression
- **Metadata Extraction**: Analyzes dataset characteristics automatically
- **Intelligent Data Preprocessing**: Automatically handles missing values with smart imputation
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

## 🚀 Deploy on Hugging Face Spaces

You can run MetaFlow in the cloud for free using [Hugging Face Spaces](https://huggingface.co/spaces) (Streamlit).

### Option A: Deploy via the website

1. **Create a Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
   - Choose **Streamlit** as the SDK.
   - Pick a name (e.g. `metaflow-ml-pipeline`) and create the Space.

2. **Upload your project**
   - Clone the Space repo (e.g. `git clone https://huggingface.co/spaces/YOUR_USERNAME/metaflow-ml-pipeline`).
   - Copy into the repo:
     - `app.py` (at the root)
     - `src/` (entire folder)
     - `config/` (entire folder)
     - `requirements-huggingface.txt` → rename or copy as **`requirements.txt`** in the Space root.
   - Commit and push:
     ```bash
     cd metaflow-ml-pipeline
     git add app.py src config requirements.txt
     git commit -m "Add MetaFlow app"
     git push
     ```
   - The Space will build and run your app.

3. **Or use “Upload files”**
   - In the Space page, use **Files → Upload files** and add `app.py`, then upload the `src` and `config` folders and a `requirements.txt` (from `requirements-huggingface.txt`).

### Option B: Deploy from your repo (Git-based)

1. Create a new Space and choose **Streamlit**.
2. In Space **Settings → Repository**, you can connect a GitHub repo or copy files from this repo so that the Space root contains:
   - `app.py`
   - `src/`
   - `config/`
   - `requirements.txt` (use `requirements-huggingface.txt` as contents).

### Can the free CPU handle model training?

**Short answer: only for light workloads.** Free Spaces have limited resources:

| Free tier (CPU Basic) | Limit |
|----------------------|--------|
| CPU                  | 2 vCPU cores |
| RAM                  | 16 GB        |
| Disk                 | 50 GB (non-persistent) |

- **Small datasets (e.g. &lt; 10k rows, &lt; 50 features):** Usually fine. Training 3–5 pipelines with 3–5 fold CV can complete in a few minutes.
- **Medium/large datasets or many pipelines:** Can be slow, hit RAM limits, or time out. The Space may become unresponsive or restart.

**Recommendations for free-tier deployment:**

1. **Use lighter settings** in the Streamlit UI (or in `config/config.yaml` before deploying):
   - **Number of candidate pipelines:** 3 (not 5–10).
   - **Max optimization iterations:** 3–5 (not 10).
   - **CV folds:** 3 (in `config/config.yaml`: `training.cv_folds: 3`).
2. **Ask users to upload small/medium datasets** (e.g. &lt; 5k rows for a smooth experience).
3. Free Spaces run on CPU; all models (XGBoost, LightGBM, etc.) use CPU.
4. For **heavy training or big data**, use a **paid CPU Upgrade** (e.g. 8 vCPU, 32 GB RAM) or run the app on your own server/Colab.

### Other notes for Hugging Face

- **Faster install:** Use `requirements-huggingface.txt` as your Space `requirements.txt` to avoid installing dev/test packages.
- **File size:** Users upload their own datasets in the UI; you don’t need to bundle data.
- **Sleep:** Inactive free Spaces sleep after a while; the first load after sleep can be slow.

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
