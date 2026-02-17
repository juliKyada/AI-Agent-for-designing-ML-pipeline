# MetaFlow Web UI - Quick Start Guide

## 🚀 Launch the Web Interface

### Method 1: Double-click (Windows)
```
Double-click: start_ui.bat
```

### Method 2: Command Line
```bash
# Install Streamlit first (if not installed)
pip install streamlit plotly

# Run the web UI
streamlit run app.py
```

### Method 3: Python command
```bash
python -m streamlit run app.py
```

## 📱 Access the Interface

Once started, open your browser and go to:
```
http://localhost:8501
```

The interface will automatically open in your default browser!

## 🎯 How to Use

### Step 1: Upload Dataset
- Click "Browse files" in the sidebar
- Select your CSV or Excel file
- Or try one of the demo datasets

### Step 2: Select Target Column
- Choose the column you want to predict from the dropdown

### Step 3: Configure (Optional)
- Expand "Advanced Settings" to adjust:
  - Max optimization iterations
  - Number of candidate pipelines

### Step 4: Run MetaFlow
- Click the "🚀 Run MetaFlow" button
- Wait for the analysis to complete (30s - 2min)

### Step 5: View Results
- See performance metrics
- Compare all pipelines
- Check issues and recommendations
- Download the best model

## 📊 Features

### Upload Options
✅ CSV files (.csv)
✅ Excel files (.xlsx, .xls)
✅ Demo datasets (Iris, Diabetes, Synthetic)

### What You Get
✅ Automatic task detection (Classification/Regression)
✅ Multiple ML pipelines tested
✅ Performance metrics and visualizations
✅ Issue detection and recommendations
✅ Downloadable trained model
✅ Complete evaluation report

### Interactive Features
- 📊 Performance radar charts
- 📈 Pipeline comparison bar charts
- 👁️ Data preview
- 📈 Statistical summaries
- ⚠️ Issue warnings
- 💡 Smart recommendations

## 🔧 Troubleshooting

### Port already in use?
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Browser doesn't open?
Manually navigate to: `http://localhost:8501`

### Installation issues?
```bash
# Install all dependencies
pip install -r requirements.txt
```

## 💡 Tips

1. **Start with demo datasets** to see how it works
2. **Check data preview** before running
3. **Use advanced settings** for larger datasets
4. **Download the report** for detailed analysis
5. **Save the model** for future use

## 🎨 UI Features

- **Responsive Design** - Works on desktop and tablets
- **Real-time Updates** - See progress as it runs
- **Interactive Charts** - Hover for details
- **Clean Interface** - Easy to navigate
- **Professional Look** - Production-ready

## 📧 Support

Check the main README.md for more information or examples/sample_usage.py for code examples.

Enjoy using MetaFlow! 🚀
