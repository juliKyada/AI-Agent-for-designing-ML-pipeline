"""
MetaFlow Web UI - Streamlit Interface
Upload your dataset and let AI design your ML pipeline!
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import os
import time

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.main import MetaFlowAgent
from src.utils.logger import add_streamlit_sink
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="MetaFlow - AI ML Pipeline Designer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        padding-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'execution_logs' not in st.session_state:
    st.session_state.execution_logs = []
if 'trigger_run' not in st.session_state:
    st.session_state.trigger_run = False
if 'run_params' not in st.session_state:
    st.session_state.run_params = {}

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 MetaFlow</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered ML Pipeline Designer</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset (CSV or Excel)"
        )
        
        if uploaded_file is not None:
            # Load dataset
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.dataset = df
                st.success(f"✅ Loaded: {uploaded_file.name}")
                st.info(f"📊 {len(df)} rows × {len(df.columns)} columns")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return
        
        # Target column selection
        if st.session_state.dataset is not None:
            df = st.session_state.dataset
            
            st.markdown("---")
            target_column = st.selectbox(
                "Select Target Column",
                options=df.columns.tolist(),
                index=len(df.columns) - 1,
                help="The column you want to predict"
            )
            
            # Advanced settings
            st.markdown("---")
            with st.expander("🔧 Advanced Settings"):
                max_iterations = st.slider(
                    "Max Optimization Iterations",
                    min_value=1,
                    max_value=20,
                    value=10,
                    help="Maximum iterations for pipeline improvement"
                )
                
                n_pipelines = st.slider(
                    "Number of Candidate Pipelines",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="More pipelines = better results but slower"
                )
            
            # Run button
            st.markdown("---")
            run_button = st.button("🚀 Run MetaFlow", type="primary", use_container_width=True)
            
            if run_button:
                st.session_state.trigger_run = True
                st.session_state.run_params = {
                    'target_column': target_column,
                    'max_iterations': max_iterations
                }
                st.rerun()
    
    # Main content
    if st.session_state.dataset is None:
        show_landing_page()
    else:
        show_dataset_overview()
        
        if st.session_state.results is not None:
            show_results()

def show_landing_page():
    """Show landing page when no dataset is uploaded"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📤 Upload Dataset")
        st.write("Upload your CSV or Excel file using the sidebar")
    
    with col2:
        st.markdown("### 🎯 Select Target")
        st.write("Choose the column you want to predict")
    
    with col3:
        st.markdown("### 🚀 Get Results")
        st.write("MetaFlow designs the best ML pipeline automatically!")
    
    st.markdown("---")
    
    # Features
    st.markdown("## ✨ What MetaFlow Does")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ✅ **Automatic Task Detection**
        - Identifies Classification or Regression
        - Analyzes target variable characteristics
        
        ✅ **Smart Pipeline Generation**
        - Creates 5+ candidate ML pipelines
        - Tests multiple algorithms
        
        ✅ **Hyperparameter Tuning**
        - Automatic optimization
        - Cross-validation
        """)
    
    with col2:
        st.markdown("""
        ✅ **Performance Evaluation**
        - Comprehensive metrics
        - Overfitting detection
        
        ✅ **Issue Detection**
        - Identifies problems automatically
        - Provides improvement suggestions
        
        ✅ **Best Model Selection**
        - Picks optimal pipeline
        - Ready-to-use model
        """)
    
    # Demo datasets
    st.markdown("---")
    st.markdown("## 📊 Try with Demo Datasets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌸 Iris Dataset (Classification)", use_container_width=True):
            load_demo_dataset('iris')
    
    with col2:
        if st.button("🏥 Diabetes Dataset (Regression)", use_container_width=True):
            load_demo_dataset('diabetes')
    
    with col3:
        if st.button("🎲 Synthetic Dataset", use_container_width=True):
            load_demo_dataset('synthetic')

def load_demo_dataset(dataset_name):
    """Load a demo dataset"""
    from sklearn.datasets import load_iris, load_diabetes, make_classification
    
    if dataset_name == 'iris':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    
    elif dataset_name == 'diabetes':
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    
    elif dataset_name == 'synthetic':
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                   n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
    
    st.session_state.dataset = df
    st.rerun()

def show_dataset_overview():
    """Show dataset overview"""
    df = st.session_state.dataset
    
    col_left, col_right = st.columns([0.7, 0.3])
    
    with col_left:
        st.markdown("## 📊 Dataset Overview")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.metric("Rows", f"{len(df):,}")
        with m_col2:
            st.metric("Columns", len(df.columns))
        with m_col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with m_col4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory", f"{memory_mb:.2f} MB")
        
        # Show data
        with st.expander("👁️ View Data", expanded=False):
            st.dataframe(df.head(100), use_container_width=True)
    
    with col_right:
        if st.session_state.get('trigger_run', False):
            # Process trigger
            params = st.session_state.run_params
            st.session_state.trigger_run = False 
            st.markdown("### 🪵 Activity")
            run_metaflow(df, params['target_column'], params['max_iterations'])
        elif st.session_state.execution_logs:
            st.markdown("### 🪵 Latest Activity")
            latest_log = st.session_state.execution_logs[-5:]
            st.code("\n".join(latest_log), language="text")
        else:
            st.markdown("### 🪵 Activity")
            st.info("Start a run to see logs here.")
    
    # Show statistics
    with st.expander("📈 Statistics", expanded=False):
        st.dataframe(df.describe(), use_container_width=True)

def run_metaflow(df, target_column, max_iterations):
    """Run MetaFlow on the dataset"""
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Clear previous results but KEEP logs (until new ones start)
    st.session_state.results = None
    st.session_state.execution_logs = []
    
    # Progress and Logs
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    
    def streamlit_log_callback(message):
        # Extract the plain message
        msg = str(message).strip()
        st.session_state.execution_logs.append(msg)
        # Live update during execution - show last 10 lines in right column
        log_placeholder.code("\n".join(st.session_state.execution_logs[-10:]))

    # Add the sink
    sink_id = add_streamlit_sink(streamlit_log_callback)
    
    with st.spinner("🤖 MetaFlow is working..."):
        try:
            # Initialize agent
            status_placeholder.info("🔄 Initializing MetaFlow Agent...")
            agent = MetaFlowAgent()
            
            # Run pipeline
            status_placeholder.info("🔄 Analyzing dataset and designing pipelines...")
            
            results = agent.run(
                dataframe=df,
                target_column=target_column,
                max_iterations=max_iterations
            )
            
            # Store results
            st.session_state.results = results
            
            status_placeholder.success("✅ MetaFlow completed successfully!")
            time.sleep(1)
            
            # Cleanup sink before rerun
            try:
                from loguru import logger
                logger.remove(sink_id)
            except ValueError:
                pass
            
            # Rerun to show results
            st.rerun()
            
        except Exception as e:
            try:
                from loguru import logger
                logger.remove(sink_id)
            except ValueError:
                pass
            status_placeholder.error(f"❌ Error: {str(e)}")
            st.error(f"Error details: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def show_results():
    """Show MetaFlow results"""
    results = st.session_state.results
    
    st.markdown("---")
    st.markdown("## 🎯 Results")
    
    # Task type
    col1, col2, col3 = st.columns(3)
    
    with col1:
        task_type = results['task_type'].upper()
        st.markdown(f"### 📋 Task Type")
        st.markdown(f"**{task_type}**")
    
    with col2:
        best_pipeline = results['best_pipeline']
        st.markdown(f"### 🏆 Best Model")
        st.markdown(f"**{best_pipeline['name']}**")
    
    with col3:
        metrics = best_pipeline['metrics']
        if results['task_type'] == 'classification':
            score = metrics['test_accuracy']
            metric_name = "Accuracy"
        else:
            score = metrics['test_r2']
            metric_name = "R² Score"
        
        st.markdown(f"### 📊 {metric_name}")
        st.markdown(f"**{score:.4f}**")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance", 
        "📈 All Pipelines", 
        "⚠️ Issues & Recommendations", 
        "📄 Full Report",
        "🪵 Execution Logs"
    ])
    
    with tab1:
        show_performance_tab(results)
    
    with tab2:
        show_all_pipelines_tab(results)
    
    with tab3:
        show_issues_tab(results)
    
    with tab4:
        show_full_report_tab(results)
        
    with tab5:
        st.markdown("### 🪵 System Execution Logs")
        if st.session_state.execution_logs:
            st.code("\n".join(st.session_state.execution_logs), language="text")
        else:
            st.info("No logs available for this session.")
    
    # Download section
    st.markdown("---")
    st.markdown("## 💾 Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Save model
        model_path = 'models/best_model.pkl'
        if st.button("💾 Save Best Model", use_container_width=True):
            from src.main import MetaFlowAgent
            agent = MetaFlowAgent()
            agent.agent.results = results
            agent.save_best_model(model_path)
            st.success(f"✅ Model saved to: {model_path}")
    
    with col2:
        # Download report
        report = results['evaluation_report']
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="metaflow_report.txt",
            mime="text/plain",
            use_container_width=True
        )

def show_performance_tab(results):
    """Show performance metrics"""
    best_pipeline = results['best_pipeline']
    metrics = best_pipeline['metrics']
    
    if results['task_type'] == 'classification':
        # Classification metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Train Metrics")
            st.metric("Accuracy", f"{metrics['train_accuracy']:.4f}")
            st.metric("Precision", f"{metrics['train_precision']:.4f}")
            st.metric("Recall", f"{metrics['train_recall']:.4f}")
            st.metric("F1 Score", f"{metrics['train_f1']:.4f}")
        
        with col2:
            st.markdown("#### Test Metrics")
            st.metric("Accuracy", f"{metrics['test_accuracy']:.4f}")
            st.metric("Precision", f"{metrics['test_precision']:.4f}")
            st.metric("Recall", f"{metrics['test_recall']:.4f}")
            st.metric("F1 Score", f"{metrics['test_f1']:.4f}")
        
        # Visualization
        fig = go.Figure()
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1']
        train_values = [metrics['train_accuracy'], metrics['train_precision'], 
                       metrics['train_recall'], metrics['train_f1']]
        test_values = [metrics['test_accuracy'], metrics['test_precision'],
                      metrics['test_recall'], metrics['test_f1']]
        
        fig.add_trace(go.Scatterpolar(
            r=train_values,
            theta=categories,
            fill='toself',
            name='Train'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=test_values,
            theta=categories,
            fill='toself',
            name='Test'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Performance Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Regression metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Train Metrics")
            st.metric("R² Score", f"{metrics['train_r2']:.4f}")
            st.metric("RMSE", f"{metrics['train_rmse']:.4f}")
            st.metric("MAE", f"{metrics['train_mae']:.4f}")
        
        with col2:
            st.markdown("#### Test Metrics")
            st.metric("R² Score", f"{metrics['test_r2']:.4f}")
            st.metric("RMSE", f"{metrics['test_rmse']:.4f}")
            st.metric("MAE", f"{metrics['test_mae']:.4f}")

def show_all_pipelines_tab(results):
    """Show all evaluated pipelines"""
    all_pipelines = results['all_pipelines']
    
    # Create comparison dataframe
    comparison_data = []
    for pipeline in all_pipelines:
        metrics = pipeline['metrics']
        
        if results['task_type'] == 'classification':
            comparison_data.append({
                'Pipeline': pipeline['pipeline_name'],
                'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
                'Test F1': f"{metrics['test_f1']:.4f}",
                'CV Score': f"{metrics['cv_mean']:.4f}",
                'Issues': len(pipeline['issues'])
            })
        else:
            comparison_data.append({
                'Pipeline': pipeline['pipeline_name'],
                'Test R²': f"{metrics['test_r2']:.4f}",
                'Test RMSE': f"{metrics['test_rmse']:.4f}",
                'CV Score': f"{metrics['cv_mean']:.4f}",
                'Issues': len(pipeline['issues'])
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Visualization
    if results['task_type'] == 'classification':
        scores = [p['metrics']['test_accuracy'] for p in all_pipelines]
        metric_name = 'Test Accuracy'
    else:
        scores = [p['metrics']['test_r2'] for p in all_pipelines]
        metric_name = 'Test R²'
    
    names = [p['pipeline_name'] for p in all_pipelines]
    
    fig = px.bar(
        x=names,
        y=scores,
        labels={'x': 'Pipeline', 'y': metric_name},
        title=f'Pipeline Comparison - {metric_name}'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_issues_tab(results):
    """Show issues and recommendations"""
    best_pipeline = results['best_pipeline']
    improvement_plan = results['improvement_plan']
    
    if best_pipeline['issues']:
        st.markdown("### ⚠️ Detected Issues")
        for issue in best_pipeline['issues']:
            st.warning(f"• {issue}")
    else:
        st.success("✅ No issues detected! The model is performing well.")
    
    st.markdown("---")
    
    if improvement_plan['needs_improvement']:
        st.markdown("### 💡 Recommendations")
        
        if 'overall_recommendations' in improvement_plan and improvement_plan['overall_recommendations']:
            for rec in improvement_plan['overall_recommendations']:
                st.info(f"• {rec}")
        else:
            st.info("• Consider hyperparameter tuning for further optimization")
            st.info("• Review feature engineering opportunities")
            st.info("• Collect more training data if possible")
    else:
        st.success("🎉 All pipelines are performing well! No immediate improvements needed.")

def show_full_report_tab(results):
    """Show full report"""
    st.markdown("### 📄 Complete Evaluation Report")
    
    # Show explanation
    st.markdown(results['explanation'])
    
    # Show full report in expandable section
    with st.expander("📊 Detailed Technical Report", expanded=False):
        st.code(results['evaluation_report'])

if __name__ == '__main__':
    main()
