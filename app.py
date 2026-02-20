"""
MetaFlow Web UI - Streamlit Interface
Upload your dataset and let AI design your ML pipeline!
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sys
from pathlib import Path
import os
import time

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.main import MetaFlowAgent
from src.utils import get_logger
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
    /* Keep main content (right side) at full opacity during pipeline execution */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] section.main,
    [data-testid="stAppViewContainer"] .block-container,
    [data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"] {
        opacity: 1 !important;
    }
    /* Prevent Streamlit's script-running overlay from dimming the main area */
    [data-testid="stAppViewContainer"] > div {
        opacity: 1 !important;
    }
    /* Smaller, compact text for the evaluation report */
    .metaflow-report, .metaflow-report p, .metaflow-report li, .metaflow-report strong {
        font-size: 1.3rem !important;
        line-height: 1.4 !important;
    }
    .metaflow-report hr {
        margin: 0.6rem 0 !important;
        border: none;
        border-top: 1px solid rgba(128, 128, 128, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'pipeline_logs' not in st.session_state:
    st.session_state.pipeline_logs = []
if 'run_pipeline' not in st.session_state:
    st.session_state.run_pipeline = False

def _render_logs_scrollable(log_lines, max_height_px=360):
    """Render log lines in a fixed-height scrollable box with smart auto-scroll (stay at bottom unless user scrolls up)."""
    import html
    if not log_lines:
        return ""
    lines_html = "<br>".join(html.escape(line) for line in log_lines)
    scroll_script = """
    <script>
    (function() {
        var el = document.getElementById('metaflow-log-scroll');
        if (!el) return;
        var keyAuto = 'metaflow_log_autoscroll';
        var keyTop = 'metaflow_log_scrollTop';
        var keyHeight = 'metaflow_log_scrollHeight';
        function getStorage() {
            try {
                if (window.parent && window.parent !== window && window.parent.localStorage)
                    return window.parent.localStorage;
            } catch (e) {}
            try { return localStorage; } catch (e) {}
            return sessionStorage;
        }
        var storage = getStorage();
        el.addEventListener('scroll', function() {
            var atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 25;
            try {
                storage.setItem(keyAuto, atBottom ? '1' : '0');
                storage.setItem(keyTop, String(el.scrollTop));
                storage.setItem(keyHeight, String(el.scrollHeight));
            } catch (e) {}
        });
        var atBottom = storage.getItem(keyAuto) !== '0';
        if (atBottom) {
            el.scrollTop = el.scrollHeight;
            try { storage.setItem(keyAuto, '1'); storage.setItem(keyTop, String(el.scrollTop)); storage.setItem(keyHeight, String(el.scrollHeight)); } catch(e) {}
        } else {
            var savedTop = parseFloat(storage.getItem(keyTop)) || 0;
            var savedHeight = parseFloat(storage.getItem(keyHeight)) || 1;
            if (savedHeight > 0 && el.scrollHeight > 0) {
                var ratio = Math.min(1, savedTop / savedHeight);
                el.scrollTop = ratio * el.scrollHeight;
            }
        }
    })();
    </script>
    """
    # Light text so logs are visible on dark theme; single scrollbar only (no iframe scrollbar)
    return f"""
    <div style="margin:0; padding:0; overflow:hidden;">
    <div id="metaflow-log-scroll" style="max-height: {max_height_px}px; overflow-y: auto; overflow-x: auto;
                border: 1px solid rgba(49, 51, 63, 0.6); border-radius: 6px;
                padding: 12px; background: rgba(30,30,30,0.95); margin-top: 8px;
                color: #e8e8e8;">
        <div style="font-family: monospace; font-size: 13px; white-space: pre; line-height: 1.5; color: #e8e8e8;">{lines_html}</div>
    </div>
    </div>
    """ + scroll_script

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
                    help="Reserved for future multi-round optimization (currently not used)."
                )
                
                n_pipelines = st.slider(
                    "Number of Candidate Pipelines",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="More pipelines = better results but slower"
                )
            
            # Run button (only set flag; actual run happens in main area so logs appear on the right)
            st.markdown("---")
            run_button = st.button("🚀 Run MetaFlow", type="primary", width="stretch")
            
            if run_button:
                st.session_state.run_pipeline = True
                st.session_state.run_target_column = target_column
                st.session_state.run_max_iterations = max_iterations
                st.session_state.run_n_pipelines = n_pipelines
    
    # Main content
    if st.session_state.dataset is None:
        show_landing_page()
    else:
        show_dataset_overview()

        # Run pipeline from main area so status and logs render on the right, not in sidebar
        if st.session_state.get("run_pipeline", False):
            st.session_state.run_pipeline = False
            run_metaflow(
                st.session_state.dataset,
                st.session_state.run_target_column,
                st.session_state.run_max_iterations,
                st.session_state.run_n_pipelines,
            )
            return

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
        if st.button("🌸 Iris Dataset (Classification)", width="stretch"):
            load_demo_dataset('iris')
    
    with col2:
        if st.button("🏥 Diabetes Dataset (Regression)", width="stretch"):
            load_demo_dataset('diabetes')
    
    with col3:
        if st.button("🎲 Synthetic Dataset", width="stretch"):
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
    
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.2f} MB")
    
    # Show data
    with st.expander("👁️ View Data", expanded=False):
        st.dataframe(df.head(100), width="stretch")
    
    # Show statistics
    with st.expander("📈 Statistics", expanded=False):
        st.dataframe(df.describe(), width="stretch")

    # Execution logs — auto-expand when Run is clicked so user can see; scrollable box with smart auto-scroll
    with st.expander(
        "📋 Execution Logs",
        expanded=st.session_state.get("run_pipeline", False) or bool(st.session_state.pipeline_logs),
    ):
        st.session_state.execution_log_status_ph = st.empty()
        st.session_state.execution_log_content_ph = st.empty()
        if st.session_state.pipeline_logs:
            with st.session_state.execution_log_content_ph.container():
                components.html(
                    _render_logs_scrollable(st.session_state.pipeline_logs),
                    height=400,
                    scrolling=False,
                )
        else:
            st.session_state.execution_log_content_ph.info(
                "No execution logs yet. Run MetaFlow to see logs here."
            )

def run_metaflow(df, target_column, max_iterations, n_pipelines):
    """Run MetaFlow on the dataset"""
    import threading

    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Clear and prepare logs for this run
    st.session_state.pipeline_logs = []
    log_list = st.session_state.pipeline_logs
    _logger = get_logger()

    # Use placeholders inside Execution Logs expander (set by show_dataset_overview)
    status_ph = st.session_state.get("execution_log_status_ph")
    log_ph = st.session_state.get("execution_log_content_ph")

    result_holder = [None]
    exception_holder = [None]

    def run_agent():
        sink_id = None
        try:
            agent = MetaFlowAgent()  # setup_logger() runs here and removes all handlers
            # Add our sink after agent init so it is not removed by setup_logger
            sink_id = _logger.add(
                lambda msg: log_list.append(msg.strip()),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                level="DEBUG",
            )
            result_holder[0] = agent.run(
                dataframe=df,
                target_column=target_column,
                max_iterations=max_iterations,
                n_pipelines=n_pipelines,
            )
        except Exception as e:
            exception_holder[0] = e
        finally:
            if sink_id is not None:
                try:
                    _logger.remove(sink_id)
                except ValueError:
                    pass

    thread = threading.Thread(target=run_agent)
    thread.start()

    try:
        if status_ph:
            status_ph.info("🔄 Running pipeline... Live logs below.")
        while thread.is_alive():
            if log_list and log_ph:
                with log_ph.container():
                    components.html(
                        _render_logs_scrollable(log_list),
                        height=400,
                        scrolling=False,
                    )
            time.sleep(0.5)
        thread.join()

        if exception_holder[0] is not None:
            raise exception_holder[0]

        results = result_holder[0]
        st.session_state.results = results

        if status_ph:
            status_ph.success("✅ MetaFlow completed successfully!")
        time.sleep(1)
        if status_ph:
            status_ph.empty()
        if log_ph and log_list:
            with log_ph.container():
                components.html(
                    _render_logs_scrollable(log_list),
                    height=400,
                    scrolling=False,
                )

        st.rerun()

    except Exception as e:
        if status_ph:
            status_ph.error(f"❌ Error: {str(e)}")
        st.error(f"Error details: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        if log_ph and log_list:
            with log_ph.container():
                components.html(
                    _render_logs_scrollable(log_list),
                    height=400,
                    scrolling=False,
                )

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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 All Pipelines", "⚠️ Issues & Recommendations", "📄 Full Report"])
    
    with tab1:
        show_performance_tab(results)
    
    with tab2:
        show_all_pipelines_tab(results)
    
    with tab3:
        show_issues_tab(results)
    
    with tab4:
        show_full_report_tab(results)
    
    # Download section
    st.markdown("---")
    st.markdown("## 💾 Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Save model
        model_path = 'models/best_model.pkl'
        if st.button("💾 Save Best Model", width="stretch"):
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
            width="stretch"
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
        
        st.plotly_chart(fig, width="stretch")
    
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
    st.dataframe(df_comparison, width="stretch")
    
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
    
    st.plotly_chart(fig, width="stretch")

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
    
    # Show explanation in a smaller font container
    report_html = f'<div class="metaflow-report">\n\n{results["explanation"]}\n\n</div>'
    st.markdown(report_html, unsafe_allow_html=True)
    
    # Show preprocessing information
    if 'preprocessing' in results and results['preprocessing']:
        st.markdown("---")
        st.markdown("### 🔧 Data Preprocessing Report")
        
        preprocessing = results['preprocessing']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Imputation Strategy", preprocessing.get('imputation_strategy', 'N/A'))
        
        with col2:
            removed_features = len(preprocessing.get('removed_features', []))
            st.metric("Features Removed", removed_features)
        
        with col3:
            imputed_features = len(preprocessing.get('imputation_values', {}))
            st.metric("Features Imputed", imputed_features)
        
        with col4:
            rows_removed = preprocessing.get('rows_removed_by_target_na', 0)
            st.metric("Rows Removed (NaN Target)", rows_removed)
        
        # Show removed features
        if preprocessing.get('removed_features'):
            st.markdown("#### Removed Features (High Missing Ratio)")
            st.write(", ".join(preprocessing['removed_features']))
        
        # Show imputation values
        if preprocessing.get('imputation_values'):
            st.markdown("#### Imputation Values Used")
            imputation_df = pd.DataFrame({
                'Feature': list(preprocessing['imputation_values'].keys()),
                'Value': list(preprocessing['imputation_values'].values())
            })
            st.dataframe(imputation_df, width="stretch")
    
    # Show full report in expandable section
    with st.expander("📊 Detailed Technical Report", expanded=False):
        st.code(results['evaluation_report'])

if __name__ == '__main__':
    main()
