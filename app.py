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
from src.report import GroqReportGenerator
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    :root{
        --bg:#0f1724;
        --card:#0b1220;
        --muted:#9aa4b2;
        --accent:#4f46e5;
        --accent-2:#06b6d4;
        --glass: rgba(255,255,255,0.03);
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #071029 0%, #0f1724 100%);
        color: #e6eef8;
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: left;
        color: white;
        margin: 0;
        letter-spacing: -0.6px;
    }
    .sub-header {
        font-size: 1.05rem;
        text-align: left;
        color: var(--muted);
        margin-top: 6px;
    }

    /* Hero */
    .hero {
        padding: 28px 28px 18px 28px;
        background: linear-gradient(90deg, rgba(79,70,229,0.06), rgba(6,182,212,0.03));
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.04);
        box-shadow: 0 6px 30px rgba(2,6,23,0.6);
    }
    .cta-btn {
        display:inline-block;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white !important;
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 700;
        border: none;
        box-shadow: 0 8px 20px rgba(79,70,229,0.18);
    }
    .feature-card{
        background: var(--card);
        border-radius: 10px;
        padding: 14px;
        border: 1px solid rgba(255,255,255,0.03);
        min-height: 120px;
    }
    .feature-title{font-weight:700; color:#fff; margin-bottom:6px}
    .feature-desc{color:var(--muted); font-size:0.95rem}

    /* Sidebar tweaks */
    [data-testid="stSidebar"] .css-1d391kg { padding: 18px 16px; }
    [data-testid="stSidebar"] h2 { color: #fff; }
    .upload-box{ background: var(--glass); padding: 10px; border-radius:8px; border:1px solid rgba(255,255,255,0.02); }

    /* Logs monospace */
    .log-mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, 'Roboto Mono', monospace; font-size:13px }

    /* Smaller, compact text for the evaluation report */
    .metaflow-report, .metaflow-report p, .metaflow-report li, .metaflow-report strong {
        font-size: 1rem !important;
        line-height: 1.4 !important;
        color: #dfe7f7;
    }
    .metaflow-report hr { margin: 0.6rem 0 !important; border: none; border-top: 1px solid rgba(255,255,255,0.06); }

    /* Responsive tweaks */
    @media (max-width: 900px){ .main-header{ font-size:1.8rem } .hero{ padding:18px } }
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
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'cancel_training' not in st.session_state:
    st.session_state.cancel_training = False
if 'cancel_event' not in st.session_state:
    st.session_state.cancel_event = None
if 'ai_report' not in st.session_state:
    st.session_state.ai_report = None
if 'ai_report_generating' not in st.session_state:
    st.session_state.ai_report_generating = False

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
    
    # (Header is included in the landing hero to keep layout consistent)
    
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
                st.session_state.dataset_filename = uploaded_file.name
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
                    max_value=5,
                    value=3,
                    help="Maximum rounds of optimization to improve pipeline quality."
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
            run_button = st.button("🚀 Run MetaFlow", type="primary", use_container_width=True)
            
            if run_button:
                # Start a new run and reset cancellation state so that the
                # Stop button becomes available immediately in the UI.
                st.session_state.cancel_training = False
                st.session_state.pipeline_running = True
                st.session_state.run_pipeline = True
                st.session_state.run_target_column = target_column
                st.session_state.run_max_iterations = max_iterations
                st.session_state.run_n_pipelines = n_pipelines

            # When a run is active, expose a Stop button directly in the sidebar
            if st.session_state.get("pipeline_running", False):
                if st.button("⏹ Stop Training", key="stop_training_sidebar", type="secondary"):
                    st.session_state.cancel_training = True
                    # Notify the background training thread via shared Event
                    cancel_event = st.session_state.get("cancel_event", None)
                    if cancel_event is not None:
                        cancel_event.set()
    
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
    # Hero + features layout
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div style="padding:6px 6px 2px 6px">', unsafe_allow_html=True)
        st.markdown('<div class="main-header">🤖 MetaFlow</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-powered ML pipeline designer — from data to production-ready model faster.</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--muted); font-size:1rem; max-width:720px">MetaFlow analyzes your dataset, detects the task, builds multiple candidate pipelines, tunes hyperparameters, evaluates performance and selects the best ready-to-deploy model — all with minimal effort.</p>', unsafe_allow_html=True)
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # small summary cards
        st.markdown('<div style="display:flex;flex-direction:column;gap:10px">', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><div class="feature-title">Fast Results</div><div class="feature-desc">Generate candidate pipelines and get recommendations in minutes.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><div class="feature-title">Robust Evaluation</div><div class="feature-desc">Cross-validation, overfitting checks and clear metrics for model choice.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><div class="feature-title">Ready To Deploy</div><div class="feature-desc">Export the selected model and evaluation report for production use.</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    # Detailed features section
    st.markdown('## ✨ Key Capabilities')
    f1, f2, f3 = st.columns([1,1,1])
    with f1:
        st.markdown('<div class="feature-card"><div class="feature-title">Automatic Task Detection</div><div class="feature-desc">Detects whether your problem is classification or regression and adapts the pipeline accordingly.</div></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="feature-card"><div class="feature-title">Smart Pipeline Generation</div><div class="feature-desc">Builds multiple candidate pipelines combining feature processing and model choices.</div></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="feature-card"><div class="feature-title">Hyperparameter Tuning</div><div class="feature-desc">Automatic optimization with cross-validation to improve generalization.</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)



    st.markdown(' ') 

    # Demo datasets (prominent)
    st.markdown('## 📊 Try Demo Datasets')
    d1, d2, d3 = st.columns(3)
    with d1:
        if st.button('🌸 Iris — Classification', key='demo_iris'):
            load_demo_dataset('iris')
    with d2:
        if st.button('🏥 Diabetes — Regression', key='demo_diabetes'):
            load_demo_dataset('diabetes')
    with d3:
        if st.button('🎲 Synthetic — Binary', key='demo_synth'):
            load_demo_dataset('synthetic')

    st.markdown('---')

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
        st.dataframe(df.head(100), use_container_width=True)
    
    # Show statistics
    with st.expander("📈 Statistics", expanded=False):
        st.dataframe(df.describe(), use_container_width=True)

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

        # Allow user to request stopping training while it is running
        if st.session_state.get("pipeline_running", False):
            if st.button("⏹ Stop Training", key="stop_training_button"):
                st.session_state.cancel_training = True
                cancel_event = st.session_state.get("cancel_event", None)
                if cancel_event is not None:
                    cancel_event.set()

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

    # Reset cancellation state and mark pipeline as running
    st.session_state.cancel_training = False
    st.session_state.pipeline_running = True

    # Create a shared Event so the UI thread can signal cancellation
    cancel_event = threading.Event()
    st.session_state.cancel_event = cancel_event

    def stop_requested() -> bool:
        """Check whether the user has requested training cancellation.

        This consults the shared threading.Event, which is set by the
        Stop buttons in the UI. Using an Event avoids relying on
        Streamlit's session_state across threads.
        """
        return cancel_event.is_set()

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
                stop_callback=stop_requested,
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
            # Update status if user has requested cancellation
            if st.session_state.get("cancel_training", False) and status_ph:
                status_ph.warning("⏹ Stop requested. Finishing current training step before stopping...")
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
        error_msg = str(e)
        
        # Check for data quality issues and show friendly message
        if "No models trained successfully" in error_msg or "No evaluations available" in error_msg:
            st.error("### ⚠️ Data Quality Issue")
            st.markdown(f"""
**Your dataset has insufficient samples for reliable model training.**

This usually happens when:
- **Classes are too small**: Some classes have fewer than 2 samples per class
- **Extreme imbalance**: One class dominates significantly  
- **Dataset too small**: Very few total samples overall

**Recommended Solutions:**
1. **Collect more data** - Gather additional samples, especially for minority classes
2. **Combine similar classes** - Merge small classes into broader categories
3. **Use simpler models** - Some models are more tolerant of small sample sizes
4. **Rebalance the target** - Work towards more balanced class distribution

**Your Current Data:**
- Total samples: {len(df)}
- Target variable: {target_column}
- Class distribution: {df[target_column].value_counts().to_dict() if target_column in df.columns else 'N/A'}
            """)
        else:
            st.error(f"❌ Error: {error_msg}")
        
        if status_ph:
            status_ph.empty()
        if log_ph and log_list:
            with log_ph.container():
                components.html(
                    _render_logs_scrollable(log_list),
                    height=400,
                    scrolling=False,
                )
    finally:
        # Ensure flags are reset when run finishes (successfully or not)
        st.session_state.pipeline_running = False

def show_results():
    """Show MetaFlow results"""
    results = st.session_state.results

    st.markdown("---")
    st.markdown("## 🎯 Results")
    
    # Display any data quality warnings
    if 'training_warnings' in results and results['training_warnings']:
        with st.container():
            st.info("### ⚠️ Data Quality Notes")
            for warning in results['training_warnings']:
                st.write(warning)
    
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Performance", "📈 All Pipelines", "⚠️ Issues & Recommendations", "📄 Full Report", "🤖 AI Report"])
    
    with tab1:
        show_performance_tab(results)
    
    with tab2:
        show_all_pipelines_tab(results)
    
    with tab3:
        show_issues_tab(results)
    
    with tab4:
        show_full_report_tab(results)
    
    with tab5:
        show_ai_report_tab(results)
    
    # Download section
    st.markdown("---")
    st.markdown("## 💾 Download")
    
    col1, col2, col3 = st.columns(3)
    
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
        # Download technical report
        report = results['evaluation_report']
        st.download_button(
            label="📄 Download Technical Report",
            data=report,
            file_name="metaflow_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col3:
        # Download AI report (if generated)
        ai_report = st.session_state.get("ai_report")
        if ai_report:
            st.download_button(
                label="🤖 Download AI Report (.md)",
                data=ai_report,
                file_name="metaflow_ai_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.button("🤖 Download AI Report (.md)", disabled=True, use_container_width=True,
                      help="Generate the AI Report first (🤖 AI Report tab)")

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
            st.dataframe(imputation_df, use_container_width=True)
    
    # Show full report in expandable section
    with st.expander("📊 Detailed Technical Report", expanded=False):
        st.code(results['evaluation_report'])

def show_ai_report_tab(results):
    """Generate and display an industry-grade AI report powered by Groq."""

    st.markdown("### 🤖 AI-Generated Industry Report")
    st.markdown(
        "Uses the **Groq LLM API** (Llama-3.3-70B) to generate a comprehensive, "
        "professional ML pipeline report — covering methodology, risk assessment, "
        "production readiness, and a full improvement roadmap."
    )

    # ── Status / cached report ──────────────────────────────────────────────── #
    existing_report = st.session_state.get("ai_report")

    if existing_report:
        st.success("✅ AI Report ready. Scroll down to read it or download it from the **Download** section.")
        st.markdown("---")

        # Render the Markdown report inside a styled container
        st.markdown(
            f'<div class="metaflow-report" style="background:rgba(15,23,36,0.6);'
            f'border:1px solid rgba(79,70,229,0.25);border-radius:12px;padding:28px 32px;">'
            f'\n\n{existing_report}\n\n</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        col_regen, _ = st.columns([1, 3])
        with col_regen:
            if st.button("🔄 Regenerate Report", use_container_width=True):
                st.session_state.ai_report = None
                st.rerun()
        return

    # ── Generation UI ────────────────────────────────────────────────────────── #
    st.info(
        "Click **Generate AI Report** to let Groq analyse your pipeline results "
        "and produce a full industry-grade report (takes ~10–20 seconds)."
    )

    # Show which model / key will be used
    api_key_present = bool(os.getenv("GROQ_API_KEY", "").strip())
    if api_key_present:
        st.markdown(
            "<span style='color:#22c55e;font-size:0.9rem'>🔑 GROQ_API_KEY detected from environment</span>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "⚠️ GROQ_API_KEY not found in environment. "
            "Make sure your `.env` file contains `GROQ_API_KEY=<your_key>`."
        )

    if st.button("✨ Generate AI Report", type="primary", use_container_width=False, disabled=not api_key_present):
        with st.spinner("🤖 Groq is analysing your ML pipeline results… (this may take 15–30 seconds)"):
            try:
                reporter = GroqReportGenerator()
                # Enrich results with dataset filename and first-5-rows preview
                enriched_results = dict(results)
                raw_df = st.session_state.get("dataset")
                if raw_df is not None:
                    enriched_results["data_preview"] = raw_df.head(5).to_string()
                enriched_results["file_name"] = st.session_state.get("dataset_filename", "unknown")
                result_meta = reporter.generate_with_metadata(enriched_results)
                st.session_state.ai_report = result_meta["report"]

                # Show token usage
                tokens = result_meta.get("tokens_used", 0)
                finish = result_meta.get("finish_reason", "unknown")
                model_used = result_meta.get("model", reporter.model)

                st.success(
                    f"✅ Report generated! "
                    f"Model: `{model_used}` · "
                    f"Tokens used: `{tokens:,}` · "
                    f"Finish reason: `{finish}`"
                )
                st.rerun()

            except ImportError:
                st.error(
                    "❌ The `groq` Python package is not installed. "
                    "Run `pip install groq` then restart the app."
                )
            except ValueError as exc:
                st.error(f"❌ Configuration error: {exc}")
            except Exception as exc:
                st.error(f"❌ Report generation failed: {exc}")


if __name__ == '__main__':
    main()
