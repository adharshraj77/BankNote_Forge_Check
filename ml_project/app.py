"""
BankNote Authentication — Premium Streamlit Application

A professional, client-ready ML web application for banknote
authentication using wavelet-transformed image features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import io
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import (
    load_data,
    analyze_data_quality,
    detect_outliers_iqr,
    clean_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CLASS_LABELS,
    FEATURE_DESCRIPTIONS,
)
from src.feature_engineering import (
    compute_correlation_matrix,
    compute_mutual_information,
    check_multicollinearity,
    get_feature_importance_summary,
    compute_feature_target_correlation,
)
from src.predict import predict_single, predict_batch, load_artifacts
from src.evaluate import compute_classification_metrics

# ═══════════════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BankNote Authenticator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# Custom CSS — Premium Glassmorphism + Responsive Design
# ═══════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 40%, #0f0c29 100%);
    }

    /* ── Animated Background Orbs ── */
    .bg-orbs {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 0; overflow: hidden;
    }
    .bg-orbs::before, .bg-orbs::after {
        content: ''; position: absolute; border-radius: 50%; filter: blur(100px); opacity: 0.12;
    }
    .bg-orbs::before {
        width: 500px; height: 500px; top: -100px; right: -100px;
        background: radial-gradient(circle, #667eea, transparent);
        animation: orb-float 20s ease-in-out infinite;
    }
    .bg-orbs::after {
        width: 400px; height: 400px; bottom: -100px; left: -100px;
        background: radial-gradient(circle, #764ba2, transparent);
        animation: orb-float 25s ease-in-out infinite reverse;
    }
    @keyframes orb-float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(60px, -40px) scale(1.1); }
        66% { transform: translate(-30px, 50px) scale(0.9); }
    }

    /* ── Main Header ── */
    .main-header {
        background: linear-gradient(135deg, rgba(15,12,41,0.9) 0%, rgba(48,43,99,0.8) 50%, rgba(36,36,62,0.9) 100%);
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(102,126,234,0.2);
        box-shadow: 0 20px 60px rgba(48,43,99,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.5), transparent);
    }
    .main-header h1 {
        color: #ffffff; font-size: 2.4rem; font-weight: 900;
        margin: 0; letter-spacing: -0.03em;
        background: linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.8) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: rgba(255,255,255,0.55); font-size: 1.05rem;
        margin: 0.6rem 0 0 0; font-weight: 300; letter-spacing: 0.01em;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(20,20,40,0.55);
        backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transition: transform 0.3s cubic-bezier(.4,0,.2,1), box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.35);
        border-color: rgba(102,126,234,0.3);
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: rgba(20,20,40,0.6);
        backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        transition: all 0.3s cubic-bezier(.4,0,.2,1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::after {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        opacity: 0; transition: opacity 0.3s ease;
    }
    .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0,0,0,0.35); }
    .kpi-card:hover::after { opacity: 1; }
    .kpi-value {
        font-size: 1.9rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0.3rem 0; line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.78rem; color: rgba(255,255,255,0.45);
        text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
    }
    .kpi-sublabel {
        font-size: 0.72rem; color: rgba(255,255,255,0.3); margin-top: 0.25rem;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.35rem; font-weight: 700; color: #e2e2e2;
        margin: 2.5rem 0 1.2rem 0; padding-bottom: 0.6rem;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, rgba(102,126,234,0.5), transparent) 1;
        display: flex; align-items: center; gap: 0.5rem;
    }

    /* ── Prediction Result Cards ── */
    .prediction-genuine {
        background: linear-gradient(135deg, rgba(13,79,60,0.7) 0%, rgba(26,122,90,0.5) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 20px; padding: 2.5rem; text-align: center;
        box-shadow: 0 12px 48px rgba(16,185,129,0.15);
        animation: result-pop 0.5s cubic-bezier(.4,0,.2,1);
    }
    .prediction-forged {
        background: linear-gradient(135deg, rgba(74,17,17,0.7) 0%, rgba(122,26,26,0.5) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 20px; padding: 2.5rem; text-align: center;
        box-shadow: 0 12px 48px rgba(239,68,68,0.15);
        animation: result-pop 0.5s cubic-bezier(.4,0,.2,1);
    }
    @keyframes result-pop {
        0% { transform: scale(0.9); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    .prediction-label { font-size: 2.2rem; font-weight: 800; margin: 0.5rem 0; }
    .prediction-confidence { font-size: 1rem; opacity: 0.8; }

    /* ── Model Card ── */
    .model-card {
        background: rgba(20,20,40,0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 1.6rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        transition: all 0.35s cubic-bezier(.4,0,.2,1);
        position: relative; overflow: hidden;
    }
    .model-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 48px rgba(102,126,234,0.2);
        border-color: rgba(102,126,234,0.35);
    }
    .model-card-best {
        border-color: rgba(16,185,129,0.4);
        background: rgba(16,185,129,0.06);
    }
    .model-card-best::before {
        content: '🏆 BEST MODEL'; position: absolute; top: 0; right: 0;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white; font-size: 0.65rem; font-weight: 700;
        padding: 0.3rem 0.8rem; border-radius: 0 16px 0 12px;
        letter-spacing: 0.08em;
    }
    .model-card h4 {
        color: #e2e2e2; margin: 0 0 0.8rem 0; font-weight: 700; font-size: 1.05rem;
    }
    .model-stat { display: flex; justify-content: space-between; padding: 0.35rem 0; }
    .model-stat-label { color: rgba(255,255,255,0.5); font-size: 0.82rem; }
    .model-stat-value { color: #e2e2e2; font-size: 0.85rem; font-weight: 600; }

    /* ── Rank Badge ── */
    .rank-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 32px; height: 32px; border-radius: 50%; font-weight: 800;
        font-size: 0.85rem; color: white;
    }
    .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); }
    .rank-2 { background: linear-gradient(135deg, #C0C0C0, #A0A0A0); }
    .rank-3 { background: linear-gradient(135deg, #CD7F32, #A0522D); }
    .rank-other { background: rgba(102,126,234,0.3); }

    /* ── Info Card ── */
    .info-card {
        background: rgba(30,30,55,0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 1.3rem; margin-bottom: 1rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #0f0c29 50%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] .stMarkdown { color: rgba(255,255,255,0.85); }

    /* ── Table / DataFrame ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Hide defaults ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Metric cards ── */
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 10px 22px;
        font-weight: 600; transition: all 0.2s ease;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.5rem; }
        .main-header p { font-size: 0.9rem; }
        .main-header { padding: 1.5rem 1.2rem; }
        .kpi-value { font-size: 1.4rem; }
        .kpi-card { padding: 1rem 0.8rem; }
        .section-header { font-size: 1.1rem; }
        .prediction-label { font-size: 1.6rem; }
        .model-card { padding: 1.2rem; }
    }
    @media (max-width: 480px) {
        .main-header h1 { font-size: 1.2rem; }
        .kpi-value { font-size: 1.2rem; }
        .kpi-label { font-size: 0.68rem; }
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
    ::-webkit-scrollbar-thumb { background: rgba(102,126,234,0.4); border-radius: 3px; }

    /* ── Stat pill ── */
    .stat-pill {
        display: inline-block; padding: 0.25rem 0.8rem;
        border-radius: 20px; font-size: 0.78rem; font-weight: 600;
        margin: 0.15rem;
    }
    .pill-green { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }
    .pill-red   { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25);  }
    .pill-blue  { background: rgba(59,130,246,0.15);  color: #3b82f6; border: 1px solid rgba(59,130,246,0.25);  }
    .pill-purple{ background: rgba(118,75,162,0.15);  color: #a78bfa; border: 1px solid rgba(118,75,162,0.25);  }
    .pill-amber { background: rgba(245,158,11,0.15);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.25);  }
</style>
<div class="bg-orbs"></div>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
# Color Palette & Layout Config
# ═══════════════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "info": "#3b82f6",
    "genuine": "#10b981",
    "forged": "#ef4444",
    "gradient": ["#667eea", "#764ba2", "#f093fb", "#f5576c"],
    "chart_palette": [
        "#667eea", "#f5576c", "#4facfe", "#00f2fe",
        "#f093fb", "#fbc2eb", "#ffecd2", "#a18cd1",
        "#43e97b",
    ],
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", family="Inter, sans-serif"),
)


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════


@st.cache_data
def load_dataset():
    """Load and cache the dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "BankNoteAuthentication.csv")
    return load_data(data_path)


@st.cache_resource
def load_model_artifacts():
    """Load and cache model artifacts."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    try:
        artifacts = load_artifacts(model_dir)

        # Load additional metadata
        metrics_path = os.path.join(model_dir, "test_metrics.json")
        comparison_path = os.path.join(model_dir, "model_comparison.csv")
        importance_path = os.path.join(model_dir, "feature_importance.csv")
        overfit_path = os.path.join(model_dir, "overfitting_check.json")

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                artifacts["metrics"] = json.load(f)

        if os.path.exists(comparison_path):
            artifacts["comparison"] = pd.read_csv(comparison_path)

        if os.path.exists(importance_path):
            artifacts["importance"] = pd.read_csv(importance_path, index_col=0)

        if os.path.exists(overfit_path):
            with open(overfit_path) as f:
                artifacts["overfitting"] = json.load(f)

        return artifacts
    except FileNotFoundError:
        return None


def render_kpi_card(label: str, value: str, sublabel: str = ""):
    """Render a styled KPI card."""
    sublabel_html = f'<div class="kpi-sublabel">{sublabel}</div>' if sublabel else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sublabel_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_gauge_chart(value: float, title: str, color: str = COLORS["primary"]) -> go.Figure:
    """Create a gauge chart for metrics."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "white"}},
            title={"text": title, "font": {"size": 14, "color": "rgba(255,255,255,0.6)"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)"},
                "bar": {"color": color, "thickness": 0.8},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 60], "color": "rgba(239,68,68,0.12)"},
                    {"range": [60, 80], "color": "rgba(245,158,11,0.12)"},
                    {"range": [80, 100], "color": "rgba(16,185,129,0.12)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.8,
                    "value": value * 100,
                },
            },
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=250,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


def render_model_card(model_name, acc, f1, roc, time_s, rank, is_best=False):
    """Render a styled model card."""
    best_cls = " model-card-best" if is_best else ""
    rank_cls = f"rank-{rank}" if rank <= 3 else "rank-other"
    st.markdown(
        f"""
        <div class="model-card{best_cls}">
            <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.8rem;">
                <span class="rank-badge {rank_cls}">{rank}</span>
                <h4 style="margin:0;color:#e2e2e2;">{model_name}</h4>
            </div>
            <div class="model-stat">
                <span class="model-stat-label">CV ROC-AUC</span>
                <span class="model-stat-value">{roc:.4f}</span>
            </div>
            <div class="model-stat">
                <span class="model-stat-label">CV Accuracy</span>
                <span class="model-stat-value">{acc:.4f}</span>
            </div>
            <div class="model-stat">
                <span class="model-stat-label">CV F1-Score</span>
                <span class="model-stat-value">{f1:.4f}</span>
            </div>
            <div class="model-stat" style="border-top:1px solid rgba(255,255,255,0.06);padding-top:0.5rem;margin-top:0.3rem;">
                <span class="model-stat-label">Train Time</span>
                <span class="model-stat-value">{time_s:.3f}s</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <div style="font-size:2.2rem;">🏦</div>
            <div style="font-size:1.1rem;font-weight:800;color:white;letter-spacing:-0.02em;">
                BankNote Auth
            </div>
            <div style="font-size:0.72rem;color:rgba(255,255,255,0.4);letter-spacing:0.08em;text-transform:uppercase;">
                ML Classification Suite
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "📊 Dashboard",
            "🔮 Predict",
            "⚔️ Model Comparison",
            "🧠 Model Insights",
            "🔍 Data Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
    <div style="padding:1rem;background:rgba(102,126,234,0.08);border-radius:12px;
        border:1px solid rgba(102,126,234,0.15);margin-top:0.5rem;">
        <p style="color:rgba(255,255,255,0.6);font-size:0.78rem;margin:0;line-height:1.6;">
            <strong style="color:rgba(255,255,255,0.8);">About</strong><br>
            ML-powered banknote authentication using wavelet-transformed image features.
            Classifies banknotes as <span style="color:#10b981;font-weight:600;">Genuine</span>
            or <span style="color:#ef4444;font-weight:600;">Forged</span>.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="position:absolute;bottom:1rem;left:1rem;right:1rem;text-align:center;">
        <p style="color:rgba(255,255,255,0.2);font-size:0.65rem;margin:0;">
            Built with Streamlit • v2.0
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# Load Data & Model
# ═══════════════════════════════════════════════════════════════════════

try:
    df = load_dataset()
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

artifacts = load_model_artifacts()

# ═══════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═══════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":
    st.markdown(
        """
    <div class="main-header">
        <h1>📊 BankNote Authentication Dashboard</h1>
        <p>Real-time analytics, model performance, and dataset intelligence</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ─── Executive KPIs ───
    quality = analyze_data_quality(df)

    if artifacts and "metrics" in artifacts:
        metrics = artifacts["metrics"]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_kpi_card("Accuracy", f"{metrics['accuracy']:.1%}", "Test Set")
        with col2:
            render_kpi_card("F1-Score", f"{metrics['f1_score']:.1%}", "Harmonic Mean")
        with col3:
            render_kpi_card("ROC-AUC", f"{metrics.get('roc_auc', 0):.1%}", "Discrimination")
        with col4:
            render_kpi_card("Precision", f"{metrics['precision']:.1%}", "Positive Predictive")
        with col5:
            render_kpi_card("Recall", f"{metrics['recall']:.1%}", "Sensitivity")
    else:
        st.warning("⚠️ Model not trained yet. Run `python run_training.py` first.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi_card("Samples", f"{len(df):,}", "Total Records")
        with col2:
            render_kpi_card("Features", str(len(FEATURE_COLUMNS)), "Input Variables")
        with col3:
            render_kpi_card("Genuine", str(quality["class_distribution"].get(0, 0)), "Class 0")
        with col4:
            render_kpi_card("Forged", str(quality["class_distribution"].get(1, 0)), "Class 1")

    st.markdown("")

    # ─── Dataset Overview & Class Distribution ───
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

        overview_data = {
            "Metric": [
                "Total Samples", "Features", "Missing Values",
                "Duplicate Records", "Genuine Notes (Class 0)",
                "Forged Notes (Class 1)", "Class Balance Ratio",
            ],
            "Value": [
                f"{len(df):,}", str(len(FEATURE_COLUMNS)),
                str(sum(quality["missing_values"].values())),
                f"{quality['duplicates']} ({quality['duplicate_percentage']:.1f}%)",
                str(quality["class_distribution"].get(0, 0)),
                str(quality["class_distribution"].get(1, 0)),
                f"{quality['class_balance_ratio']:.3f}",
            ],
        }
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)

    with col_right:
        st.markdown('<div class="section-header">⚖️ Class Distribution</div>', unsafe_allow_html=True)

        class_data = pd.DataFrame({
            "Class": ["Genuine (0)", "Forged (1)"],
            "Count": [
                quality["class_distribution"].get(0, 0),
                quality["class_distribution"].get(1, 0),
            ],
        })
        fig_class = px.pie(
            class_data, values="Count", names="Class", color="Class",
            color_discrete_map={"Genuine (0)": COLORS["genuine"], "Forged (1)": COLORS["forged"]},
            hole=0.6,
        )
        fig_class.update_layout(**PLOTLY_LAYOUT, height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
        )
        fig_class.update_traces(textinfo="percent+value", textfont_size=14)
        st.plotly_chart(fig_class, use_container_width=True)

    # ─── Feature Distributions ───
    st.markdown('<div class="section-header">📈 Feature Distributions by Class</div>', unsafe_allow_html=True)

    fig_dist = make_subplots(rows=2, cols=2, subplot_titles=FEATURE_COLUMNS, vertical_spacing=0.14, horizontal_spacing=0.08)
    for i, col in enumerate(FEATURE_COLUMNS):
        row, col_idx = (i // 2) + 1, (i % 2) + 1
        for cls, color, name in [(0, COLORS["genuine"], "Genuine"), (1, COLORS["forged"], "Forged")]:
            fig_dist.add_trace(
                go.Histogram(
                    x=df[df[TARGET_COLUMN] == cls][col], name=name,
                    marker_color=color, opacity=0.7, showlegend=(i == 0), nbinsx=40,
                ), row=row, col=col_idx,
            )

    fig_dist.update_layout(**PLOTLY_LAYOUT, height=520, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ─── Quick Model Comparison ───
    if artifacts and "comparison" in artifacts:
        st.markdown('<div class="section-header">🏆 Model Performance Overview</div>', unsafe_allow_html=True)

        comp = artifacts["comparison"]
        fig_comp = go.Figure()
        for metric_col, name, color in [
            ("CV_ROC_AUC_Mean", "CV ROC-AUC", COLORS["primary"]),
            ("Val_Accuracy", "Val Accuracy", COLORS["secondary"]),
            ("Val_F1", "Val F1", "#f093fb"),
        ]:
            fig_comp.add_trace(go.Bar(
                name=name, x=comp["Model"], y=comp[metric_col],
                marker_color=color, marker_line_width=0,
                **({"error_y": dict(type="data", array=comp["CV_ROC_AUC_Std"], visible=True)}
                   if metric_col == "CV_ROC_AUC_Mean" else {}),
            ))
        fig_comp.update_layout(**PLOTLY_LAYOUT, barmode="group", height=420,
            yaxis_title="Score",
            yaxis_range=[max(0, comp[["CV_ROC_AUC_Mean", "Val_Accuracy", "Val_F1"]].min().min() - 0.05), 1.02],
            legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.5, xanchor="center"),
            margin=dict(l=40, r=20, t=30, b=90), xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # ─── Correlation Heatmap ───
    st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)

    corr_matrix = compute_correlation_matrix(df)
    fig_corr = px.imshow(
        corr_matrix, text_auto=".3f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="equal",
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=400, margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: Predict
# ═══════════════════════════════════════════════════════════════════════

elif page == "🔮 Predict":
    st.markdown(
        """
    <div class="main-header">
        <h1>🔮 Prediction Workspace</h1>
        <p>Authenticate banknotes using wavelet-transformed image features</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if artifacts is None:
        st.error("❌ Model not available. Please run `python run_training.py` first.")
        st.stop()

    model = artifacts["model"]
    pipeline = artifacts["pipeline"]

    tab1, tab2 = st.tabs(["🔢 Single Prediction", "📁 Batch Prediction"])

    # ─── Single Prediction ───
    with tab1:
        st.markdown("### Enter Banknote Features")
        st.markdown(
            """
        <div class="info-card">
            <p style="color:rgba(255,255,255,0.6);margin:0;font-size:0.85rem;line-height:1.6;">
                Enter the four wavelet-transformed features extracted from a banknote image.
                The model will classify whether the note is
                <span style="color:#10b981;font-weight:600;">Genuine</span> or
                <span style="color:#ef4444;font-weight:600;">Forged</span>.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            variance = st.number_input(
                "📊 Variance of Wavelet Transform", value=0.0, step=0.1, format="%.4f",
                help="Variance of the Wavelet Transformed image. Typical range: -7.0 to 7.0",
            )
            curtosis = st.number_input(
                "📐 Kurtosis of Wavelet Transform", value=0.0, step=0.1, format="%.4f",
                help="Kurtosis of the Wavelet Transformed image. Typical range: -5.0 to 18.0",
            )
        with col2:
            skewness = st.number_input(
                "📈 Skewness of Wavelet Transform", value=0.0, step=0.1, format="%.4f",
                help="Skewness of the Wavelet Transformed image. Typical range: -14.0 to 13.0",
            )
            entropy = st.number_input(
                "🔑 Entropy of Image", value=0.0, step=0.1, format="%.4f",
                help="Entropy of the image. Typical range: -8.5 to 2.5",
            )

        st.markdown("")
        if st.button("🔍 Authenticate Banknote", type="primary", use_container_width=True):
            with st.spinner("Analyzing banknote features..."):
                input_data = {"variance": variance, "skewness": skewness, "curtosis": curtosis, "entropy": entropy}
                result = predict_single(input_data, model, pipeline)

                if result["success"]:
                    st.markdown("")
                    if result["label"] == "Genuine":
                        icon, css_class, color = "✅", "prediction-genuine", COLORS["genuine"]
                    else:
                        icon, css_class, color = "⚠️", "prediction-forged", COLORS["forged"]

                    confidence = result.get("confidence", 0) * 100
                    st.markdown(
                        f"""
                    <div class="{css_class}">
                        <div style="font-size:3rem;">{icon}</div>
                        <div class="prediction-label" style="color:{color};">{result['label']} Banknote</div>
                        <div class="prediction-confidence" style="color:rgba(255,255,255,0.7);">
                            Confidence: {confidence:.1f}%
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("")

                    # Probability breakdown
                    if "probability_genuine" in result:
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            st.plotly_chart(create_gauge_chart(result["probability_genuine"], "Genuine Probability", COLORS["genuine"]), use_container_width=True)
                        with col_p2:
                            st.plotly_chart(create_gauge_chart(result["probability_forged"], "Forged Probability", COLORS["forged"]), use_container_width=True)

                    # Feature context
                    st.markdown("#### 📊 Input Feature Analysis")
                    feature_stats = df[FEATURE_COLUMNS].describe()
                    context_data = []
                    for feat in FEATURE_COLUMNS:
                        val = input_data[feat]
                        feat_min, feat_max = feature_stats[feat]["min"], feature_stats[feat]["max"]
                        feat_mean = feature_stats[feat]["mean"]
                        pct = (val - feat_min) / (feat_max - feat_min) * 100 if feat_max != feat_min else 50
                        context_data.append({
                            "Feature": feat.capitalize(), "Input Value": f"{val:.4f}",
                            "Dataset Mean": f"{feat_mean:.4f}", "Min": f"{feat_min:.4f}",
                            "Max": f"{feat_max:.4f}", "Percentile Position": f"{pct:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(context_data), use_container_width=True, hide_index=True)

                    result_json = json.dumps(result, indent=2, default=str)
                    st.download_button("📥 Download Result (JSON)", data=result_json, file_name="prediction_result.json", mime="application/json")
                else:
                    for error in result["errors"]:
                        st.error(f"❌ {error}")

    # ─── Batch Prediction ───
    with tab2:
        st.markdown("### Upload Dataset for Batch Prediction")
        st.markdown(
            """
        <div class="info-card">
            <p style="color:rgba(255,255,255,0.6);margin:0;font-size:0.85rem;line-height:1.6;">
                Upload a CSV file with columns: <code>variance</code>, <code>skewness</code>,
                <code>curtosis</code>, <code>entropy</code>. The model will predict the
                authenticity of each banknote in the file.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(upload_df)} records")
                missing_cols = set(FEATURE_COLUMNS) - set(upload_df.columns)
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    if st.button("🔍 Run Batch Prediction", type="primary", use_container_width=True):
                        with st.spinner("Processing batch predictions..."):
                            results_df = predict_batch(upload_df, model, pipeline)
                            n_genuine = (results_df["prediction"] == 0).sum()
                            n_forged = (results_df["prediction"] == 1).sum()
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                render_kpi_card("Total", str(len(results_df)), "Processed")
                            with col2:
                                render_kpi_card("Genuine", str(n_genuine), f"{n_genuine/len(results_df)*100:.1f}%")
                            with col3:
                                render_kpi_card("Forged", str(n_forged), f"{n_forged/len(results_df)*100:.1f}%")
                            st.markdown("")
                            st.dataframe(results_df, use_container_width=True, height=400)
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            st.download_button("📥 Download Predictions (CSV)", data=csv_buffer.getvalue(), file_name="batch_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")


# ═══════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison  ★ NEW ★
# ═══════════════════════════════════════════════════════════════════════

elif page == "⚔️ Model Comparison":
    st.markdown(
        """
    <div class="main-header">
        <h1>⚔️ Model Comparison Arena</h1>
        <p>Head-to-head performance analysis across all trained classifiers</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if artifacts is None or "comparison" not in artifacts:
        st.error("❌ Model comparison data not available. Please run `python run_training.py` first.")
        st.stop()

    comp = artifacts["comparison"].copy()
    best_model_name = artifacts.get("metrics", {}).get("best_model_name", comp.iloc[0]["Model"])

    # ─── Ranking Table (top KPIs) ───
    st.markdown('<div class="section-header">🏅 Overall Ranking</div>', unsafe_allow_html=True)

    comp_sorted = comp.sort_values("CV_ROC_AUC_Mean", ascending=False).reset_index(drop=True)

    # Render model cards in a 3-col grid
    n_models = len(comp_sorted)
    for row_start in range(0, n_models, 3):
        cols = st.columns(3)
        for ci, col in enumerate(cols):
            idx = row_start + ci
            if idx >= n_models:
                break
            r = comp_sorted.iloc[idx]
            with col:
                render_model_card(
                    r["Model"], r["CV_Accuracy_Mean"], r["CV_F1_Mean"],
                    r["CV_ROC_AUC_Mean"], r["Training_Time_s"],
                    rank=idx + 1, is_best=(r["Model"] == best_model_name),
                )

    st.markdown("")

    # ─── Radar Chart ───
    st.markdown('<div class="section-header">🕸️ Multi-Metric Radar Comparison</div>', unsafe_allow_html=True)

    radar_metrics = ["CV_Accuracy_Mean", "CV_F1_Mean", "CV_ROC_AUC_Mean", "Val_Accuracy", "Val_F1"]
    radar_labels = ["CV Accuracy", "CV F1", "CV ROC-AUC", "Val Accuracy", "Val F1"]

    # Let user select models to compare
    default_models = list(comp_sorted["Model"].head(4))
    selected_models = st.multiselect(
        "Select models to compare on radar", comp_sorted["Model"].tolist(),
        default=default_models, max_selections=6,
    )

    if selected_models:
        fig_radar = go.Figure()
        radar_colors = COLORS["chart_palette"]
        for mi, m_name in enumerate(selected_models):
            row = comp_sorted[comp_sorted["Model"] == m_name].iloc[0]
            values = [row[c] for c in radar_metrics]
            values.append(values[0])  # close the polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=radar_labels + [radar_labels[0]],
                fill="toself", fillcolor=f"rgba({int(radar_colors[mi % len(radar_colors)][1:3], 16)},{int(radar_colors[mi % len(radar_colors)][3:5], 16)},{int(radar_colors[mi % len(radar_colors)][5:7], 16)},0.08)",
                line=dict(color=radar_colors[mi % len(radar_colors)], width=2),
                name=m_name,
            ))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT, height=500,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[
                    max(0, comp_sorted[radar_metrics].min().min() - 0.05), 1.01
                ], gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
            margin=dict(l=60, r=60, t=40, b=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ─── Performance Heatmap ───
    st.markdown('<div class="section-header">🔥 Performance Heatmap</div>', unsafe_allow_html=True)

    heatmap_cols = ["CV_Accuracy_Mean", "CV_F1_Mean", "CV_ROC_AUC_Mean", "Val_Accuracy", "Val_F1", "Val_ROC_AUC"]
    heatmap_labels = ["CV Acc", "CV F1", "CV AUC", "Val Acc", "Val F1", "Val AUC"]
    heatmap_data = comp_sorted[heatmap_cols].values

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=heatmap_labels,
        y=comp_sorted["Model"],
        text=np.round(heatmap_data, 4),
        texttemplate="%{text:.4f}",
        textfont={"size": 11, "color": "white"},
        colorscale=[[0, "#1a1a2e"], [0.5, "#667eea"], [1, "#10b981"]],
        showscale=True,
        colorbar=dict(title="Score", titlefont=dict(color="white"), tickfont=dict(color="white")),
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=420,
        margin=dict(l=140, r=20, t=30, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ─── Cross-Validation Stability (error bars) ───
    st.markdown('<div class="section-header">📊 Cross-Validation Stability</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="info-card">
        <p style="color:rgba(255,255,255,0.6);margin:0;font-size:0.85rem;line-height:1.6;">
            Lower error bars = more consistent performance across folds.
            A model that's both <strong>high-scoring</strong> and <strong>stable</strong> is ideal for production.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig_stability = go.Figure()
    for metric, std_col, name, color in [
        ("CV_Accuracy_Mean", "CV_Accuracy_Std", "Accuracy", COLORS["primary"]),
        ("CV_F1_Mean", "CV_F1_Std", "F1-Score", "#f093fb"),
        ("CV_ROC_AUC_Mean", "CV_ROC_AUC_Std", "ROC-AUC", COLORS["success"]),
    ]:
        fig_stability.add_trace(go.Scatter(
            x=comp_sorted["Model"], y=comp_sorted[metric],
            error_y=dict(type="data", array=comp_sorted[std_col], visible=True, color=color),
            mode="markers+lines", name=name,
            marker=dict(size=10, color=color), line=dict(color=color, width=2),
        ))
    fig_stability.update_layout(**PLOTLY_LAYOUT, height=400,
        yaxis_title="Score", yaxis_range=[max(0, comp_sorted["CV_Accuracy_Mean"].min() - 0.05), 1.02],
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.5, xanchor="center"),
        margin=dict(l=50, r=20, t=40, b=90), xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_stability, use_container_width=True)

    # ─── Training Time Analysis ───
    st.markdown('<div class="section-header">⏱️ Training Time vs Performance</div>', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        fig_time = px.scatter(
            comp_sorted, x="Training_Time_s", y="CV_ROC_AUC_Mean",
            size="CV_F1_Mean", color="Model", text="Model",
            size_max=30,
            color_discrete_sequence=COLORS["chart_palette"],
        )
        fig_time.update_traces(textposition="top center", textfont=dict(size=9, color="rgba(255,255,255,0.7)"))
        fig_time.update_layout(**PLOTLY_LAYOUT, height=400,
            xaxis_title="Training Time (seconds)", yaxis_title="CV ROC-AUC",
            yaxis_range=[max(0, comp_sorted["CV_ROC_AUC_Mean"].min() - 0.03), 1.01],
            showlegend=False, margin=dict(l=50, r=20, t=30, b=50),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with col_t2:
        st.markdown(
            """
        <div class="info-card">
            <h4 style="color:#667eea;margin-top:0;">⚡ Efficiency Analysis</h4>
            <p style="color:rgba(255,255,255,0.6);font-size:0.85rem;line-height:1.7;">
                Bubble size represents F1-Score. The ideal model sits in the
                <strong>top-left</strong> corner — high performance with minimal training time.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        # Quick stats
        fastest = comp_sorted.loc[comp_sorted["Training_Time_s"].idxmin()]
        slowest = comp_sorted.loc[comp_sorted["Training_Time_s"].idxmax()]
        st.markdown(
            f"""
        <div class="info-card" style="margin-top:0.5rem;">
            <div class="model-stat"><span class="model-stat-label">⚡ Fastest</span>
                <span class="model-stat-value">{fastest['Model']} ({fastest['Training_Time_s']:.3f}s)</span></div>
            <div class="model-stat"><span class="model-stat-label">🐢 Slowest</span>
                <span class="model-stat-value">{slowest['Model']} ({slowest['Training_Time_s']:.3f}s)</span></div>
            <div class="model-stat"><span class="model-stat-label">📏 Avg Time</span>
                <span class="model-stat-value">{comp_sorted['Training_Time_s'].mean():.3f}s</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ─── Detailed Comparison Table ───
    st.markdown('<div class="section-header">📋 Detailed Metrics Table</div>', unsafe_allow_html=True)

    display_df = comp_sorted.copy()
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))
    display_df = display_df.rename(columns={
        "CV_Accuracy_Mean": "CV Acc (μ)", "CV_Accuracy_Std": "CV Acc (σ)",
        "CV_F1_Mean": "CV F1 (μ)", "CV_F1_Std": "CV F1 (σ)",
        "CV_ROC_AUC_Mean": "CV AUC (μ)", "CV_ROC_AUC_Std": "CV AUC (σ)",
        "Val_Accuracy": "Val Acc", "Val_F1": "Val F1",
        "Val_ROC_AUC": "Val AUC", "Training_Time_s": "Time (s)",
    })
    st.dataframe(
        display_df.style.format({
            "CV Acc (μ)": "{:.4f}", "CV Acc (σ)": "{:.4f}",
            "CV F1 (μ)": "{:.4f}", "CV F1 (σ)": "{:.4f}",
            "CV AUC (μ)": "{:.4f}", "CV AUC (σ)": "{:.4f}",
            "Val Acc": "{:.4f}", "Val F1": "{:.4f}", "Val AUC": "{:.4f}",
            "Time (s)": "{:.4f}",
        }),
        use_container_width=True, hide_index=True, height=380,
    )

    # ─── Head-to-Head Comparison ───
    st.markdown('<div class="section-header">🤼 Head-to-Head Comparison</div>', unsafe_allow_html=True)

    h2h_col1, h2h_col2 = st.columns(2)
    model_names = comp_sorted["Model"].tolist()
    with h2h_col1:
        model_a = st.selectbox("Model A", model_names, index=0)
    with h2h_col2:
        model_b = st.selectbox("Model B", model_names, index=min(1, len(model_names) - 1))

    if model_a and model_b:
        row_a = comp_sorted[comp_sorted["Model"] == model_a].iloc[0]
        row_b = comp_sorted[comp_sorted["Model"] == model_b].iloc[0]

        compare_metrics = [
            ("CV Accuracy", "CV_Accuracy_Mean"), ("CV F1-Score", "CV_F1_Mean"),
            ("CV ROC-AUC", "CV_ROC_AUC_Mean"), ("Val Accuracy", "Val_Accuracy"),
            ("Val F1", "Val_F1"), ("Training Time", "Training_Time_s"),
        ]

        h2h_rows = ""
        for label, col_key in compare_metrics:
            va, vb = row_a[col_key], row_b[col_key]
            if col_key == "Training_Time_s":
                # Lower is better for time
                winner = "a" if va < vb else ("b" if vb < va else "tie")
                fmt = ".4f"
            else:
                winner = "a" if va > vb else ("b" if vb > va else "tie")
                fmt = ".4f"
            color_a = "#10b981" if winner == "a" else ("#ef4444" if winner == "b" else "rgba(255,255,255,0.7)")
            color_b = "#10b981" if winner == "b" else ("#ef4444" if winner == "a" else "rgba(255,255,255,0.7)")
            diff = va - vb
            diff_str = f"+{diff:{fmt}}" if diff > 0 else f"{diff:{fmt}}"
            diff_color = "#10b981" if (diff > 0 and col_key != "Training_Time_s") or (diff < 0 and col_key == "Training_Time_s") else "#ef4444"

            h2h_rows += f"""
            <tr>
                <td style="color:{color_a};font-weight:600;text-align:right;padding:0.5rem 1rem;">{va:{fmt}}</td>
                <td style="color:rgba(255,255,255,0.55);text-align:center;padding:0.5rem 0.8rem;font-size:0.85rem;">{label}</td>
                <td style="color:{color_b};font-weight:600;text-align:left;padding:0.5rem 1rem;">{vb:{fmt}}</td>
                <td style="color:{diff_color};font-size:0.8rem;text-align:center;padding:0.5rem;">{diff_str}</td>
            </tr>"""

        st.markdown(
            f"""
        <div class="glass-card" style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;">
                <thead>
                    <tr>
                        <th style="color:#667eea;text-align:right;padding:0.6rem 1rem;font-size:0.9rem;">{model_a}</th>
                        <th style="color:rgba(255,255,255,0.4);text-align:center;padding:0.6rem;font-size:0.8rem;">METRIC</th>
                        <th style="color:#764ba2;text-align:left;padding:0.6rem 1rem;font-size:0.9rem;">{model_b}</th>
                        <th style="color:rgba(255,255,255,0.4);text-align:center;padding:0.6rem;font-size:0.75rem;">DIFF</th>
                    </tr>
                </thead>
                <tbody>{h2h_rows}</tbody>
            </table>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# PAGE: Model Insights
# ═══════════════════════════════════════════════════════════════════════

elif page == "🧠 Model Insights":
    st.markdown(
        """
    <div class="main-header">
        <h1>🧠 Model Insights</h1>
        <p>Deep dive into model performance, feature importance, and diagnostics</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if artifacts is None:
        st.error("❌ Model not available. Please run `python run_training.py` first.")
        st.stop()

    metrics = artifacts.get("metrics", {})

    # ─── Performance Gauges ───
    st.markdown('<div class="section-header">📈 Performance Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.plotly_chart(create_gauge_chart(metrics.get("accuracy", 0), "Accuracy", COLORS["primary"]), use_container_width=True)
    with col2:
        st.plotly_chart(create_gauge_chart(metrics.get("precision", 0), "Precision", COLORS["info"]), use_container_width=True)
    with col3:
        st.plotly_chart(create_gauge_chart(metrics.get("recall", 0), "Recall", COLORS["warning"]), use_container_width=True)
    with col4:
        st.plotly_chart(create_gauge_chart(metrics.get("f1_score", 0), "F1-Score", COLORS["success"]), use_container_width=True)

    # ─── Confusion Matrix ───
    st.markdown('<div class="section-header">🎯 Confusion Matrix</div>', unsafe_allow_html=True)

    if "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted Genuine", "Predicted Forged"],
            y=["Actual Genuine", "Actual Forged"],
            text=cm, texttemplate="%{text}", textfont={"size": 28, "color": "white"},
            colorscale=[[0, "#1a1a2e"], [1, "#667eea"]], showscale=False,
        ))
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=350, margin=dict(l=40, r=20, t=30, b=40), xaxis=dict(side="bottom"))

        col_cm, col_insight = st.columns([3, 2])
        with col_cm:
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_insight:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            total = tn + fp + fn + tp
            st.markdown(
                f"""
            <div class="info-card">
                <h4 style="color:#667eea;margin-top:0;">Interpretation</h4>
                <p style="color:rgba(255,255,255,0.7);font-size:0.9rem;line-height:1.8;">
                    <span style="color:#10b981;">✅ True Negatives:</span> {tn} genuine notes correctly identified<br>
                    <span style="color:#10b981;">✅ True Positives:</span> {tp} forged notes correctly caught<br>
                    <span style="color:#ef4444;">❌ False Positives:</span> {fp} genuine notes incorrectly flagged<br>
                    <span style="color:#ef4444;">❌ False Negatives:</span> {fn} forged notes that slipped through<br><br>
                    <strong>Overall Accuracy:</strong> {(tn+tp)/total*100:.1f}%
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ─── Feature Importance ───
    st.markdown('<div class="section-header">🔑 Feature Importance</div>', unsafe_allow_html=True)

    if "importance" in artifacts:
        imp_df = artifacts["importance"]
        col_imp1, col_imp2 = st.columns(2)
        with col_imp1:
            if "permutation_importance" in imp_df.columns:
                imp_sorted = imp_df.sort_values("permutation_importance", ascending=True)
                fig_imp = go.Figure(go.Bar(
                    x=imp_sorted["permutation_importance"], y=imp_sorted.index, orientation="h",
                    marker=dict(color=imp_sorted["permutation_importance"], colorscale=[[0, "#667eea"], [1, "#f5576c"]]),
                    error_x=dict(type="data", array=imp_sorted.get("permutation_std", pd.Series([0]*len(imp_sorted))), visible=True),
                ))
                fig_imp.update_layout(**PLOTLY_LAYOUT, title="Permutation Importance", height=300, margin=dict(l=100, r=20, t=50, b=40), xaxis_title="Importance")
                st.plotly_chart(fig_imp, use_container_width=True)
        with col_imp2:
            if "model_importance" in imp_df.columns:
                imp_sorted_m = imp_df.sort_values("model_importance", ascending=True)
                fig_imp_m = go.Figure(go.Bar(
                    x=imp_sorted_m["model_importance"], y=imp_sorted_m.index, orientation="h",
                    marker=dict(color=imp_sorted_m["model_importance"], colorscale=[[0, "#764ba2"], [1, "#f093fb"]]),
                ))
                fig_imp_m.update_layout(**PLOTLY_LAYOUT, title="Model-Based Importance (Gini/Gain)", height=300, margin=dict(l=100, r=20, t=50, b=40), xaxis_title="Importance")
                st.plotly_chart(fig_imp_m, use_container_width=True)

        st.markdown(
            """
        <div class="info-card">
            <h4 style="color:#667eea;margin-top:0;">📖 What This Means</h4>
            <p style="color:rgba(255,255,255,0.7);font-size:0.9rem;line-height:1.7;">
                <strong>Permutation Importance</strong> measures how much model performance drops when a feature's values
                are randomly shuffled — the larger the drop, the more important the feature.<br><br>
                <strong>Model-Based Importance</strong> (for tree models) shows how much each feature contributes to
                reducing prediction error across all decision splits.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ─── Overfitting Check ───
    if "overfitting" in artifacts:
        st.markdown('<div class="section-header">⚖️ Overfitting Diagnostics</div>', unsafe_allow_html=True)

        overfit = artifacts["overfitting"]
        col_o1, col_o2, col_o3 = st.columns(3)
        with col_o1:
            render_kpi_card("Train Accuracy", f"{overfit['train_accuracy']:.1%}", "Training Set")
        with col_o2:
            render_kpi_card("Val Accuracy", f"{overfit['val_accuracy']:.1%}", "Validation Set")
        with col_o3:
            render_kpi_card("Gap", f"{overfit['accuracy_gap']:.1%}", overfit["diagnosis"][:50])

        fig_overfit = go.Figure()
        fig_overfit.add_trace(go.Bar(name="Training", x=["Accuracy", "F1-Score"], y=[overfit["train_accuracy"], overfit["train_f1"]], marker_color=COLORS["primary"]))
        fig_overfit.add_trace(go.Bar(name="Validation", x=["Accuracy", "F1-Score"], y=[overfit["val_accuracy"], overfit["val_f1"]], marker_color=COLORS["secondary"]))
        fig_overfit.update_layout(**PLOTLY_LAYOUT, barmode="group", height=300,
            yaxis_range=[max(0, min(overfit["val_accuracy"], overfit["val_f1"]) - 0.05), 1.02],
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_overfit, use_container_width=True)

    # ─── Selected Model Info ───
    st.markdown('<div class="section-header">🏅 Selected Model</div>', unsafe_allow_html=True)

    model_name = metrics.get("best_model_name", "Unknown")
    best_params = metrics.get("best_params", {})

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            f"""
        <div class="info-card">
            <h4 style="color:#667eea;margin-top:0;">Model: {model_name}</h4>
            <p style="color:rgba(255,255,255,0.7);font-size:0.9rem;line-height:1.7;">
                Selected based on highest Cross-Validation ROC-AUC score,
                validated on held-out test data. The model uses wavelet-transformed
                features to distinguish genuine from forged banknotes.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_m2:
        if best_params:
            params_html = "".join(
                f'<div class="model-stat"><span class="model-stat-label">{k}</span><span class="model-stat-value">{v}</span></div>'
                for k, v in best_params.items()
            )
            st.markdown(
                f"""
            <div class="info-card">
                <h4 style="color:#667eea;margin-top:0;">Optimized Hyperparameters</h4>
                {params_html}
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Default hyperparameters used.")


# ═══════════════════════════════════════════════════════════════════════
# PAGE: Data Explorer
# ═══════════════════════════════════════════════════════════════════════

elif page == "🔍 Data Explorer":
    st.markdown(
        """
    <div class="main-header">
        <h1>🔍 Data Explorer</h1>
        <p>Interactive data browsing, filtering, and statistical analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ─── Data Preview ───
    st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        class_filter = st.multiselect("Filter by Class", options=[0, 1], default=[0, 1], format_func=lambda x: CLASS_LABELS[x])
    with col_f2:
        n_rows = st.slider("Number of rows to display", 10, len(df), 50)

    filtered_df = df[df[TARGET_COLUMN].isin(class_filter)]
    st.dataframe(filtered_df.head(n_rows), use_container_width=True, height=400)
    st.markdown(f"*Showing {min(n_rows, len(filtered_df))} of {len(filtered_df)} filtered records*")

    # ─── Summary Statistics ───
    st.markdown('<div class="section-header">📊 Summary Statistics</div>', unsafe_allow_html=True)

    tab_all, tab_genuine, tab_forged = st.tabs(["All Data", "Genuine Only", "Forged Only"])
    with tab_all:
        st.dataframe(df[FEATURE_COLUMNS].describe().round(4), use_container_width=True)
    with tab_genuine:
        st.dataframe(df[df[TARGET_COLUMN] == 0][FEATURE_COLUMNS].describe().round(4), use_container_width=True)
    with tab_forged:
        st.dataframe(df[df[TARGET_COLUMN] == 1][FEATURE_COLUMNS].describe().round(4), use_container_width=True)

    # ─── Interactive Scatter ───
    st.markdown('<div class="section-header">🎯 Interactive Feature Explorer</div>', unsafe_allow_html=True)

    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("X-Axis Feature", FEATURE_COLUMNS, index=0)
    with col_y:
        y_axis = st.selectbox("Y-Axis Feature", FEATURE_COLUMNS, index=1)

    df_scatter = df.copy()
    df_scatter["Class"] = df_scatter[TARGET_COLUMN].map(CLASS_LABELS)
    fig_scatter = px.scatter(
        df_scatter, x=x_axis, y=y_axis, color="Class",
        color_discrete_map={"Genuine": COLORS["genuine"], "Forged": COLORS["forged"]},
        opacity=0.6, hover_data=FEATURE_COLUMNS,
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=500, margin=dict(l=40, r=20, t=30, b=40), legend_title_text="Class")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ─── Box Plots ───
    st.markdown('<div class="section-header">📦 Feature Box Plots</div>', unsafe_allow_html=True)

    df_melted = df.melt(id_vars=[TARGET_COLUMN], value_vars=FEATURE_COLUMNS, var_name="Feature", value_name="Value")
    df_melted["Class"] = df_melted[TARGET_COLUMN].map(CLASS_LABELS)
    fig_box = px.box(df_melted, x="Feature", y="Value", color="Class",
        color_discrete_map={"Genuine": COLORS["genuine"], "Forged": COLORS["forged"]},
    )
    fig_box.update_layout(**PLOTLY_LAYOUT, height=450, margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # ─── Outlier Analysis ───
    st.markdown('<div class="section-header">🔍 Outlier Analysis</div>', unsafe_allow_html=True)

    outliers = detect_outliers_iqr(df, FEATURE_COLUMNS)
    outlier_data = []
    for col, info in outliers.items():
        outlier_data.append({
            "Feature": col, "Outlier Count": info["count"],
            "Percentage": f"{info['percentage']:.1f}%",
            "Lower Bound": f"{info['lower_bound']:.4f}",
            "Upper Bound": f"{info['upper_bound']:.4f}",
        })
    st.dataframe(pd.DataFrame(outlier_data), use_container_width=True, hide_index=True)

    # ─── Download ───
    st.markdown('<div class="section-header">📥 Download Data</div>', unsafe_allow_html=True)

    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Filtered Data (CSV)", data=csv_buffer.getvalue(), file_name="banknote_data_filtered.csv", mime="text/csv")
