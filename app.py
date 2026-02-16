"""
GPU Resource Estimator — Streamlit Web Interface
Imports all logic from gpu_estimator.py (no duplication).

Usage:
    pip install streamlit plotly
    streamlit run app.py
"""

import json
import math
import streamlit as st
import plotly.graph_objects as go

from gpu_estimator import (
    GPU_DATABASE,
    TASK_TYPES,
    MODEL_PRESETS,
    DEEPSPEED_STAGES,
    FRAMEWORKS,
    estimate_resources,
    estimate_training_time,
    generate_pytorch_code,
    generate_tf_code,
    format_bytes,
    format_time,
    format_flops,
    format_cost,
    _generate_recommendations,
)

# ─── Theme Configuration ───────────────────────────────────────
def init_theme():
    """Initialize theme state and return current theme mode."""
    if 'theme_mode' not in st.session_state:
        st.session_state['theme_mode'] = 'dark'
    return st.session_state['theme_mode']

def toggle_theme():
    """Toggle between dark and light theme."""
    st.session_state['theme_mode'] = 'light' if st.session_state['theme_mode'] == 'dark' else 'dark'

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="GPU Resource Estimator",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize theme
theme_mode = init_theme()

# ─── Custom CSS (Dark Theme) ────────────────────────────────────
DARK_CSS = """
<style>
/* ── Global ────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header banner ─────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2d1b69 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(139, 92, 246, 0.3);
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.15), transparent 60%);
}
.header-banner h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
    position: relative;
}
.header-banner p {
    margin: 0.3rem 0 0 0;
    font-size: 0.95rem;
    color: #94a3b8;
    position: relative;
}
.header-badge {
    display: inline-block;
    background: rgba(139, 92, 246, 0.2);
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #c4b5fd;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-left: 0.7rem;
    vertical-align: middle;
}

/* ── Metric cards ──────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #1e1e2e, #252540);
    border: 1px solid #333355;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.15s, border-color 0.15s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #8b5cf6;
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #8b8ba7;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.35rem;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1.2;
}
.metric-card .sub {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 0.25rem;
}
.metric-card.green  { border-color: #22c55e; }
.metric-card.red    { border-color: #ef4444; }
.metric-card.blue   { border-color: #3b82f6; }
.metric-card.purple { border-color: #8b5cf6; }
.metric-card.amber  { border-color: #f59e0b; }
.metric-card.green  .value { color: #4ade80; }
.metric-card.red    .value { color: #f87171; }
.metric-card.blue   .value { color: #60a5fa; }
.metric-card.purple .value { color: #a78bfa; }
.metric-card.amber  .value { color: #fbbf24; }

/* ── Status banner ─────────────────────────── */
.status-pass {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #4ade80;
    font-weight: 600;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.status-fail {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #f87171;
    font-weight: 600;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── VRAM gauge ────────────────────────────── */
.vram-bar-outer {
    background: #1e1e2e;
    border: 1px solid #333355;
    border-radius: 8px;
    padding: 0.15rem;
    margin: 0.8rem 0;
}
.vram-bar-inner {
    height: 24px;
    border-radius: 6px;
    transition: width 0.4s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
    min-width: 45px;
}
.vram-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ── Section titles ────────────────────────── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #333355;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Recommendation cards ──────────────────── */
.rec-card {
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    line-height: 1.5;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}
.rec-card.error {
    background: rgba(239,68,68,0.08);
    border-left: 3px solid #ef4444;
    color: #fca5a5;
}
.rec-card.warning {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid #f59e0b;
    color: #fcd34d;
}
.rec-card.info {
    background: rgba(59,130,246,0.08);
    border-left: 3px solid #3b82f6;
    color: #93c5fd;
}
.rec-card.success {
    background: rgba(34,197,94,0.08);
    border-left: 3px solid #22c55e;
    color: #86efac;
}

/* ── Formula cards ─────────────────────────── */
.formula-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
}
.formula-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8b8ba7;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.formula-result {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* ── Sidebar ───────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #12121e;
    border-right: 1px solid #1e1e35;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSlider label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #94a3b8;
}
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b5cf6;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #252545;
}
.gpu-spec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4rem;
    margin-top: 0.4rem;
}
.gpu-spec-item {
    background: #1a1a2e;
    border: 1px solid #252545;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    text-align: center;
}
.gpu-spec-item .spec-label {
    font-size: 0.62rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.gpu-spec-item .spec-value {
    font-size: 0.88rem;
    font-weight: 700;
    color: #c4b5fd;
}

/* ── Tabs ──────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #16162a;
    border-radius: 10px;
    padding: 0.25rem;
    border: 1px solid #252545;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    color: #8b8ba7;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #8b5cf6;
    color: white;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Tables ────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Styled HTML Table (dark) ─────────────── */
.styled-table-wrap {
    overflow-x: auto;
    border-radius: 10px;
    border: 1px solid #252545;
    margin-bottom: 1rem;
}
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.styled-table thead tr {
    background: #16162a;
}
.styled-table th {
    color: #8b5cf6 !important;
    background: #16162a !important;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 2px solid #252545;
}
.styled-table td {
    color: #e2e8f0 !important;
    background: #1e1e2e !important;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #252545;
}
.styled-table tbody tr:hover td {
    background: #252540 !important;
}

/* ── Theme Toggle Button ───────────────────── */
.theme-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 1rem;
    background: linear-gradient(145deg, #1e1e2e, #252540);
    border: 1px solid #333355;
    border-radius: 8px;
    margin-bottom: 1rem;
    cursor: pointer;
}
.theme-toggle-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #e2e8f0;
}
.theme-toggle-icon {
    font-size: 1.2rem;
}

/* ── Hide default streamlit elements ───────── */
div[data-testid="stMetric"] { display: none; }
.stDeployButton { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Main content background for dark theme ─── */
[data-testid="stApp"] {
    background: #0f0f17;
}
</style>
"""

# ─── Light Theme CSS ───────────────────────────────────────────
LIGHT_CSS = """
<style>
/* ── Global ────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header banner ─────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 50%, #c7d2fe 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99, 102, 241, 0.3);
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 20%, rgba(99, 102, 241, 0.15), transparent 60%);
}
.header-banner h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 700;
    color: #1e1b4b;
    position: relative;
}
.header-banner p {
    margin: 0.3rem 0 0 0;
    font-size: 0.95rem;
    color: #4b5563;
    position: relative;
}
.header-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #4338ca;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-left: 0.7rem;
    vertical-align: middle;
}

/* ── Metric cards ──────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.15s, border-color 0.15s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #6366f1;
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.35rem;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.2;
}
.metric-card .sub {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}
.metric-card.green  { border-color: #22c55e; }
.metric-card.red    { border-color: #ef4444; }
.metric-card.blue   { border-color: #3b82f6; }
.metric-card.purple { border-color: #8b5cf6; }
.metric-card.amber  { border-color: #f59e0b; }
.metric-card.green  .value { color: #16a34a; }
.metric-card.red    .value { color: #dc2626; }
.metric-card.blue   .value { color: #2563eb; }
.metric-card.purple .value { color: #7c3aed; }
.metric-card.amber  .value { color: #d97706; }

/* ── Status banner ─────────────────────────── */
.status-pass {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #16a34a;
    font-weight: 600;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.status-fail {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #dc2626;
    font-weight: 600;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── VRAM gauge ────────────────────────────── */
.vram-bar-outer {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.15rem;
    margin: 0.8rem 0;
}
.vram-bar-inner {
    height: 24px;
    border-radius: 6px;
    transition: width 0.4s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
    min-width: 45px;
}
.vram-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ── Section titles ────────────────────────── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Recommendation cards ──────────────────── */
.rec-card {
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    line-height: 1.5;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}
.rec-card.error {
    background: rgba(239,68,68,0.08);
    border-left: 3px solid #ef4444;
    color: #dc2626;
}
.rec-card.warning {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid #f59e0b;
    color: #d97706;
}
.rec-card.info {
    background: rgba(59,130,246,0.08);
    border-left: 3px solid #3b82f6;
    color: #2563eb;
}
.rec-card.success {
    background: rgba(34,197,94,0.08);
    border-left: 3px solid #22c55e;
    color: #16a34a;
}

/* ── Formula cards ─────────────────────────── */
.formula-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
}
.formula-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.formula-result {
    font-size: 0.82rem;
    color: #475569;
    margin-top: 0.3rem;
}

/* ── Sidebar ───────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSlider label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #475569;
}
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6366f1;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #e2e8f0;
}
.gpu-spec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4rem;
    margin-top: 0.4rem;
}
.gpu-spec-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    text-align: center;
}
.gpu-spec-item .spec-label {
    font-size: 0.62rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.gpu-spec-item .spec-value {
    font-size: 0.88rem;
    font-weight: 700;
    color: #6366f1;
}

/* ── Tabs ──────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #f1f5f9;
    border-radius: 10px;
    padding: 0.25rem;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    color: #64748b;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #6366f1;
    color: white;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Tables ────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Styled HTML Table (light) ────────────── */
.styled-table-wrap {
    overflow-x: auto;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.styled-table thead tr {
    background: #f8fafc;
}
.styled-table th {
    color: #6366f1 !important;
    background: #f8fafc !important;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 2px solid #e2e8f0;
}
.styled-table td {
    color: #1e293b !important;
    background: #ffffff !important;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #f1f5f9;
}
.styled-table tbody tr:hover td {
    background: #f8fafc !important;
}

/* ── Theme Toggle Button ───────────────────── */
.theme-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 1rem;
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-bottom: 1rem;
    cursor: pointer;
}
.theme-toggle-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #1e293b;
}
.theme-toggle-icon {
    font-size: 1.2rem;
}

/* ── Hide default streamlit elements ───────── */
div[data-testid="stMetric"] { display: none; }
.stDeployButton { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Main content background for light theme ─── */
[data-testid="stApp"] {
    background: #fafbfc !important;
    color: #1e293b !important;
}
[data-testid="stApp"] > div,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: #fafbfc !important;
}
[data-testid="stHeader"] {
    background: #fafbfc !important;
}

/* ── Global text color ────────────────────────── */
[data-testid="stApp"] p,
[data-testid="stApp"] span,
[data-testid="stApp"] div,
[data-testid="stApp"] label {
    color: #1e293b;
}

/* ── Headers ───────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    color: #1e293b !important;
}

/* ── Markdown text ─────────────────────────────── */
div[data-testid="stMarkdown"] p,
div[data-testid="stMarkdown"] span,
div[data-testid="stMarkdown"] li {
    color: #1e293b;
}

/* ── All input fields (text, number, textarea) ── */
input, textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-color: #d1d5db !important;
    caret-color: #1e293b !important;
}

/* ── Number inputs ─────────────────────────────── */
div[data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-color: #d1d5db !important;
}

/* ── Number input stepper buttons (+ / -) ──────── */
div[data-testid="stNumberInput"] button {
    background-color: #f1f5f9 !important;
    color: #475569 !important;
    border-color: #d1d5db !important;
}
div[data-testid="stNumberInput"] button:hover {
    background-color: #e2e8f0 !important;
    color: #1e293b !important;
}
div[data-testid="stNumberInput"] button svg {
    fill: #475569 !important;
    stroke: #475569 !important;
}

/* ── Selectbox / Dropdown ──────────────────────── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-color: #d1d5db !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #1e293b !important;
}
div[data-testid="stSelectbox"] svg {
    fill: #475569 !important;
}

/* ── Dropdown menu (popover) ───────────────────── */
[data-baseweb="popover"] {
    background-color: #ffffff !important;
    border-color: #d1d5db !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}
[data-baseweb="popover"] *:not(svg):not(path) {
    color: #1e293b !important;
}
[data-baseweb="popover"] li,
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
[data-baseweb="popover"] > div {
    background-color: #ffffff !important;
}
[data-baseweb="menu"] li,
[data-baseweb="popover"] li {
    background-color: #ffffff !important;
    color: #1e293b !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] li:hover {
    background-color: #f1f5f9 !important;
}
[data-baseweb="menu"] li[aria-selected="true"],
[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #e0e7ff !important;
}

/* ── All form labels ───────────────────────────── */
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stTextArea"] label {
    color: #475569 !important;
}

/* ── Slider ────────────────────────────────────── */
div[data-testid="stSlider"] [data-baseweb="slider"] div {
    color: #475569 !important;
}
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #1e293b !important;
}
div[data-testid="stSlider"] [role="slider"] {
    background-color: #6366f1 !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
    background: #e2e8f0 !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:first-child {
    background: #6366f1 !important;
}

/* ── Checkbox ──────────────────────────────────── */
div[data-testid="stCheckbox"] label span {
    color: #1e293b !important;
}
div[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] {
    color: #475569 !important;
}

/* ── Radio buttons ─────────────────────────────── */
div[data-testid="stRadio"] label {
    color: #475569 !important;
}
div[data-testid="stRadio"] label span {
    color: #1e293b !important;
}

/* ── Buttons ───────────────────────────────────── */
.stButton button,
button[kind="secondary"] {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
    border-color: #d1d5db !important;
}
.stButton button:hover {
    background-color: #e2e8f0 !important;
    border-color: #94a3b8 !important;
}

/* ── Download button ───────────────────────────── */
div[data-testid="stDownloadButton"] button {
    background-color: #6366f1 !important;
    color: white !important;
    border-color: #6366f1 !important;
}
div[data-testid="stDownloadButton"] button:hover {
    background-color: #4f46e5 !important;
}

/* ── Expander ──────────────────────────────────── */
div[data-testid="stExpander"] {
    background: #ffffff !important;
    border-color: #e2e8f0 !important;
}
div[data-testid="stExpander"] summary {
    color: #1e293b !important;
}
div[data-testid="stExpander"] summary span {
    color: #1e293b !important;
}

/* ── DataFrames / Tables ───────────────────────── */
div[data-testid="stDataFrame"] {
    background: #ffffff !important;
}
div[data-testid="stDataFrame"] [data-testid="glideDataEditor"],
div[data-testid="stDataFrame"] canvas {
    background: #ffffff !important;
}

table {
    color: #1e293b !important;
    background: #ffffff !important;
}
table th {
    color: #475569 !important;
    background: #f8fafc !important;
    border-color: #e2e8f0 !important;
}
table td {
    color: #1e293b !important;
    background: #ffffff !important;
    border-color: #e2e8f0 !important;
}

/* ── Code block — light bg + dark text ────────── */
div[data-testid="stCode"] pre,
div[data-testid="stCode"] code,
div[data-testid="stCode"],
.stCodeBlock pre,
.stCode pre {
    background-color: #f8f9fc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px;
    overflow: hidden;
}
div[data-testid="stCode"] pre,
div[data-testid="stCode"] pre code {
    color: #1e293b;
}

/* ── Caption / small text ──────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #64748b !important;
}
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] code {
    color: #64748b !important;
}

/* ── Sidebar overrides for light theme ─────────── */
section[data-testid="stSidebar"] {
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] > div {
    background: #f8fafc !important;
}
section[data-testid="stSidebar"] input {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-color: #d1d5db !important;
    -webkit-text-fill-color: #1e293b !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-color: #d1d5db !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #1e293b !important;
}
section[data-testid="stSidebar"] button {
    background-color: #f1f5f9 !important;
    color: #475569 !important;
    border-color: #d1d5db !important;
}
section[data-testid="stSidebar"] button svg {
    fill: #475569 !important;
    stroke: #475569 !important;
}
section[data-testid="stSidebar"] label {
    color: #475569 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #475569;
}

/* ── Vertical block / generic containers ───────── */
div[data-testid="stVerticalBlock"] {
    color: #1e293b;
}

/* ── BaseWeb input overrides ───────────────────── */
[data-baseweb="input"] {
    background-color: #ffffff !important;
    border-color: #d1d5db !important;
}
[data-baseweb="input"] input {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
}
[data-baseweb="base-input"] {
    background-color: #ffffff !important;
    border-color: #d1d5db !important;
}

/* ── Tooltip / popover background fix ──────────── */
[data-baseweb="tooltip"] {
    background-color: #1e293b !important;
    color: #ffffff !important;
}
</style>
"""

# Apply CSS based on current theme
if theme_mode == 'dark':
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HELPER — render metric card
# ═══════════════════════════════════════════════════════════════
def metric_card(label: str, value: str, sub: str = "", color: str = ""):
    cls = f"metric-card {color}" if color else "metric-card"
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>'


# ═══════════════════════════════════════════════════════════════
# HELPER — render themed HTML table (replaces st.dataframe)
# ═══════════════════════════════════════════════════════════════
def render_table(rows: list[dict], key_filter: str = "_"):
    """Render a list of dicts as a styled HTML table.
    Skips keys starting with key_filter (default '_')."""
    if not rows:
        return
    headers = [k for k in rows[0] if not k.startswith(key_filter)]
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    body_html = ""
    for row in rows:
        cells = "".join(f"<td>{row[h]}</td>" for h in headers)
        body_html += f"<tr>{cells}</tr>"
    st.markdown(
        f'<div class="styled-table-wrap"><table class="styled-table">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{body_html}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════════
def get_plotly_theme():
    """Get Plotly layout and colors based on current theme."""
    if theme_mode == 'dark':
        return {
            'layout': dict(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#94a3b8"),
                margin=dict(l=0, r=0, t=35, b=30),
            ),
            'grid': dict(gridcolor="#1e1e35", zerolinecolor="#1e1e35"),
            'colors': ["#8b5cf6", "#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"],
            'bar_text': "white",
            'donut_hole': "#16162a",
            'donut_text': "#e2e8f0",
        }
    else:
        return {
            'layout': dict(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#475569"),
                margin=dict(l=0, r=0, t=35, b=30),
            ),
            'grid': dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
            'colors': ["#6366f1", "#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"],
            'bar_text': "#1e293b",
            'donut_hole': "#ffffff",
            'donut_text': "#1e293b",
        }

plotly_theme = get_plotly_theme()
PLOTLY_LAYOUT = plotly_theme['layout']
GRID_STYLE = plotly_theme['grid']
COLORS = plotly_theme['colors']


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Theme Toggle ───────────────────────────────────────────
    col_theme1, col_theme2 = st.columns([3, 1])
    with col_theme1:
        st.markdown('<div class="sidebar-section">Appearance</div>', unsafe_allow_html=True)
    with col_theme2:
        if st.button("🌙" if theme_mode == "light" else "☀️", key="theme_toggle"):
            toggle_theme()
            st.rerun()
    
    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    model_names = list(MODEL_PRESETS.keys())
    model_preset = st.selectbox("Model preset", model_names, index=model_names.index("LLaMA 3 8B"), label_visibility="collapsed")
    preset = MODEL_PRESETS[model_preset]

    if model_preset == "Custom":
        model_params = st.number_input("Parameters (B)", min_value=0.01, max_value=2000.0, value=7.0, step=0.1)
        num_layers = st.number_input("Layers", min_value=1, max_value=200, value=32)
        hidden_size = st.number_input("Hidden size", min_value=64, max_value=32768, value=4096, step=64)
    else:
        model_params = preset["params"]
        num_layers = preset["layers"]
        hidden_size = preset["hidden"]
        st.caption(f"`{model_params}B` params  ·  `{num_layers}` layers  ·  hidden `{hidden_size}`")

    st.markdown('<div class="sidebar-section">Task</div>', unsafe_allow_html=True)
    task_names = list(TASK_TYPES.keys())
    task_type = st.selectbox("Task type", task_names, index=0, label_visibility="collapsed")
    task_info = TASK_TYPES[task_type]
    is_training = task_info["is_training"]
    badge = "TRAIN" if is_training else "INFERENCE"
    badge_color = "#22c55e" if is_training else "#3b82f6"
    st.markdown(
        f'<span style="display:inline-block;background:rgba({",".join(str(int(badge_color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.15);'
        f'border:1px solid {badge_color};color:{badge_color};padding:0.1rem 0.5rem;border-radius:12px;font-size:0.68rem;font-weight:700;letter-spacing:0.05em">'
        f'{badge}</span> <span style="color:#64748b;font-size:0.8rem;margin-left:0.4rem">{task_info["description"]}</span>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">GPU</div>', unsafe_allow_html=True)
    gpu_names = list(GPU_DATABASE.keys())
    gpu_name = st.selectbox("GPU model", gpu_names, index=gpu_names.index("NVIDIA A100 (80GB)"), label_visibility="collapsed")
    gpu_spec = GPU_DATABASE[gpu_name].copy()

    # Custom GPU inputs
    if gpu_name == "Custom GPU":
        with st.expander("⚙️ Custom GPU Configuration", expanded=True):
            gpu_spec['vram'] = st.number_input("VRAM (GB)", min_value=1, max_value=1024, value=int(gpu_spec['vram']), step=1)
            gpu_spec['tflops_fp16'] = st.number_input("FP16 TFLOPs", min_value=1, max_value=10000, value=int(gpu_spec['tflops_fp16']), step=1)
            gpu_spec['bandwidth'] = st.number_input("Memory Bandwidth (GB/s)", min_value=100, max_value=20000, value=int(gpu_spec['bandwidth']), step=50)
            
            st.caption("Optional fields (leave at 0 if unknown):")
            col1, col2 = st.columns(2)
            with col1:
                gpu_spec['tflops_fp32'] = st.number_input("FP32 TFLOPs", min_value=0, max_value=5000, value=0, step=1)
                gpu_spec['tflops_tf32'] = st.number_input("TF32 TFLOPs", min_value=0, max_value=10000, value=0, step=1)
            with col2:
                gpu_spec['price_hr'] = st.number_input("Price/hour ($)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
                gpu_spec['tdp'] = st.number_input("TDP (Watts)", min_value=0, max_value=2000, value=0, step=10)
            
            gpu_spec['nvlink'] = st.checkbox("NVLink Support", value=gpu_spec['nvlink'])

    st.markdown(f"""
    <div class="gpu-spec-grid">
        <div class="gpu-spec-item"><div class="spec-label">VRAM</div><div class="spec-value">{gpu_spec['vram']} GB</div></div>
        <div class="gpu-spec-item"><div class="spec-label">FP16</div><div class="spec-value">{gpu_spec['tflops_fp16']} TF</div></div>
        <div class="gpu-spec-item"><div class="spec-label">Bandwidth</div><div class="spec-value">{gpu_spec['bandwidth']} GB/s</div></div>
        <div class="gpu-spec-item"><div class="spec-label">NVLink</div><div class="spec-value">{"Yes" if gpu_spec['nvlink'] else "No"}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Parameters</div>', unsafe_allow_html=True)
    batch_size = st.number_input("Batch size", min_value=1, max_value=512, value=4)
    seq_length = st.number_input("Sequence length", min_value=128, max_value=131072, value=2048, step=128)
    precision = st.selectbox("Precision", ["FP32", "FP16/BF16", "INT8", "INT4"], index=1)
    num_gpus = st.slider("Number of GPUs", min_value=1, max_value=128, value=1)

    grad_checkpoint = False
    deepspeed_stage = "None"
    dataset_tokens = 10.0
    epochs = 1
    framework = "PyTorch + DeepSpeed"

    if is_training:
        st.markdown('<div class="sidebar-section">Training</div>', unsafe_allow_html=True)
        grad_checkpoint = st.checkbox("Gradient checkpointing")
        deepspeed_stage = st.selectbox("DeepSpeed ZeRO", list(DEEPSPEED_STAGES.keys()), index=0)
        dataset_tokens = st.number_input("Dataset tokens (B)", min_value=0.001, max_value=15000.0, value=10.0, step=1.0)
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=1)
        framework = st.selectbox("Framework", list(FRAMEWORKS.keys()), index=0)

# ═══════════════════════════════════════════════════════════════
# RUN ESTIMATION
# ═══════════════════════════════════════════════════════════════
config = {
    "model_preset": model_preset,
    "model_params": model_params,
    "num_layers": num_layers,
    "hidden_size": hidden_size,
    "task_type": task_type,
    "gpu": gpu_name,
    "num_gpus": num_gpus,
    "batch_size": batch_size,
    "seq_length": seq_length,
    "precision": precision,
    "grad_checkpoint": grad_checkpoint,
    "deepspeed_stage": deepspeed_stage,
    "dataset_tokens": dataset_tokens,
    "epochs": epochs,
    "framework": framework,
}

estimation = estimate_resources(
    model_params=model_params,
    task_type=task_type,
    batch_size=batch_size,
    seq_length=seq_length,
    precision=precision,
    grad_checkpoint=grad_checkpoint,
    deepspeed_stage=deepspeed_stage,
    num_gpus=num_gpus,
    dataset_tokens=dataset_tokens * 1e9,
    epochs=epochs,
    num_layers=num_layers,
    hidden_size=hidden_size,
)

time_est = None
if is_training:
    time_est = estimate_training_time(
        estimation, gpu_spec, num_gpus, precision, deepspeed_stage, framework,
    )

total_vram_available = gpu_spec["vram"] * num_gpus
fits = estimation.total_vram <= total_vram_available
gpus_needed = math.ceil(estimation.total_vram / gpu_spec["vram"])

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
model_label = f"{model_preset} ({model_params}B)" if model_preset != "Custom" else f"Custom ({model_params}B)"
st.markdown(f"""
<div class="header-banner">
    <h1>GPU Resource Estimator <span class="header-badge">Data Engineering & Semantics</span></h1>
    <p>{model_label} &nbsp;&middot;&nbsp; {task_type} &nbsp;&middot;&nbsp; {gpu_name} &times;{num_gpus}</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
if is_training:
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time & FLOPs", "Multi-GPU Scaling", "Code & Export"])
else:
    tab1, tab4 = st.tabs(["Overview", "Code & Export"])

# ───────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ───────────────────────────────────────────────────────────────
with tab1:
    # ── Key metrics ──
    if is_training and time_est:
        cost_hr = gpu_spec["price_hr"] * max(num_gpus, gpus_needed)
        total_cost = cost_hr * (time_est.training_time_seconds / 3600)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("VRAM Required", format_bytes(estimation.total_vram), f"of {format_bytes(total_vram_available)} available", "purple"), unsafe_allow_html=True)
        c2.markdown(metric_card("Training Time", format_time(time_est.training_time_seconds), f"{estimation.total_steps:,} steps", "blue"), unsafe_allow_html=True)
        c3.markdown(metric_card("Total Cost", format_cost(total_cost), f"{format_cost(cost_hr)}/hr", "amber"), unsafe_allow_html=True)
        c4.markdown(metric_card("MFU", f"{time_est.effective_mfu * 100:.1f}%", f"{time_est.effective_tflops_total:.0f} effective TFLOPS", "green"), unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("VRAM Required", format_bytes(estimation.total_vram), f"of {format_bytes(total_vram_available)} available", "purple"), unsafe_allow_html=True)
        c2.markdown(metric_card("Mode", "Inference", task_type, "blue"), unsafe_allow_html=True)
        c3.markdown(metric_card("Cost / hr", f"${gpu_spec['price_hr'] * num_gpus:.2f}", f"{num_gpus} GPU(s)", "amber"), unsafe_allow_html=True)
        c4.markdown(metric_card("GPUs Needed", str(gpus_needed), f"{gpu_spec['vram']} GB each", "green" if fits else "red"), unsafe_allow_html=True)

    # ── VRAM gauge ──
    usage_pct = min(estimation.total_vram / total_vram_available, 1.0) if total_vram_available > 0 else 0
    bar_color = "#22c55e" if usage_pct < 0.7 else ("#f59e0b" if usage_pct < 0.9 else "#ef4444")
    st.markdown(f"""
    <div class="vram-bar-outer">
        <div class="vram-bar-inner" style="width:{max(usage_pct*100,3):.1f}%;background:linear-gradient(90deg,{bar_color}dd,{bar_color}88)">
            {usage_pct*100:.0f}%
        </div>
    </div>
    <div class="vram-bar-label">
        <span>0 GB</span>
        <span>{format_bytes(estimation.total_vram)} / {format_bytes(total_vram_available)}</span>
        <span>{format_bytes(total_vram_available)}</span>
    </div>
    """, unsafe_allow_html=True)

    if fits:
        st.markdown(f'<div class="status-pass">PASS &mdash; Feasible with {num_gpus} GPU(s)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-fail">FAIL &mdash; Insufficient VRAM. Minimum {gpus_needed} GPU(s) needed.</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Memory breakdown & donut — side by side ──
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="section-title">Memory Breakdown</div>', unsafe_allow_html=True)

        if is_training:
            labels = ["Weights", "Gradients", "Optimizer", "Activations"]
            values = [estimation.weights_gb, estimation.gradients_gb, estimation.optimizer_gb, estimation.activations_gb]
        else:
            labels = ["Weights", "KV Cache"]
            values = [estimation.weights_gb, estimation.kv_cache_gb]

        colors = COLORS[:len(labels)]

        fig_bar = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[format_bytes(v) for v in values],
            textposition="auto",
            textfont=dict(color=plotly_theme['bar_text'], size=12, family="Inter"),
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            height=220,
            xaxis=dict(**GRID_STYLE, title=""),
            yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with right_col:
        st.markdown('<div class="section-title">Distribution</div>', unsafe_allow_html=True)

        fig_donut = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.65,
            marker=dict(colors=colors, line=dict(color=plotly_theme['donut_hole'], width=2)),
            textinfo="percent",
            textfont=dict(size=11, color=plotly_theme['bar_text']),
            hovertemplate="<b>%{label}</b><br>%{value:.2f} GB<br>%{percent}<extra></extra>",
        ))
        fig_donut.update_layout(
            **PLOTLY_LAYOUT,
            height=220,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{format_bytes(estimation.total_vram)}</b>",
                x=0.5, y=0.5, font_size=16, font_color=plotly_theme['donut_text'],
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Recommendations ──
    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
    recs = _generate_recommendations(estimation, time_est, config, gpu_spec, total_vram_available, fits, gpus_needed)
    for r in recs:
        clean = r.replace("[red]", "").replace("[yellow]", "").replace("[blue]", "").replace("[green]", "").replace("[/]", "")
        if "[red]" in r:
            st.markdown(f'<div class="rec-card error">{clean}</div>', unsafe_allow_html=True)
        elif "[yellow]" in r:
            st.markdown(f'<div class="rec-card warning">{clean}</div>', unsafe_allow_html=True)
        elif "[green]" in r:
            st.markdown(f'<div class="rec-card success">{clean}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="rec-card info">{clean}</div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# TAB 2 — TIME & FLOPs (training only)
# ───────────────────────────────────────────────────────────────
if is_training and time_est:
    with tab2:
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.markdown(metric_card("Total FLOPs", format_flops(estimation.total_flops), "", "purple"), unsafe_allow_html=True)
        r1c2.markdown(metric_card("FLOPs / token", format_flops(estimation.flops_per_token), "", "blue"), unsafe_allow_html=True)
        r1c3.markdown(metric_card("Effective TFLOPS", f"{time_est.effective_tflops_total:.1f}", f"peak {time_est.peak_tflops}", "green"), unsafe_allow_html=True)
        st.markdown("")
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.markdown(metric_card("Throughput", f"{time_est.throughput_tokens_per_sec:,.0f} tok/s", "", "amber"), unsafe_allow_html=True)
        r2c2.markdown(metric_card("Time / step", f"{time_est.time_per_step * 1000:.1f} ms", f"{estimation.total_steps:,} total steps", "blue"), unsafe_allow_html=True)
        r2c3.markdown(metric_card("Scaling Efficiency", f"{time_est.scaling_efficiency * 100:.1f}%", f"{'NVLink' if gpu_spec['nvlink'] else 'PCIe'} interconnect", "purple"), unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-title">Scaling Formulas</div>', unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown('<div class="formula-card"><div class="formula-label">Total Compute (Kaplan / Chinchilla)</div>', unsafe_allow_html=True)
            st.latex(r"C = 6 \times N \times D")
            st.markdown(f'<div class="formula-result">= {format_flops(estimation.total_flops)}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="formula-card"><div class="formula-label">Model FLOPs Utilization</div>', unsafe_allow_html=True)
            st.latex(r"\text{MFU} = \frac{\text{Achieved FLOPS}}{\text{Peak FLOPS}}")
            st.markdown(f'<div class="formula-result">{time_est.effective_tflops_total:.1f} / {time_est.peak_tflops} = {time_est.effective_mfu * 100:.1f}%</div></div>', unsafe_allow_html=True)

        with fc2:
            st.markdown('<div class="formula-card"><div class="formula-label">Attention FLOPs Overhead</div>', unsafe_allow_html=True)
            st.latex(r"\text{Attention} = 12 \times L \times H \times S^2")
            st.markdown(
                f'<div class="formula-result">12 &times; {estimation.num_layers} &times; {estimation.hidden_size} &times; {seq_length}&sup2; '
                f'= {format_flops(estimation.attention_flops_per_token)} / token</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="formula-card"><div class="formula-label">Multi-GPU Scaling</div>', unsafe_allow_html=True)
            st.latex(r"\eta = \alpha^{\log_2(n)}")
            alpha = "0.95" if gpu_spec["nvlink"] else "0.88"
            link = "NVLink" if gpu_spec["nvlink"] else "PCIe"
            st.markdown(
                f'<div class="formula-result">{link} (&alpha;={alpha}) &rarr; {time_est.scaling_efficiency * 100:.1f}% with {num_gpus} GPU(s)</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="formula-card"><div class="formula-label">Training Time</div>', unsafe_allow_html=True)
        st.latex(r"\text{Time} = \frac{C}{\text{GPUs} \times \text{Peak} \times \text{MFU} \times \eta}")
        st.markdown(f'<div class="formula-result">= {format_time(time_est.training_time_seconds)}</div></div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# TAB 3 — MULTI-GPU SCALING (training only)
# ───────────────────────────────────────────────────────────────
if is_training:
    with tab3:
        st.markdown('<div class="section-title">Scaling Projection</div>', unsafe_allow_html=True)

        gpu_configs = [1, 2, 4, 8, 16, 32, 64]
        scaling_rows = []
        for n in gpu_configs:
            est_n = estimate_resources(
                model_params=model_params,
                task_type=task_type,
                batch_size=batch_size,
                seq_length=seq_length,
                precision=precision,
                grad_checkpoint=grad_checkpoint,
                deepspeed_stage=deepspeed_stage,
                num_gpus=n,
                dataset_tokens=dataset_tokens * 1e9,
                epochs=epochs,
                num_layers=num_layers,
                hidden_size=hidden_size,
            )
            t_n = estimate_training_time(est_n, gpu_spec, n, precision, deepspeed_stage, framework)
            cost_n = gpu_spec["price_hr"] * n * (t_n.training_time_seconds / 3600)
            scaling_rows.append({
                "GPUs": f"{n}x",
                "Time": format_time(t_n.training_time_seconds),
                "Cost": format_cost(cost_n),
                "Efficiency": f"{t_n.scaling_efficiency * 100:.0f}%",
                "TFLOPS": f"{t_n.effective_tflops_total:.0f}",
                "tok/s": f"{t_n.throughput_tokens_per_sec:,.0f}",
                "_hours": t_n.training_time_seconds / 3600,
                "_cost": cost_n,
                "_gpus": n,
            })

        render_table(scaling_rows)

        # ── Dual-axis chart ──
        fig_scale = go.Figure()
        fig_scale.add_trace(go.Scatter(
            x=[r["_gpus"] for r in scaling_rows],
            y=[r["_hours"] for r in scaling_rows],
            name="Time (hours)",
            mode="lines+markers",
            line=dict(color="#8b5cf6", width=2.5),
            marker=dict(size=8, symbol="circle"),
            fill="tozeroy",
            fillcolor="rgba(139,92,246,0.08)",
        ))
        fig_scale.add_trace(go.Scatter(
            x=[r["_gpus"] for r in scaling_rows],
            y=[r["_cost"] for r in scaling_rows],
            name="Cost ($)",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=8, symbol="diamond"),
            yaxis="y2",
        ))
        fig_scale.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            xaxis=dict(**GRID_STYLE, title="Number of GPUs"),
            yaxis=dict(**GRID_STYLE, title="Time (hours)", side="left"),
            yaxis2=dict(title="Cost ($)", side="right", overlaying="y", gridcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5, font=dict(size=12)),
        )
        st.plotly_chart(fig_scale, use_container_width=True)

        # ── GPU comparison ──
        st.markdown('<div class="section-title">GPU Comparison (sorted by cost)</div>', unsafe_allow_html=True)
        comp_rows = []
        for g_name, g_spec in GPU_DATABASE.items():
            # Skip Custom GPU in comparison table
            if g_name == "Custom GPU":
                continue
            needed = math.ceil(estimation.total_vram / g_spec["vram"])
            t_g = estimate_training_time(estimation, g_spec, needed, precision, deepspeed_stage, framework)
            cost_g = g_spec["price_hr"] * needed * (t_g.training_time_seconds / 3600)
            comp_rows.append({
                "GPU": g_name,
                "GPUs Needed": needed,
                "VRAM/GPU": f"{g_spec['vram']} GB",
                "Time": format_time(t_g.training_time_seconds),
                "Cost": format_cost(cost_g),
                "$/hr": f"${g_spec['price_hr']:.2f}",
                "_cost": cost_g,
            })
        comp_rows.sort(key=lambda r: r["_cost"])
        render_table(comp_rows)


# ───────────────────────────────────────────────────────────────
# TAB 4 — CODE & EXPORT
# ───────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Generated Profiling Code</div>', unsafe_allow_html=True)
    code_framework = st.radio("Framework", ["PyTorch", "TensorFlow"], horizontal=True, label_visibility="collapsed")

    if code_framework == "PyTorch":
        code = generate_pytorch_code(config)
    else:
        code = generate_tf_code(config)

    st.code(code, language="python")

    st.markdown("")
    st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)

    export_data = {
        "config": config,
        "estimation": {
            "vram_required_gb": round(estimation.total_vram, 2),
            "vram_available_gb": total_vram_available,
            "fits": fits,
            "gpus_needed": gpus_needed,
            "model_size_gb": round(estimation.model_size_gb, 2),
            "total_flops": estimation.total_flops,
            "total_steps": estimation.total_steps,
            "memory_breakdown": {
                "weights_gb": round(estimation.weights_gb, 2),
                "gradients_gb": round(estimation.gradients_gb, 2),
                "optimizer_gb": round(estimation.optimizer_gb, 2),
                "activations_gb": round(estimation.activations_gb, 2),
                "kv_cache_gb": round(estimation.kv_cache_gb, 2),
            },
        },
    }
    if time_est:
        cost_hr = gpu_spec["price_hr"] * max(num_gpus, gpus_needed)
        export_data["training_time"] = {
            "seconds": round(time_est.training_time_seconds, 1),
            "formatted": format_time(time_est.training_time_seconds),
            "throughput_tokens_per_sec": round(time_est.throughput_tokens_per_sec, 0),
            "mfu": round(time_est.effective_mfu, 4),
            "scaling_efficiency": round(time_est.scaling_efficiency, 4),
            "effective_tflops": round(time_est.effective_tflops_total, 1),
            "cost_total": round(cost_hr * (time_est.training_time_seconds / 3600), 2),
        }

    dl1, dl2, _ = st.columns([1, 1, 2])
    with dl1:
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name="gpu_estimation.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="Download Code",
            data=code,
            file_name=f"gpu_profiler.{'py'}",
            mime="text/x-python",
            use_container_width=True,
        )
