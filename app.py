import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import mysql.connector
import os
from datetime import datetime
import warnings
from config import MYSQL_CONFIG, DEFAULT_QUERY
try:
    from config import AI_CONFIG
except ImportError:
    AI_CONFIG = {
        'api_key': os.getenv('GEMINI_API_KEY', ''),
        'model': os.getenv('GEMINI_MODEL', 'gemini-3.6-flash'),
        'temperature': float(os.getenv('GEMINI_TEMPERATURE', '0.2')),
        'max_output_tokens': int(os.getenv('GEMINI_MAX_TOKENS', '2048'))
    }
import ai_engine
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Universal Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── v3 Instrument Panel CSS ──────────────────────────────────────────────────
# Google Fonts: Space Grotesk (headings) · Inter (body) · IBM Plex Mono (numbers)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* ── Design tokens ── */
    :root {
        --bg:             #0B0E14;
        --surface:        #12161F;
        --surface-raised: #171C27;
        --border:         #232A38;
        --accent-signal:  #5EEAD4;
        --accent-warn:    #FDBA74;
        --accent-secondary: #A5B4FC;
        --text-primary:   #F1F5F9;
        --text-muted:     #8B95A7;
        --font-display:   'Space Grotesk', system-ui, sans-serif;
        --font-body:      'Inter', system-ui, sans-serif;
        --font-mono:      'IBM Plex Mono', 'Consolas', monospace;
    }

    /* ── Base ── */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: var(--bg) !important;
        font-family: var(--font-body);
        color: var(--text-primary);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebarHeader"] {
        background-color: var(--surface) !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label {
        color: var(--text-muted);
        font-family: var(--font-body);
        font-size: 0.875rem;
    }

    /* ── Status bar (instrument readout line above title) ── */
    .status-bar {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-bar .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-signal);
        display: inline-block;
        animation: blink 2s ease-in-out infinite;
    }
    @media (prefers-reduced-motion: reduce) {
        .status-bar .status-dot { animation: none; }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.3; }
    }

    /* ── Main title ── */
    .main-header {
        font-family: var(--font-display);
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin: 0 0 0 0;
        line-height: 1.2;
    }

    /* ── Live readout strip ── */
    .readout-strip {
        width: 100%;
        height: 36px;
        background-color: var(--surface);
        border-top: 1px solid var(--border);
        border-bottom: 1px solid var(--border);
        margin: 14px 0 20px 0;
        display: flex;
        align-items: center;
        padding: 0 16px;
        gap: 32px;
        overflow: hidden;
        position: relative;
    }
    .readout-stat {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .readout-stat span {
        color: var(--accent-signal);
        margin-left: 4px;
    }
    /* SVG waveform in the strip */
    .readout-wave {
        flex: 1;
        display: flex;
        align-items: center;
        overflow: hidden;
        opacity: 0.5;
    }
    .readout-wave svg {
        width: 100%;
        height: 20px;
    }
    /* Scrolling animation for waveform */
    .wave-path {
        stroke: var(--accent-signal);
        stroke-width: 1.5;
        fill: none;
        stroke-dasharray: 800;
        stroke-dashoffset: 800;
        animation: drawWave 3s ease forwards;
    }
    @media (prefers-reduced-motion: reduce) {
        .wave-path { animation: none; stroke-dashoffset: 0; }
    }
    @keyframes drawWave {
        to { stroke-dashoffset: 0; }
    }

    /* ── Metric cards ── */
    .metric-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 18px 20px 16px 20px;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: var(--accent-signal);
    }
    .metric-card:focus-within {
        outline: 2px solid var(--accent-signal);
        outline-offset: 2px;
    }
    .metric-value {
        font-family: var(--font-mono);
        font-size: 2rem;
        font-weight: 600;
        color: var(--accent-signal);
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-family: var(--font-body);
        font-size: 0.68rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    /* ── Tabs — underline channel-selector style ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-radius: 0;
        padding: 0;
        border: none;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background: transparent !important;
        border-radius: 0;
        color: var(--text-muted);
        font-family: var(--font-body);
        font-weight: 500;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        padding: 0 20px;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s ease, border-color 0.15s ease;
        margin-bottom: -1px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: transparent !important;
        border-bottom-color: var(--border) !important;
    }
    .stTabs [data-baseweb="tab"]:focus-visible {
        outline: 2px solid var(--accent-signal);
        outline-offset: -2px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important;
        font-weight: 600;
        background: transparent !important;
        border-bottom: 2px solid var(--accent-signal) !important;
    }
    /* Tab panel area */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.5rem;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: transparent;
        color: var(--accent-signal);
        border: 1px solid var(--accent-signal);
        border-radius: 3px;
        padding: 8px 20px;
        font-family: var(--font-mono);
        font-weight: 500;
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        transition: background-color 0.15s ease, color 0.15s ease;
        box-shadow: none;
    }
    .stButton > button:hover {
        background-color: var(--accent-signal);
        color: var(--bg);
        box-shadow: none;
    }
    .stButton > button:focus-visible {
        outline: 2px solid var(--accent-signal);
        outline-offset: 2px;
    }
    .stButton > button:active {
        opacity: 0.85;
    }

    /* ── Column type tags ── */
    .auto-column {
        display: inline-block;
        background-color: transparent;
        color: var(--accent-signal);
        border: 1px solid var(--accent-signal);
        padding: 2px 10px;
        border-radius: 2px;
        font-family: var(--font-mono);
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        margin: 3px 4px 3px 0;
        opacity: 0.85;
    }

    /* ── File type badge ── */
    .file-type-badge {
        background-color: transparent;
        color: var(--accent-signal);
        border: 1px solid var(--border);
        padding: 2px 10px;
        border-radius: 2px;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-left: 10px;
        vertical-align: middle;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border-radius: 4px;
        border: 1px solid var(--border);
        overflow: hidden;
    }

    /* ── Inputs ── */
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stSelectbox > div > div {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-primary);
        font-family: var(--font-body);
    }
    .stTextInput > div > div:focus-within,
    .stTextArea > div > div:focus-within {
        border-color: var(--accent-signal);
        box-shadow: 0 0 0 2px rgba(94,234,212,0.15);
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 3px !important;
        color: var(--text-muted) !important;
        font-family: var(--font-body);
        font-weight: 500;
        font-size: 0.875rem;
    }
    .streamlit-expanderHeader:hover {
        background-color: var(--surface-raised) !important;
        color: var(--text-primary) !important;
    }
    .streamlit-expanderContent {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 3px 3px !important;
        padding: 16px;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: var(--accent-signal);
        border-color: var(--accent-signal);
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background-color: var(--accent-signal);
    }

    /* ── Alerts ── */
    div[data-baseweb="notification"] {
        border-radius: 3px;
    }

    /* ── Section label ── */
    .section-title {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-left: 2px solid var(--accent-signal);
        padding-left: 8px;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* ── Idle / empty state ── */
    .idle-screen {
        border: 1px solid var(--border);
        background-color: var(--surface);
        border-radius: 4px;
        padding: 40px 36px;
        margin-top: 12px;
    }
    .idle-status {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        color: var(--accent-warn);
        text-transform: uppercase;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .idle-status::before {
        content: '';
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-warn);
        opacity: 0.8;
    }
    .idle-title {
        font-family: var(--font-display);
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 10px 0;
    }
    .idle-body {
        font-family: var(--font-body);
        font-size: 0.875rem;
        color: var(--text-muted);
        line-height: 1.7;
        max-width: 520px;
        margin: 0 0 28px 0;
    }
    .idle-formats {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .idle-format-tag {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: var(--text-muted);
        border: 1px solid var(--border);
        padding: 3px 10px;
        border-radius: 2px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ── Idle waveform divider ── */
    .idle-wave {
        margin: 28px 0 0 0;
        opacity: 0.25;
    }
    .idle-wave svg { width: 100%; height: 28px; }

    /* ── AI Intelligence Channel Styles ── */
    .ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        padding: 4px 12px;
        border-radius: 3px;
        border: 1px solid var(--border);
        background: var(--surface);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .ai-badge-gemini {
        border-color: #5EEAD4 !important;
        color: #5EEAD4 !important;
        background: rgba(94, 234, 212, 0.08) !important;
    }
    .ai-badge-heuristic {
        border-color: #FDBA74 !important;
        color: #FDBA74 !important;
        background: rgba(253, 186, 116, 0.08) !important;
    }
    .ai-briefing-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    .ai-action-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease, transform 0.15s ease;
    }
    .ai-action-card:hover {
        border-color: var(--accent-signal);
    }
    .ai-action-card h4 {
        margin: 0 0 6px 0;
        font-family: var(--font-display);
        font-size: 0.95rem;
        color: var(--text-primary);
    }
    .ai-action-card p {
        margin: 0;
        font-size: 0.82rem;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* ── Markdown Content Typography & Cards ── */
    .stMarkdown h3 {
        font-family: var(--font-display) !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.6rem !important;
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
    }
    .stMarkdown h4 {
        font-family: var(--font-display) !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--accent-signal) !important;
        margin-top: 1rem !important;
        margin-bottom: 0.4rem !important;
    }
    .stMarkdown ul, .stMarkdown ol {
        margin-left: 1.2rem !important;
        line-height: 1.65;
    }
    .stMarkdown li {
        color: #CBD5E1 !important;
        font-size: 0.9rem !important;
        margin-bottom: 6px;
    }
    .stMarkdown strong {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    [data-testid="stChatMessage"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stChatMessage"] p {
        color: var(--text-primary) !important;
        font-family: var(--font-body);
        font-size: 0.92rem;
        line-height: 1.6;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        padding: 20px 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header: STATUS bar + title + live readout strip ──────────────────────────
_df_loaded = st.session_state.get('df') is not None
_rows      = f"{len(st.session_state.df):,}" if _df_loaded else "—"
_cols      = str(len(st.session_state.df.columns)) if _df_loaded else "—"
_ncols     = str(len(st.session_state.get('numeric_columns', []))) if _df_loaded else "—"
_status_txt = f"DATASET LOADED · {_rows} ROWS · {_cols} COLS · {_ncols} NUMERIC" if _df_loaded else "NO DATASET LOADED · AWAITING INPUT"
_dot_style  = "background:#5EEAD4;" if _df_loaded else "background:#FDBA74;"

st.markdown(f"""
<div class="status-bar">
    <span class="status-dot" style="{_dot_style}"></span>
    <span>{_status_txt}</span>
</div>
<p class="main-header">Universal Data Analysis Dashboard</p>
""", unsafe_allow_html=True)

# Live readout strip — waveform + key stats
_wave_pts = "0,10 20,5 35,18 50,3 65,16 80,8 95,20 110,4 125,14 140,7 160,18 175,5 190,12 210,3 225,16 240,8 260,19 275,6 290,13 310,4 325,17 340,9 360,20 375,5 390,11 410,2 425,15 440,8 460,18 475,4 490,13 510,6 525,17 540,9 560,19 575,3 590,14"
st.markdown(f"""
<div class="readout-strip">
    <div class="readout-stat">ROWS <span>{_rows}</span></div>
    <div class="readout-stat">COLS <span>{_cols}</span></div>
    <div class="readout-stat">NUMERIC <span>{_ncols}</span></div>
    <div class="readout-wave">
        <svg viewBox="0 0 600 20" preserveAspectRatio="none">
            <polyline class="wave-path" points="{_wave_pts}"/>
        </svg>
    </div>
    <div class="readout-stat">SYS <span style="color:#5EEAD4;">READY</span></div>
</div>
""", unsafe_allow_html=True)

# Sidebar for data source selection
st.sidebar.title("Data Source Configuration")

data_source = st.sidebar.radio(
    "Select Data Source:",
    ('Upload File', 'MySQL Database')
)

# AI Engine Sidebar Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Intelligence Engine")

ai_key_input = st.sidebar.text_input(
    "Google Gemini API Key",
    value=st.session_state.get('ai_api_key', AI_CONFIG.get('api_key', '')),
    type="password",
    help="Enter Gemini API key to enable generative reasoning. If blank, the dashboard operates via the Intelligent Heuristic Engine."
)
st.session_state.ai_api_key = ai_key_input

models_list = ['gemini-flash-lite-latest', 'gemini-3.5-flash-lite', 'gemini-3.7-flash', 'gemini-3.6-flash']
current_saved_model = st.session_state.get('ai_model', AI_CONFIG.get('model', 'gemini-flash-lite-latest'))
default_idx = models_list.index(current_saved_model) if current_saved_model in models_list else 0

ai_model_choice = st.sidebar.selectbox(
    "Gemini Model",
    models_list,
    index=default_idx
)
st.session_state.ai_model = ai_model_choice

_has_key = bool(ai_key_input and ai_key_input.strip())
if _has_key:
    st.sidebar.markdown(f"""
    <div class="ai-badge" style="border-color:#5EEAD4; color:#5EEAD4;">
        <span class="status-dot" style="background:#5EEAD4;"></span>
        ONLINE · {ai_model_choice.upper()}
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class="ai-badge" style="border-color:#FDBA74; color:#FDBA74;">
        <span class="status-dot" style="background:#FDBA74;"></span>
        HEURISTIC ENGINE · OFFLINE READY
    </div>
    """, unsafe_allow_html=True)

if st.sidebar.button("Test AI Connection"):
    if _has_key:
        with st.sidebar.spinner("Testing Gemini connection..."):
            ok, msg = ai_engine.test_gemini_connection(ai_key_input, model=ai_model_choice)
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(f"Failed: {msg}")
    else:
        st.sidebar.info("Operating in Heuristic Mode (no API key required). Enter a Gemini key to activate generative LLM reasoning.")

st.sidebar.markdown("---")

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
if 'date_columns' not in st.session_state:
    st.session_state.date_columns = []
if 'mysql_error' not in st.session_state:
    st.session_state.mysql_error = None
if 'ai_chat_history' not in st.session_state:
    st.session_state.ai_chat_history = []
if 'ai_api_key' not in st.session_state:
    st.session_state.ai_api_key = AI_CONFIG.get('api_key', '')
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = AI_CONFIG.get('model', 'gemini-3.6-flash')
if 'ai_exec_summary' not in st.session_state:
    st.session_state.ai_exec_summary = None

# Data loading functions
@st.cache_data
def load_excel_data(file):
    """Load data from Excel file"""
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None

@st.cache_data
def load_csv_data(file):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

@st.cache_data
def load_json_data(file):
    """Load data from JSON file"""
    try:
        df = pd.read_json(file)
        return df
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return None

@st.cache_data
def load_parquet_data(file):
    """Load data from Parquet file"""
    try:
        df = pd.read_parquet(file)
        return df
    except Exception as e:
        st.error(f"Error loading Parquet file: {str(e)}")
        return None

@st.cache_data
def load_feather_data(file):
    """Load data from Feather file"""
    try:
        df = pd.read_feather(file)
        return df
    except Exception as e:
        st.error(f"Error loading Feather file: {str(e)}")
        return None

@st.cache_data
def load_mysql_data(host, user, password, database, query):
    """Load data from MySQL database. Returns (DataFrame, error_message)."""
    try:
        # Split host and port if port is specified
        if ':' in host:
            host_parts = host.split(':')
            host = host_parts[0]
            port = int(host_parts[1])
        else:
            port = 3306  # Default MySQL port

        connection = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )

        df = pd.read_sql(query, connection)
        connection.close()
        return df, None
    except Exception as e:
        return None, str(e)

# Function to automatically detect column types (optimized for high speed on multi-million rows)
def detect_column_types(df):
    """Automatically detect numeric, categorical, and date columns efficiently using a fast sample"""
    numeric_columns = []
    categorical_columns = []
    date_columns = []
    
    # Inspect first 5,000 rows for instant type detection instead of scanning millions of rows
    sample_df = df.iloc[:5000] if len(df) > 5000 else df
    
    for col in df.columns:
        if sample_df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'float16']:
            numeric_columns.append(col)
        elif sample_df[col].dtype == 'object':
            # Check if it's a date
            try:
                pd.to_datetime(sample_df[col].dropna().iloc[:5])
                date_columns.append(col)
            except:
                # Check if it's categorical
                if sample_df[col].nunique() <= 50:
                    categorical_columns.append(col)
                else:
                    try:
                        pd.to_numeric(sample_df[col].dropna().iloc[:100])
                        numeric_columns.append(col)
                    except:
                        categorical_columns.append(col)
        elif 'datetime' in str(sample_df[col].dtype):
            date_columns.append(col)
        else:
            categorical_columns.append(col)
    
    return numeric_columns, categorical_columns, date_columns

# Advanced analytics functions (optimized for multi-million row scale)
def perform_clustering(df, numeric_columns, n_clusters=3):
    """Perform K-means clustering on numeric data with smart sampling"""
    if len(numeric_columns) < 2:
        return None, "Need at least 2 numeric columns for clustering"
    
    # Prepare data with smart sampling for multi-million row datasets
    sample_size = min(25000, len(df))
    data = df[numeric_columns].sample(sample_size, random_state=42).dropna() if len(df) > 50000 else df[numeric_columns].dropna()
    if len(data) < n_clusters:
        return None, f"Not enough data points ({len(data)}) for {n_clusters} clusters"
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Calculate silhouette score (subsample to avoid O(N^2) delay)
    try:
        eval_sample = min(5000, len(scaled_data))
        sil_score = silhouette_score(scaled_data[:eval_sample], cluster_labels[:eval_sample])
    except:
        sil_score = -1
    
    # Add cluster labels to dataframe
    result_df = data.copy()
    result_df['Cluster'] = cluster_labels
    
    return result_df, sil_score

def detect_anomalies(df, numeric_columns):
    """Detect anomalies using Isolation Forest with smart sampling"""
    if len(numeric_columns) < 2:
        return None, "Need at least 2 numeric columns for anomaly detection"
    
    # Prepare data with smart sampling
    sample_size = min(25000, len(df))
    data = df[numeric_columns].sample(sample_size, random_state=42).dropna() if len(df) > 50000 else df[numeric_columns].dropna()
    if len(data) < 10:
        return None, "Need at least 10 data points for anomaly detection"
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Detect anomalies with optimized parameters
    iso_forest = IsolationForest(random_state=42, n_estimators=50, max_samples=min(1000, len(data)))
    anomaly_labels = iso_forest.fit_predict(scaled_data)
    
    # Add anomaly labels to dataframe
    result_df = data.copy()
    result_df['Anomaly'] = anomaly_labels  # -1 for anomalies, 1 for normal
    
    # Count anomalies
    anomaly_count = sum(anomaly_labels == -1)
    
    return result_df, anomaly_count

def perform_pca(df, numeric_columns, n_components=2):
    """Perform Principal Component Analysis with smart sampling"""
    if len(numeric_columns) < 2:
        return None, "Need at least 2 numeric columns for PCA"
    
    # Prepare data with smart sampling
    sample_size = min(25000, len(df))
    data = df[numeric_columns].sample(sample_size, random_state=42).dropna() if len(df) > 50000 else df[numeric_columns].dropna()
    if len(data) < n_components:
        return None, f"Not enough data points ({len(data)}) for {n_components} components"
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, len(numeric_columns)))
    pca_result = pca.fit_transform(scaled_data)
    
    # Create result dataframe
    pca_columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
    result_df = pd.DataFrame(pca_result)
    result_df.columns = list(pca_columns)
    
    # Add explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    return result_df, explained_variance

# Add new advanced analytics functions
def perform_regression_analysis(df, numeric_columns):
    """Perform simple regression analysis between pairs of numeric columns"""
    if len(numeric_columns) < 2:
        return None, "Need at least 2 numeric columns for regression analysis"
    
    sample_df = df.sample(min(25000, len(df)), random_state=42) if len(df) > 50000 else df
    
    results = []
    for i in range(min(5, len(numeric_columns))):  # Limit to first 5 columns
        for j in range(i+1, min(5, len(numeric_columns))):
            col1, col2 = numeric_columns[i], numeric_columns[j]
            clean_data = sample_df[[col1, col2]].dropna()
            if len(clean_data) > 10:  # Need minimum data points
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                X = clean_data[col1].values.reshape(-1, 1)
                y = clean_data[col2].values
                
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                results.append({
                    'X_Variable': col1,
                    'Y_Variable': col2,
                    'Coefficient': model.coef_[0],
                    'Intercept': model.intercept_,
                    'R_Squared': r2
                })
    
    if results:
        return pd.DataFrame(results), f"Found {len(results)} relationships"
    else:
        return None, "Not enough data for regression analysis"

def generate_data_profile(df):
    """Generate comprehensive data profile (instant execution)"""
    profile = {
        'Dataset_Info': {
            'Total_Rows': len(df),
            'Total_Columns': len(df.columns),
            'Memory_Usage_MB': round(df.memory_usage(deep=False).sum() / 1024 / 1024, 2)
        },
        'Column_Types': {
            'Numeric': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical': len(df.select_dtypes(include=['object', 'category']).columns),
            'DateTime': len(df.select_dtypes(include=['datetime', 'timedelta']).columns)
        },
        'Missing_Data': {}
    }
    
    # Missing data analysis
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    profile['Missing_Data'] = missing_pct.to_dict()
    
    return profile

# Data source handling
if data_source == 'Upload File':
    st.sidebar.subheader("File Upload")
    
    # Add information about supported file types
    st.sidebar.info("""
    **Supported File Types:**
    - Excel: .xlsx, .xls
    - CSV: .csv
    - JSON: .json
    - Parquet: .parquet
    - Feather: .feather
    """)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file", 
        type=['xlsx', 'xls', 'csv', 'json', 'parquet', 'feather'],
        key="file_uploader_main"
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            try:
                if file_extension in ['xlsx', 'xls']:
                    df = load_excel_data(uploaded_file)
                elif file_extension == 'csv':
                    df = load_csv_data(uploaded_file)
                elif file_extension == 'json':
                    df = load_json_data(uploaded_file)
                elif file_extension == 'parquet':
                    df = load_parquet_data(uploaded_file)
                elif file_extension == 'feather':
                    df = load_feather_data(uploaded_file)
                else:
                    st.error("Unsupported file type. Please upload a supported file format.")
                    df = None
                    
                if df is not None:
                    st.session_state.df = df
                    # Auto-detect column types
                    numeric_cols, categorical_cols, date_cols = detect_column_types(df)
                    st.session_state.numeric_columns = numeric_cols
                    st.session_state.categorical_columns = categorical_cols
                    st.session_state.date_columns = date_cols
                    st.success(f"Data loaded successfully from {file_extension.upper()} file!")
                else:
                    st.error("Failed to load data from the uploaded file. Please check the file format and try again.")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info("Please make sure the file is not corrupted and is in the correct format.")

elif data_source == 'MySQL Database':
    st.sidebar.subheader("MySQL Connection")
    host = st.sidebar.text_input("Host", MYSQL_CONFIG['host'])
    user = st.sidebar.text_input("Username", MYSQL_CONFIG['user'])
    password = st.sidebar.text_input("Password", type="password", value=MYSQL_CONFIG['password'])
    database = st.sidebar.text_input("Database", MYSQL_CONFIG['database'])
    query = st.sidebar.text_area("SQL Query", DEFAULT_QUERY)

    if st.sidebar.button("Connect to MySQL"):
        if all([host, user, password, database, query]):
            with st.spinner("Connecting to database..."):
                df, err = load_mysql_data(host, user, password, database, query)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.mysql_error = None
                    numeric_cols, categorical_cols, date_cols = detect_column_types(df)
                    st.session_state.numeric_columns = numeric_cols
                    st.session_state.categorical_columns = categorical_cols
                    st.session_state.date_columns = date_cols
                    st.success("Data loaded successfully from MySQL!")
                else:
                    st.session_state.mysql_error = err
        else:
            st.warning("Please fill all connection fields")

    # Show error only once, right after a failed connect attempt
    if st.session_state.get('mysql_error'):
        st.error(f"Error connecting to MySQL: {st.session_state.mysql_error}")
        st.caption("Check that the database exists, credentials are correct, and MySQL is running.")

# Main dashboard content
if st.session_state.df is not None:
    df = st.session_state.df
    numeric_columns = st.session_state.numeric_columns
    categorical_columns = st.session_state.categorical_columns
    date_columns = st.session_state.date_columns
    
    # Display raw data toggle (optimized to prevent browser crashing on multi-million rows)
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data Preview")
        preview_limit = min(1000, len(df))
        st.dataframe(df.head(preview_limit), use_container_width=True)
        if len(df) > preview_limit:
            st.caption(f"Showing first {preview_limit:,} rows of {len(df):,} total records for instantaneous performance.")
    
    # Data info with metric cards (instrument panel style)
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Columns</div>
            <div class="metric-value">{len(df.columns)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Numeric Columns</div>
            <div class="metric-value">{len(numeric_columns)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show detected column types
    with st.expander("Auto-detected Column Types"):
        st.write("**Numeric Columns:**")
        for col in numeric_columns:
            st.markdown(f"<div class='auto-column'>{col}</div>", unsafe_allow_html=True)
        
        st.write("**Categorical Columns:**")
        for col in categorical_columns:
            st.markdown(f"<div class='auto-column'>{col}</div>", unsafe_allow_html=True)
        
        st.write("**Date Columns:**")
        for col in date_columns:
            st.markdown(f"<div class='auto-column'>{col}</div>", unsafe_allow_html=True)
    
    # Tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Overview", "📊 Visualizations", "🔍 Insights", "⚙️ Custom Analysis", "🤖 Advanced Analytics", "📈 Data Profiling", "🧠 AI Intelligence"])
    
    with tab1:
        st.subheader("Dataset Summary")
        
        # Basic statistics for numeric columns (sampled for instantaneous execution on 2.7M+ rows)
        if numeric_columns:
            st.write("Numeric Columns Summary:")
            calc_summary_df = df[numeric_columns].sample(min(50000, len(df)), random_state=42) if len(df) > 50000 else df[numeric_columns]
            st.dataframe(calc_summary_df.describe().round(2), use_container_width=True)
            if len(df) > 50000:
                st.caption(f"Estimated from high-precision representative sample of 50,000 records ({len(df):,} total).")
        
        # Value counts for categorical columns
        if categorical_columns:
            st.write("Categorical Columns Summary:")
            selected_cat_col = st.selectbox("Select categorical column:", categorical_columns, key="tab1_cat_col")
            if selected_cat_col:
                cat_sample = df[selected_cat_col].sample(min(50000, len(df)), random_state=42) if len(df) > 50000 else df[selected_cat_col]
                value_counts = cat_sample.value_counts().head(10)
                st.bar_chart(value_counts)
                st.write(f"Top 10 values in {selected_cat_col}:")
                st.dataframe(value_counts, use_container_width=True)

    with tab2:
        st.subheader("Data Visualizations")
        
        # Create fast visualization sample for 2.7M+ rows
        plot_df = df.sample(min(50000, len(df)), random_state=42) if len(df) > 50000 else df
        if len(df) > 50000:
            st.caption(f"Visualizing 50,000 representative data points for 60fps interactive rendering ({len(df):,} records total).")

        if numeric_columns:
            # Create visualization options
            viz_type = st.selectbox(
                "Select Visualization Type:",
                ["Histogram", "Scatter Plot", "Box Plot", "Heatmap", "Line Chart", "Area Chart", "3D Scatter"],
                key="viz_type_select"
            )
            
            if viz_type == "Histogram":
                # Histogram for numeric columns
                selected_num_col = st.selectbox("Select numeric column for histogram:", numeric_columns, key="hist_col")
                if selected_num_col:
                    nbins = st.slider("Number of bins", 5, 50, 20)
                    fig_hist = px.histogram(plot_df, x=selected_num_col, nbins=nbins, 
                                      title=f"Distribution of {selected_num_col}",
                                      color_discrete_sequence=['#5EEAD4'])
                    fig_hist.update_layout(
                        plot_bgcolor='rgba(11,14,20,0)',
                        paper_bgcolor='rgba(11,14,20,0)',
                        font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                        xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                        yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            elif viz_type == "Scatter Plot":
                # Scatter plot for two numeric columns
                if len(numeric_columns) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("Select X-axis:", numeric_columns, key="scatter_x")
                    with col2:
                        y_col = st.selectbox("Select Y-axis:", [col for col in numeric_columns if col != x_col], key="scatter_y")
                    
                    # Optional color dimension
                    color_col = st.selectbox("Select color dimension (optional):", [None] + categorical_columns + numeric_columns, key="scatter_color")
                    
                    if x_col and y_col:
                        if color_col:
                            fig_scatter = px.scatter(plot_df, x=x_col, y=y_col, color=color_col,
                                                   title=f"{y_col} vs {x_col} (colored by {color_col})",
                                                   color_continuous_scale=['#12161F','#5EEAD4'])
                        else:
                            fig_scatter = px.scatter(plot_df, x=x_col, y=y_col, 
                                                   title=f"{y_col} vs {x_col}",
                                                   color_discrete_sequence=['#A5B4FC'])
                        fig_scatter.update_layout(
                            plot_bgcolor='rgba(11,14,20,0)',
                            paper_bgcolor='rgba(11,14,20,0)',
                            font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                            xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                            yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
            
            elif viz_type == "Box Plot":
                # Box plot for numeric columns
                selected_num_col = st.selectbox("Select numeric column for box plot:", numeric_columns, key="box_col")
                if categorical_columns:
                    category_col = st.selectbox("Select category column (optional):", [None] + categorical_columns, key="box_category")
                    if category_col:
                        fig_box = px.box(plot_df, x=category_col, y=selected_num_col,
                                       title=f"Distribution of {selected_num_col} by {category_col}",
                                       color=category_col,
                                       color_discrete_sequence=px.colors.qualitative.Set3)
                    else:
                        fig_box = px.box(plot_df, y=selected_num_col,
                                       title=f"Distribution of {selected_num_col}",
                                       color_discrete_sequence=['#5EEAD4'])
                    fig_box.update_layout(
                        plot_bgcolor='rgba(11,14,20,0)',
                        paper_bgcolor='rgba(11,14,20,0)',
                        font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                        xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                        yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    fig_box = px.box(plot_df, y=selected_num_col,
                                   title=f"Distribution of {selected_num_col}",
                                   color_discrete_sequence=['#5EEAD4'])
                    fig_box.update_layout(
                        plot_bgcolor='rgba(11,14,20,0)',
                        paper_bgcolor='rgba(11,14,20,0)',
                        font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                        xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                        yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
            
            elif viz_type == "Heatmap":
                # Correlation heatmap
                if len(numeric_columns) > 1:
                    # Select columns for correlation
                    selected_cols = st.multiselect("Select columns for correlation matrix:", 
                                                 numeric_columns, 
                                                 default=numeric_columns[:min(10, len(numeric_columns))],
                                                 key="heatmap_cols")
                    
                    if selected_cols and len(selected_cols) > 1:
                        corr_subset = plot_df[selected_cols]
                        if not isinstance(corr_subset, pd.DataFrame):
                            corr_subset = pd.DataFrame(corr_subset)
                        corr_data = corr_subset.corr()
                        fig_heatmap = px.imshow(corr_data, 
                                              title="Correlation Heatmap",
                                              color_continuous_scale='RdBu_r',
                                              aspect="auto")
                        fig_heatmap.update_layout(
                            plot_bgcolor='rgba(11,14,20,0)',
                            paper_bgcolor='rgba(11,14,20,0)',
                            font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                            xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                            yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.warning("Please select at least 2 columns for correlation analysis")
                else:
                    st.info("Need at least 2 numeric columns for heatmap")
            
            elif viz_type == "Line Chart":
                # Line chart for time series or sequential data
                if len(numeric_columns) >= 1:
                    y_col = st.selectbox("Select Y-axis:", numeric_columns, key="line_y")
                    if date_columns:
                        x_col = st.selectbox("Select X-axis (time):", date_columns, key="line_x_time")
                    else:
                        x_col = st.selectbox("Select X-axis:", [None] + numeric_columns, key="line_x")
                    
                    if y_col and x_col:
                        fig_line = px.line(plot_df, x=x_col, y=y_col,
                                         title=f"{y_col} over {x_col}",
                                         color_discrete_sequence=['#5EEAD4'])
                        fig_line.update_layout(
                            plot_bgcolor='rgba(11,14,20,0)',
                            paper_bgcolor='rgba(11,14,20,0)',
                            font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                            xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                            yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                        )
                        st.plotly_chart(fig_line, use_container_width=True)
            
            elif viz_type == "Area Chart":
                # Area chart
                if len(numeric_columns) >= 1:
                    y_col = st.selectbox("Select Y-axis for area chart:", numeric_columns, key="area_y")
                    if date_columns:
                        x_col = st.selectbox("Select X-axis (time) for area chart:", date_columns, key="area_x_time")
                    else:
                        x_col = st.selectbox("Select X-axis for area chart:", [None] + numeric_columns, key="area_x")
                    
                    if y_col and x_col:
                        fig_area = px.area(plot_df, x=x_col, y=y_col,
                                         title=f"{y_col} over {x_col}",
                                         color_discrete_sequence=['#A5B4FC'])
                        fig_area.update_layout(
                            plot_bgcolor='rgba(11,14,20,0)',
                            paper_bgcolor='rgba(11,14,20,0)',
                            font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                            xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                            yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                        )
                        st.plotly_chart(fig_area, use_container_width=True)
            
            elif viz_type == "3D Scatter":
                # 3D Scatter plot
                if len(numeric_columns) >= 3:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_col = st.selectbox("Select X-axis:", numeric_columns, key="3d_x")
                    with col2:
                        y_col = st.selectbox("Select Y-axis:", [col for col in numeric_columns if col != x_col], key="3d_y")
                    with col3:
                        z_col = st.selectbox("Select Z-axis:", [col for col in numeric_columns if col != x_col and col != y_col], key="3d_z")
                    
                    # Optional color dimension
                    color_col = st.selectbox("Select color dimension (optional):", [None] + categorical_columns + numeric_columns, key="3d_color")
                    
                    if x_col and y_col and z_col:
                        if color_col:
                            fig_3d = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col, color=color_col,
                                                 title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}",
                                                 color_continuous_scale=['#12161F','#5EEAD4'])
                        else:
                            fig_3d = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col,
                                                 title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}",
                                                 color_discrete_sequence=['#5EEAD4'])
                        fig_3d.update_layout(
                            plot_bgcolor='rgba(11,14,20,0)',
                            paper_bgcolor='rgba(11,14,20,0)',
                            font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                            xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                            yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.info("Need at least 3 numeric columns for 3D scatter plot")
        else:
            st.info("No numeric columns detected for visualization.")

    with tab3:
        st.subheader("Key Insights")
        
        # Correlation matrix for numeric columns
        if len(numeric_columns) > 1:
            st.write("### Correlation Matrix")
            # Select columns for correlation
            selected_cols = st.multiselect("Select columns for detailed correlation analysis:", 
                                         numeric_columns, 
                                         default=numeric_columns[:min(5, len(numeric_columns))],
                                         key="insights_corr")
            
            if selected_cols and len(selected_cols) > 1:
                # Ensure we're working with a DataFrame
                corr_subset = df[selected_cols]
                if not isinstance(corr_subset, pd.DataFrame):
                    corr_subset = pd.DataFrame(corr_subset)
                corr_data = corr_subset.corr()
                fig_heatmap = px.imshow(corr_data, 
                                  title="Correlation Heatmap",
                                  color_continuous_scale='RdBu_r',
                                  aspect="auto",
                                  text_auto=True)
                fig_heatmap.update_layout(
                    plot_bgcolor='rgba(11,14,20,0)',
                    paper_bgcolor='rgba(11,14,20,0)',
                    font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                    xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                    yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Show strong correlations
                st.write("### Strong Correlations")
                strong_corr = []
                for i in range(len(corr_data.columns)):
                    for j in range(i+1, len(corr_data.columns)):
                        corr_val = corr_data.iloc[i, j]
                        if abs(corr_val) > 0.7:  # Strong correlation threshold
                            strong_corr.append({
                                'Variable 1': corr_data.columns[i],
                                'Variable 2': corr_data.columns[j],
                                'Correlation': round(corr_val, 3)
                            })
            
                if strong_corr:
                    strong_corr_df = pd.DataFrame(strong_corr)
                    st.dataframe(strong_corr_df, use_container_width=True)
                else:
                    st.info("No strong correlations found (|r| > 0.7)")
            else:
                st.warning("Please select at least 2 columns for correlation analysis")
        
        # Distribution analysis
        st.write("### Distribution Analysis")
        if numeric_columns:
            selected_col = st.selectbox("Select column for distribution analysis:", numeric_columns, key="dist_col")
            if selected_col:
                # Show distribution statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{df[selected_col].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[selected_col].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{df[selected_col].std():.2f}")
                with col4:
                    st.metric("Skewness", f"{df[selected_col].skew():.2f}")
                
                # Distribution visualization
                fig_dist = px.histogram(df, x=selected_col, nbins=30,
                                  title=f"Distribution of {selected_col}",
                                  color_discrete_sequence=['#5EEAD4'],
                                  marginal="box")
                fig_dist.update_layout(
                    plot_bgcolor='rgba(11,14,20,0)',
                    paper_bgcolor='rgba(11,14,20,0)',
                    font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                    xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                    yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # Categorical analysis
        if categorical_columns:
            st.write("### Categorical Analysis")
            selected_cat_col = st.selectbox("Select categorical column:", categorical_columns, key="insights_cat_col")
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts()
                
                # Bar chart
                fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Count of {selected_cat_col}",
                           color_discrete_sequence=['#A5B4FC'])
                fig_bar.update_layout(
                    plot_bgcolor='rgba(11,14,20,0)',
                    paper_bgcolor='rgba(11,14,20,0)',
                    font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                    xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                    yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Pie chart
                fig_pie = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Distribution of {selected_cat_col}",
                           color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab4:
        st.subheader("Custom Analysis")
        
        # Multiple filter options
        st.write("### Multi-Column Filtering")
        
        # Select columns to filter on
        filter_columns = st.multiselect("Select columns to filter:", df.columns.tolist(), key="filter_cols")
        
        if filter_columns:
            filtered_df = df.copy()
            
            # Create filters for each selected column
            for col in filter_columns:
                st.write(f"#### Filter by {col}")
                if col in numeric_columns:
                    # Numeric filter
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    selected_range = st.slider(
                        f"Select range for {col}:", 
                        min_val, max_val, 
                        (min_val, max_val),
                        key=f"num_filter_{col}"
                    )
                    filtered_df = filtered_df[
                        (filtered_df[col] >= selected_range[0]) & 
                        (filtered_df[col] <= selected_range[1])
                    ]
                elif col in categorical_columns:
                    # Categorical filter with unique key per column
                    val_counts = df[col].value_counts()
                    if len(val_counts) > 100:
                        search_term = st.text_input(
                            f"Search in {col} (text search):", 
                            key=f"cat_search_{col}",
                            placeholder=f"Type letters to search {col}..."
                        )
                        if search_term:
                            filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)]
                        else:
                            st.caption(f"ℹ️ High cardinality ({len(val_counts):,} unique values). Showing top 50 most frequent.")
                            top_vals = val_counts.head(50).index.tolist()
                            selected_values = st.multiselect(
                                f"Select values for {col}:", 
                                top_vals, 
                                default=top_vals,
                                key=f"cat_filter_values_{col}"
                            )
                            if selected_values and len(selected_values) < len(top_vals):
                                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                    else:
                        unique_values = val_counts.index.tolist()
                        selected_values = st.multiselect(
                            f"Select values for {col}:", 
                            unique_values, 
                            default=unique_values,
                            key=f"cat_filter_values_{col}"
                        )
                        if selected_values and len(selected_values) < len(unique_values):
                            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                elif col in date_columns:
                    # Date filter (handles both datetime.date and Timestamp gracefully)
                    col_dt_series = pd.to_datetime(df[col], errors='coerce')
                    min_date = col_dt_series.min()
                    max_date = col_dt_series.max()
                    if pd.notnull(min_date) and pd.notnull(max_date):
                        selected_dates = st.date_input(
                            f"Select date range for {col}:", 
                            value=(min_date.date(), max_date.date()),
                            key=f"filter_date_{col}"
                        )
                        if len(selected_dates) == 2:
                            filtered_col_dt = pd.to_datetime(filtered_df[col], errors='coerce')
                            start_ts = pd.Timestamp(selected_dates[0])
                            end_ts = pd.Timestamp(selected_dates[1]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
                            filtered_df = filtered_df[
                                (filtered_col_dt >= start_ts) & 
                                (filtered_col_dt <= end_ts)
                            ]
            
            # Show filtered results
            st.write(f"### Filtered Results ({len(filtered_df):,} rows)")
            if len(filtered_df) > 1000:
                st.caption(f"⚡ Showing first 1,000 of {len(filtered_df):,} matching records for maximum responsiveness.")
                st.dataframe(filtered_df.iloc[:1000], use_container_width=True)
            else:
                st.dataframe(filtered_df, use_container_width=True)
            
            # Statistics for filtered data
            if numeric_columns:
                # Ensure filtered_df is a DataFrame
                if not isinstance(filtered_df, pd.DataFrame):
                    filtered_df = pd.DataFrame(filtered_df)
                
                # Check if filtered_df has columns attribute
                if hasattr(filtered_df, 'columns'):
                    numeric_cols_in_filtered = [col for col in numeric_columns if col in filtered_df.columns]
                else:
                    # If filtered_df is a Series or array, handle accordingly
                    numeric_cols_in_filtered = []
                if numeric_cols_in_filtered:
                    st.write("### Statistics for Filtered Data")
                    # Ensure we're working with a DataFrame, not a Series
                    if len(numeric_cols_in_filtered) > 1:
                        numeric_filtered_df = filtered_df[numeric_cols_in_filtered]
                        # Convert to DataFrame if it's not already
                        if not isinstance(numeric_filtered_df, pd.DataFrame):
                            numeric_filtered_df = pd.DataFrame(numeric_filtered_df)
                        # Ensure we're working with numeric data only
                        numeric_filtered_df = numeric_filtered_df.select_dtypes(include=[np.number])
                        if not numeric_filtered_df.empty:
                            desc_calc_df = numeric_filtered_df.sample(min(50000, len(numeric_filtered_df)), random_state=42) if len(numeric_filtered_df) > 50000 else numeric_filtered_df
                            st.dataframe(desc_calc_df.describe().round(2), use_container_width=True)
                            if len(numeric_filtered_df) > 50000:
                                st.caption(f"Estimated from representative sample of 50,000 records ({len(numeric_filtered_df):,} total).")
                    else:
                        # Single column case
                        col = numeric_cols_in_filtered[0]
                        series = filtered_df[col]
                        if pd.api.types.is_numeric_dtype(series):
                            # Convert to Series if it's not already
                            if not isinstance(series, pd.Series):
                                series = pd.Series(series)
                            desc_calc_series = series.sample(min(50000, len(series)), random_state=42) if len(series) > 50000 else series
                            desc = desc_calc_series.describe()
                            # Convert to DataFrame for display
                            desc_df = pd.DataFrame(desc).round(2)
                            st.dataframe(desc_df, use_container_width=True)
        
        # Group by analysis
        st.write("### Group Analysis")
        if categorical_columns:
            group_col = st.selectbox("Select column to group by:", categorical_columns, key="group_col")
            if numeric_columns:
                agg_col = st.selectbox("Select numeric column to aggregate:", numeric_columns, key="agg_col")
                agg_func = st.selectbox("Select aggregation function:", 
                                      ["mean", "sum", "count", "min", "max", "std"])
                
                if group_col and agg_col:
                    grouped_data = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
                    grouped_data.columns = [group_col, f"{agg_func}_{agg_col}"]
                    
                    st.write(f"### {agg_func.title()} of {agg_col} by {group_col}")
                    st.dataframe(grouped_data, use_container_width=True)
                    
                    # Visualization
                    fig_group = px.bar(grouped_data, x=group_col, y=f"{agg_func}_{agg_col}",
                                 title=f"{agg_func.title()} of {agg_col} by {group_col}",
                                 color_discrete_sequence=['#5EEAD4'])
                    fig_group.update_layout(
                        plot_bgcolor='rgba(11,14,20,0)',
                        paper_bgcolor='rgba(11,14,20,0)',
                        font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                        xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                        yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                    )
                    st.plotly_chart(fig_group, use_container_width=True)

    with tab5:
        st.subheader("Advanced Analytics")
        
        if numeric_columns:
            analysis_type = st.radio(
                "Select Analysis Type:",
                ["Clustering", "Anomaly Detection", "Principal Component Analysis", "Regression Analysis", "Time Series Analysis"],
                key="analysis_type"
            )
            
            if analysis_type == "Clustering":
                st.write("### K-Means Clustering")
                col1, col2 = st.columns(2)
                with col1:
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                with col2:
                    # Show only numeric columns for clustering
                    cluster_columns = st.multiselect(
                        "Select Columns for Clustering:",
                        numeric_columns,
                        default=numeric_columns[:min(5, len(numeric_columns))]
                    )
                
                if st.button("Perform Clustering") and cluster_columns:
                    with st.spinner("Performing clustering..."):
                        clustered_data, score = perform_clustering(df, cluster_columns, n_clusters)
                        if isinstance(clustered_data, pd.DataFrame):
                            st.success(f"Clustering completed! Silhouette Score: {score:.3f}")
                            
                            # Show clustered data
                            st.write("Clustered Data:")
                            st.dataframe(clustered_data, use_container_width=True)
                            
                            # Visualization
                            if len(cluster_columns) >= 2:
                                fig = px.scatter(
                                    clustered_data, 
                                    x=cluster_columns[0], 
                                    y=cluster_columns[1],
                                    color='Cluster',
                                    title=f"Clusters Visualization (Score: {score:.3f})",
                                    color_continuous_scale=['#12161F','#5EEAD4']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Cluster statistics
                            st.write("Cluster Statistics:")
                            cluster_stats = clustered_data.groupby('Cluster').agg({
                                col: ['mean', 'std'] for col in cluster_columns
                            }).round(2)
                            st.dataframe(cluster_stats, use_container_width=True)
                        else:
                            st.error(f"Clustering failed: {score}")
            
            elif analysis_type == "Anomaly Detection":
                st.write("### Anomaly Detection")
                st.write("Using Isolation Forest to detect outliers in your data")
                
                # Show only numeric columns for anomaly detection
                anomaly_columns = st.multiselect(
                    "Select Columns for Anomaly Detection:",
                    numeric_columns,
                    default=numeric_columns[:min(5, len(numeric_columns))]
                )
                
                if st.button("Detect Anomalies") and anomaly_columns:
                    with st.spinner("Detecting anomalies..."):
                        anomaly_data, count = detect_anomalies(df, anomaly_columns)
                        if isinstance(anomaly_data, pd.DataFrame):
                            st.success(f"Anomaly detection completed! Found {count} anomalies.")
                            
                            # Show anomaly data
                            st.write("Anomaly Detection Results:")
                            st.dataframe(anomaly_data, use_container_width=True)
                            
                            # Visualization
                            if len(anomaly_columns) >= 2:
                                fig = px.scatter(
                                    anomaly_data, 
                                    x=anomaly_columns[0], 
                                    y=anomaly_columns[1],
                                    color='Anomaly',
                                    title=f"Anomaly Detection Results ({count} anomalies found)",
                                    color_discrete_map={-1: '#FDBA74', 1: '#5EEAD4'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Anomaly statistics
                            st.write("Anomaly Statistics:")
                            anomaly_subset = anomaly_data[anomaly_data['Anomaly'] == -1][anomaly_columns]
                            # Ensure we're working with a DataFrame
                            if not isinstance(anomaly_subset, pd.DataFrame):
                                anomaly_subset = pd.DataFrame(anomaly_subset)
                            elif len(anomaly_subset) > 0:
                                anomaly_stats = anomaly_subset.describe()
                                st.dataframe(anomaly_stats, use_container_width=True)
                            else:
                                st.write("No anomalies found for statistics.")
                        else:
                            st.error(f"Anomaly detection failed: {count}")
            
            elif analysis_type == "Principal Component Analysis":
                st.write("### Principal Component Analysis (PCA)")
                st.write("Reducing dimensionality of your data")
                
                # Show only numeric columns for PCA
                pca_columns = st.multiselect(
                    "Select Columns for PCA:",
                    numeric_columns,
                    default=numeric_columns[:min(5, len(numeric_columns))]
                )
                
                n_components = st.slider("Number of Components", 2, min(10, len(pca_columns)), 2)
                
                if st.button("Perform PCA") and pca_columns:
                    with st.spinner("Performing PCA..."):
                        pca_data, explained_variance = perform_pca(df, pca_columns, n_components)
                        if isinstance(pca_data, pd.DataFrame):
                            st.success("PCA completed!")
                            
                            # Show explained variance
                            st.write("Explained Variance Ratio:")
                            variance_df = pd.DataFrame({
                                'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                                'Variance Explained': explained_variance,
                                'Cumulative Variance': np.cumsum(explained_variance)
                            })
                            st.dataframe(variance_df, use_container_width=True)
                            
                            # Visualization
                            if len(pca_data.columns) >= 2:
                                fig = px.scatter(
                                    pca_data, 
                                    x='PC1', 
                                    y='PC2',
                                    title="PCA Visualization",
                                    color_continuous_scale=['#12161F','#5EEAD4']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show PCA results
                            st.write("PCA Results:")
                            st.dataframe(pca_data, use_container_width=True)
                        else:
                            st.error(f"PCA failed: {explained_variance}")
            
            elif analysis_type == "Regression Analysis":
                st.write("### Regression Analysis")
                st.write("Finding relationships between numeric variables")
                
                if st.button("Perform Regression Analysis"):
                    with st.spinner("Analyzing relationships..."):
                        regression_results, message = perform_regression_analysis(df, numeric_columns)
                        if isinstance(regression_results, pd.DataFrame):
                            st.success(message)
                            
                            # Show results
                            st.write("Regression Results:")
                            st.dataframe(regression_results, use_container_width=True)
                            
                            # Visualization
                            if len(regression_results) > 0:
                                fig = px.scatter(
                                    regression_results, 
                                    x='Coefficient', 
                                    y='R_Squared',
                                    hover_data=['X_Variable', 'Y_Variable'],
                                    title="Regression Analysis Results",
                                    color='R_Squared',
                                    color_continuous_scale=['#12161F','#5EEAD4']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(message)
            
            elif analysis_type == "Time Series Analysis":
                st.write("### Time Series Analysis")
                if date_columns and numeric_columns:
                    # Select date column
                    date_col = st.selectbox("Select date column:", date_columns, key="ts_date")
                    # Select numeric column to analyze
                    value_col = st.selectbox("Select value column:", numeric_columns, key="ts_value")
                    
                    if date_col and value_col:
                        # Prepare time series data
                        ts_data = df[[date_col, value_col]].copy()
                        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
                        ts_data = ts_data.sort_index()
                        ts_data = ts_data.set_index(date_col)
                        
                        # Resample to handle duplicates
                        ts_data = ts_data.resample('D').mean()  # Daily average
                        
                        # Show time series plot
                        fig_ts = px.line(ts_data, x=ts_data.index, y=value_col,
                                       title=f"Time Series: {value_col} over time",
                                       color_discrete_sequence=['#5EEAD4'])
                        fig_ts.update_layout(
                            plot_bgcolor='rgba(11,14,20,0)',
                            paper_bgcolor='rgba(11,14,20,0)',
                            font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                            xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                            yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)
                        
                        # Moving averages
                        st.write("### Moving Averages")
                        window = st.slider("Select moving average window:", 3, 30, 7)
                        ts_data[f'MA_{window}'] = ts_data[value_col].rolling(window=window).mean()
                        
                        fig_ma = px.line(ts_data, 
                                       title=f"Moving Average (Window: {window})",
                                       color_discrete_sequence=['#5EEAD4', '#A5B4FC'])
                        fig_ma.update_layout(
                            plot_bgcolor='rgba(11,14,20,0)',
                            paper_bgcolor='rgba(11,14,20,0)',
                            font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
                            xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
                            yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False)
                        )
                        st.plotly_chart(fig_ma, use_container_width=True)
                        
                        # Trend analysis
                        st.write("### Trend Analysis")
                        from scipy import stats
                        # Remove NaN values
                        clean_data = ts_data[value_col].dropna()
                        if len(clean_data) > 10:
                            # Calculate trend
                            # Convert to arrays to avoid tuple issues
                            x_vals = np.array(range(len(clean_data)))
                            y_vals = np.array(clean_data.values)
                            # Calculate trend using linregress
                            try:
                                result = stats.linregress(x_vals, y_vals)
                                # Access values and convert using numpy's float conversion
                                slope_f = np.float64(result[0].item())
                                r_value_f = np.float64(result[2].item())
                                p_value_f = np.float64(result[3].item())
                                
                                st.metric("Trend Slope", f"{slope_f:.4f}")
                                st.metric("R-squared", f"{r_value_f**2:.4f}")
                                st.metric("P-value", f"{p_value_f:.4f}")
                                
                                if p_value_f < 0.05:
                                    if slope_f > 0:
                                        st.success("Significant positive trend detected")
                                    else:
                                        st.success("Significant negative trend detected")
                                else:
                                    st.info("No significant trend detected")
                            except Exception as e:
                                st.warning(f"Could not calculate trend: {str(e)}")
                else:
                    st.info("Time series analysis requires date and numeric columns")
        else:
            st.info("Advanced analytics require numeric columns. Please upload a dataset with numeric data.")

    with tab6:
        st.subheader("Data Profiling")
        st.write("Comprehensive analysis of your dataset")
        if st.button("Generate Data Profile"):
            with st.spinner("Generating data profile..."):
                profile = generate_data_profile(df)
                
                # Dataset info
                st.write("### Dataset Information")
                info_df = pd.DataFrame([profile['Dataset_Info']]).T
                info_df.columns = ['Value']
                st.dataframe(info_df, use_container_width=True)
                
                # Column types
                st.write("### Column Types")
                types_df = pd.DataFrame([profile['Column_Types']]).T
                types_df.columns = ['Count']
                st.dataframe(types_df, use_container_width=True)
                
                # Missing data
                if profile['Missing_Data']:
                    st.write("### Missing Data Analysis")
                    missing_df = pd.DataFrame([profile['Missing_Data']]).T
                    missing_df.columns = ['Percentage_Missing']
                    missing_df = missing_df.sort_values('Percentage_Missing', ascending=False)
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Visualization
                    fig = px.bar(
                        missing_df.reset_index(), 
                        x='index', 
                        y='Percentage_Missing',
                        title="Missing Data by Column",
                        labels={'index': 'Column', 'Percentage_Missing': 'Percentage Missing (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No missing data found in your dataset!")
                
                # Data quality metrics (optimized for multi-million row datasets)
                st.write("### Data Quality Metrics")
                sample_n = min(50000, len(df))
                calc_dup_df = df.sample(sample_n, random_state=42) if len(df) > 50000 else df
                dups_in_sample = len(calc_dup_df) - len(calc_dup_df.drop_duplicates())
                dup_pct = round((dups_in_sample / len(calc_dup_df)) * 100, 2)
                est_dups = int(round((dup_pct / 100) * len(df)))

                quality_metrics = {
                    'Duplicate_Rows (Sampled/Est)': est_dups,
                    'Duplicate_Percentage': dup_pct,
                    'Unique_Columns': len([col for col in df.columns if calc_dup_df[col].nunique() == len(calc_dup_df)])
                }
                quality_df = pd.DataFrame([quality_metrics]).T
                quality_df.columns = ['Value']
                st.dataframe(quality_df, use_container_width=True)

    with tab7:
        st.subheader("🧠 AI Data Intelligence Channel")
        st.caption("Multi-modal conversational querying, automated C-suite briefings, natural language visualizations, and quality remediation.")

        subtab_chat, subtab_briefing, subtab_viz, subtab_quality, subtab_driver = st.tabs([
            "💬 Chat with Data",
            "📋 Executive Briefing",
            "🎨 Smart Visualizations",
            "🧹 Quality Advisor",
            "🔮 Metric Driver Analysis"
        ])

        with subtab_chat:
            st.markdown("#### 💬 Conversational Data Analyst")
            st.caption("Converse with your dataset in natural language. Powered by Gemini LLM reasoning and real-time statistical aggregations.")
            
            # Quick query chips
            st.markdown("<p style='font-family:var(--font-mono); font-size:0.7rem; color:#8B95A7; text-transform:uppercase; letter-spacing:0.08em; margin:10px 0 6px 0;'>Quick Queries</p>", unsafe_allow_html=True)
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                if st.button("📊 Average by Category", key="q_chip_1", use_container_width=True):
                    st.session_state.pending_ai_query = f"What is the average of {numeric_columns[0]} across {categorical_columns[0]}?" if (numeric_columns and categorical_columns) else "Summarize the key numeric metrics"
            with col_q2:
                if st.button("🏆 Top Performers", key="q_chip_2", use_container_width=True):
                    st.session_state.pending_ai_query = f"Which {categorical_columns[0]} has the highest {numeric_columns[0]}?" if (numeric_columns and categorical_columns) else "What are the maximum values in each column?"
            with col_q3:
                if st.button("🔗 Key Correlations", key="q_chip_3", use_container_width=True):
                    st.session_state.pending_ai_query = "What are the strongest correlations and relationships between variables?"

            # Query input form
            default_query = st.session_state.pop('pending_ai_query', '')
            user_query = st.text_input(
                "Enter your question:", 
                value=default_query, 
                placeholder="e.g., Which department has the highest average salary and why?", 
                key="ai_chat_input"
            )
            
            col_ask, col_clear = st.columns([5, 1])
            with col_ask:
                ask_submitted = st.button("Ask AI Analyst", key="btn_ask_ai", use_container_width=True)
            with col_clear:
                if st.button("Clear History", key="btn_clear_chat", use_container_width=True):
                    st.session_state.ai_chat_history = []
                    st.rerun()

            if ask_submitted and user_query:
                with st.spinner("Analyzing dataset with AI..."):
                    answer, mode = ai_engine.ask_dataset_ai(
                        df, 
                        user_query, 
                        api_key=st.session_state.get('ai_api_key'),
                        model=st.session_state.get('ai_model')
                    )
                    st.session_state.ai_chat_history.append({
                        'query': user_query,
                        'answer': answer,
                        'mode': mode,
                        'time': datetime.now().strftime("%H:%M:%S")
                    })

            # Render chat history with native clean chat message components
            if st.session_state.ai_chat_history:
                st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
                for item in reversed(st.session_state.ai_chat_history):
                    badge_class = "ai-badge-gemini" if item['mode'] == 'gemini' else "ai-badge-heuristic"
                    badge_text = f"GEMINI ({st.session_state.get('ai_model', '3.7-FLASH').upper()})" if item['mode'] == 'gemini' else "HEURISTIC ENGINE"
                    
                    with st.chat_message("user"):
                        st.markdown(f"**{item['query']}**")
                    
                    with st.chat_message("assistant"):
                        st.markdown(f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#8B95A7;">TIME: {item['time']}</span>
                            <span class="ai-badge {badge_class}">{badge_text}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(ai_engine.sanitize_ai_markdown(item['answer']))
            else:
                st.info("No questions asked yet. Choose a quick query above or type any question about your data!")

        with subtab_briefing:
            st.markdown("#### 📋 Executive Intelligence Briefing")
            st.caption("Generates a formal, high-signal business briefing summarizing dataset perimeter, primary findings, risk factors, and strategic next steps.")
            
            col_b1, col_b2 = st.columns([2, 5])
            with col_b1:
                gen_briefing_clicked = st.button("Generate Executive Briefing", key="btn_gen_exec", use_container_width=True)
            
            if gen_briefing_clicked:
                with st.spinner("Synthesizing executive briefing with Gemini..."):
                    briefing_text, mode = ai_engine.generate_executive_summary(
                        df,
                        api_key=st.session_state.get('ai_api_key'),
                        model=st.session_state.get('ai_model')
                    )
                    st.session_state.ai_exec_summary = {
                        'text': briefing_text,
                        'mode': mode,
                        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

            if st.session_state.ai_exec_summary:
                summary_data = st.session_state.ai_exec_summary
                badge_class = "ai-badge-gemini" if summary_data['mode'] == 'gemini' else "ai-badge-heuristic"
                current_model_name = st.session_state.get('ai_model', 'gemini-3.7-flash').upper()
                badge_label = f"GEMINI ({current_model_name}) ENGINE" if summary_data['mode'] == 'gemini' else "INTELLIGENT HEURISTIC ENGINE"
                
                st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"""
                    <div class="ai-briefing-header">
                        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#8B95A7;">
                            GENERATED: {summary_data['generated_at']} &nbsp;·&nbsp; MONITORING {len(df):,} RECORDS
                        </span>
                        <span class="ai-badge {badge_class}">
                            {badge_label}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(ai_engine.sanitize_ai_markdown(summary_data['text']))

                    st.markdown("---")
                    st.download_button(
                        label="Download Executive Briefing (.md)",
                        data=summary_data['text'],
                        file_name=f"executive_briefing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        key="btn_download_exec"
                    )

        with subtab_viz:
            st.markdown("#### 🎨 Natural Language & Smart Visualization Generator")
            st.caption("Convert plain English instructions into Plotly visualizations or pick from AI-suggested views tailored to your dataset.")

            st.write("##### AI-Recommended Visualizations")
            recs = ai_engine.recommend_smart_charts(df)
            if recs:
                rec_cols = st.columns(min(len(recs), 3))
                for i, rec in enumerate(recs[:3]):
                    with rec_cols[i]:
                        with st.container(border=True):
                            st.markdown(f"**{rec['title']}**")
                            st.caption(rec['description'])
                            if st.button(f"Render Chart #{i+1}", key=f"btn_render_rec_{i}", use_container_width=True):
                                st.session_state.active_smart_chart = rec

            if 'active_smart_chart' in st.session_state and st.session_state.active_smart_chart:
                st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"**Active Visualization: {st.session_state.active_smart_chart['title']}**")
                    rec_fig = ai_engine.render_smart_chart(df, st.session_state.active_smart_chart)
                    st.plotly_chart(rec_fig, use_container_width=True)

            st.markdown("---")
            st.write("##### Custom Natural Language Chart Prompt")
            nl_prompt = st.text_input(
                "Describe the chart you want to create:",
                placeholder="e.g., Box plot of Salary by Department, or Scatter plot of Experience vs Salary",
                key="nl_chart_input"
            )
            if st.button("Generate Chart from Prompt", key="btn_gen_nl_chart", use_container_width=True) and nl_prompt:
                with st.spinner("Generating visualization..."):
                    gen_fig, gen_msg = ai_engine.generate_chart_from_nl(df, nl_prompt)
                    st.success(gen_msg)
                    st.plotly_chart(gen_fig, use_container_width=True)

        with subtab_quality:
            st.markdown("#### 🧹 AI Data Quality & Cleaning Advisor")
            st.caption("Automated diagnostics audit null records, duplicates, and outliers with 1-click remediation that updates your active session data across all tabs.")

            diagnostics = ai_engine.analyze_data_quality_advisor(df)
            if not diagnostics:
                st.success("✅ Clean Health Report: Zero duplicates, missing entries, or significant outliers detected!")
            else:
                for idx, diag in enumerate(diagnostics):
                    severity_color = "#FDBA74" if diag['severity'] == 'HIGH' else "#A5B4FC"
                    with st.container(border=True):
                        col_info, col_btn = st.columns([4, 1])
                        with col_info:
                            st.markdown(f"""
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <strong style="font-size:0.95rem; color:#F1F5F9;">{diag['category']}</strong>
                                <span class="ai-badge" style="border-color:{severity_color}; color:{severity_color};">
                                    SEVERITY: {diag['severity']}
                                </span>
                            </div>
                            <p style="color:#CBD5E1; margin:0 0 6px 0; font-size:0.88rem;">{diag['issue']}</p>
                            <p style="font-size:0.8rem; color:#8B95A7; margin:0;">Recommended Action: {diag['recommendation']}</p>
                            """, unsafe_allow_html=True)
                        with col_btn:
                            action_col = diag.get('col')
                            if st.button("Apply Fix", key=f"btn_fix_{idx}", use_container_width=True):
                                cleaned_df, clean_msg = ai_engine.apply_cleaning_action(
                                    df, 
                                    diag['action_key'], 
                                    col=action_col
                                )
                                st.session_state.df = cleaned_df
                                numeric_cols, categorical_cols, date_cols = detect_column_types(cleaned_df)
                                st.session_state.numeric_columns = numeric_cols
                                st.session_state.categorical_columns = categorical_cols
                                st.session_state.date_columns = date_cols
                                st.success(f"Remediated: {clean_msg}")
                                st.rerun()

        with subtab_driver:
            st.markdown("#### 🔮 Metric Driver & Impact Analyzer")
            st.caption("Discover which numeric parameters correlate most strongly with a target KPI and uncover categorical segment performance gaps.")

            if numeric_columns:
                target_col = st.selectbox("Select Target Variable to Analyze:", numeric_columns, key="driver_target_select")
                if target_col:
                    drivers, err = ai_engine.analyze_metric_drivers(df, target_col)
                    if err:
                        st.error(err)
                    elif drivers:
                        st.markdown(drivers['summary_explanation'])
                        
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.write("##### Strongest Numeric Correlates")
                            if drivers['numeric_correlations']:
                                corr_df = pd.DataFrame(drivers['numeric_correlations'])[['feature', 'correlation', 'strength', 'direction']]
                                st.dataframe(corr_df, use_container_width=True)
                                
                                fig_corr = px.bar(
                                    corr_df, 
                                    x='feature', 
                                    y='correlation', 
                                    title=f"Correlation with {target_col}",
                                    color='correlation',
                                    color_continuous_scale=['#12161F', '#5EEAD4']
                                )
                                fig_corr = ai_engine.style_plotly_fig(fig_corr)
                                st.plotly_chart(fig_corr, use_container_width=True)
                            else:
                                st.info("No other numeric columns found.")
                        
                        with col_d2:
                            st.write("##### Categorical Segment Spreads")
                            if drivers['categorical_drivers']:
                                cat_df = pd.DataFrame(drivers['categorical_drivers'])[['feature', 'spread', 'top_segment', 'top_mean', 'bottom_segment', 'bottom_mean']]
                                st.dataframe(cat_df, use_container_width=True)
                            else:
                                st.info("No categorical columns with between 2-25 segments found.")
            else:
                st.info("Metric driver analysis requires numeric columns in the dataset.")

else:
    st.markdown("""
    <div class="idle-screen">
        <div class="idle-status">NO DATASET LOADED &nbsp;·&nbsp; AWAITING INPUT</div>
        <div class="idle-title">Data Analysis Instrument</div>
        <p class="idle-body">
            Select a data source from the sidebar to begin. Drop in a structured file or
            connect to a MySQL database — the instrument will detect column types automatically
            and activate all seven analysis channels.
        </p>
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#8B95A7;
                  text-transform:uppercase;letter-spacing:0.1em;margin:0 0 10px 0;">
            Accepted input formats
        </p>
        <div class="idle-formats">
            <span class="idle-format-tag">.xlsx</span>
            <span class="idle-format-tag">.xls</span>
            <span class="idle-format-tag">.csv</span>
            <span class="idle-format-tag">.json</span>
            <span class="idle-format-tag">.parquet</span>
            <span class="idle-format-tag">.feather</span>
            <span class="idle-format-tag">mysql</span>
        </div>
        <div class="idle-wave">
            <svg viewBox="0 0 800 28" preserveAspectRatio="none">
                <polyline
                    points="0,14 30,6 50,22 75,10 100,20 130,5 155,18 180,8 210,20 240,4 265,17 290,9 320,21 350,6 375,19 400,10 430,22 460,5 485,18 510,8 540,20 570,4 595,17 620,9 650,21 680,6 705,19 730,10 760,22 790,7 800,14"
                    stroke="#5EEAD4" stroke-width="1.5" fill="none"
                />
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    pass


