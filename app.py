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

# Enhanced Custom CSS for premium design with vibrant colors and consistent styling
st.markdown("""
<style>
    /* Premium gradient background with vibrant colors */
    .main {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #FFFFFF;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Header with premium gold and vibrant accent */
    .main-header {
        font-size: 42px !important;
        font-weight: 800;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 20px 0 30px 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        letter-spacing: 1px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Sub headers with vibrant colors */
    .sub-header {
        font-size: 28px !important;
        font-weight: 700;
        color: #FFD700;
        margin: 25px 0 20px 0;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
        border-bottom: 2px solid rgba(255, 215, 0, 0.5);
        padding-bottom: 10px;
    }
    
    /* Metric cards with glass-morphism and vibrant accents */
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: white;
        text-align: center;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 215, 0, 0.3), transparent);
        transform: rotate(45deg);
        transition: all 0.5s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        background: rgba(255, 255, 255, 0.2);
    }
    
    .metric-card:hover::before {
        transform: rotate(45deg) translate(20%, 20%);
    }
    
    .metric-value {
        font-size: 40px;
        font-weight: 800;
        color: #FFD700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 5px;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 18px;
        color: #E0E0E0;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Data frame styling with vibrant borders */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 6px 25px rgba(0,0,0,0.25);
        backdrop-filter: blur(5px);
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    
    /* Sidebar styling with consistent design */
    [data-testid="stSidebar"] {
        background: rgba(25, 25, 35, 0.95);
        backdrop-filter: blur(15px);
        border-right: 2px solid rgba(255, 215, 0, 0.4);
        padding-top: 20px;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 215, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(255, 215, 0, 0.2);
        transform: translateX(5px);
    }
    
    /* Tab styling with vibrant colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        border-radius: 15px;
        padding: 12px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background: transparent;
        border-radius: 12px;
        color: #E0E0E0;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        padding: 0 20px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFD700;
        background: rgba(255, 215, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        color: #1E1E1E;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        border-bottom: none;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
    }
    
    /* Button styling with premium effects */
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #1E1E1E;
        border: none;
        border-radius: 12px;
        padding: 12px 25px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.4);
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: 0.5s;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.6);
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    /* Slider styling with vibrant colors */
    .stSlider [data-baseweb="slider"] {
        background: rgba(255, 255, 255, 0.15);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #FFD700;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        color: #FFD700 !important;
        font-weight: 600;
        border: 1px solid rgba(255, 215, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 215, 0, 0.2);
        transform: translateX(5px);
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 215, 0, 0.2);
    }
    
    /* Auto column styling */
    .auto-column {
        background: rgba(255, 215, 0, 0.2);
        padding: 15px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 4px solid #FFD700;
        color: #FFFFFF;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .auto-column:hover {
        background: rgba(255, 215, 0, 0.3);
        transform: translateX(5px);
    }
    
    /* File type badge */
    .file-type-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #1E1E1E;
        padding: 8px 15px;
        border-radius: 25px;
        font-size: 15px;
        font-weight: 700;
        margin-left: 15px;
        box-shadow: 0 3px 15px rgba(255, 215, 0, 0.4);
        animation: badgePulse 2s infinite;
    }
    
    @keyframes badgePulse {
        0% { box-shadow: 0 3px 15px rgba(255, 215, 0, 0.4); }
        50% { box-shadow: 0 3px 20px rgba(255, 215, 0, 0.6); }
        100% { box-shadow: 0 3px 15px rgba(255, 215, 0, 0.4); }
    }
    
    /* Info boxes with vibrant styling */
    .stAlert {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    /* Success messages */
    .stSuccess {
        background: rgba(46, 204, 113, 0.25);
        backdrop-filter: blur(12px);
        border-radius: 15px;
        border: 1px solid rgba(46, 204, 113, 0.4);
        box-shadow: 0 5px 20px rgba(46, 204, 113, 0.3);
    }
    
    /* Error messages */
    .stError {
        background: rgba(231, 76, 60, 0.25);
        backdrop-filter: blur(12px);
        border-radius: 15px;
        border: 1px solid rgba(231, 76, 60, 0.4);
        box-shadow: 0 5px 20px rgba(231, 76, 60, 0.3);
    }
    
    /* Warning messages */
    .stWarning {
        background: rgba(241, 196, 15, 0.25);
        backdrop-filter: blur(12px);
        border-radius: 15px;
        border: 1px solid rgba(241, 196, 15, 0.4);
        box-shadow: 0 5px 20px rgba(241, 196, 15, 0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        border: 1px solid rgba(255, 215, 0, 0.3);
        color: #FFFFFF;
    }
    
    /* Checkbox styling */
    .stCheckbox > label > div {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    
    .stCheckbox > label > div:checked {
        background: linear-gradient(45deg, #FFD700, #FFA500);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #FFD700, #FFA500);
    }
    
    /* Text input styling */
    .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        border: 1px solid rgba(255, 215, 0, 0.3);
        color: #FFFFFF;
    }
    
    .stTextInput > div > div:focus {
        border-color: #FFD700;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    /* Text area styling */
    .stTextArea > div > div {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        border: 1px solid rgba(255, 215, 0, 0.3);
        color: #FFFFFF;
    }
    
    .stTextArea > div > div:focus {
        border-color: #FFD700;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Title with enhanced animation
st.markdown('<p class="main-header">📊 Universal Data Analysis Dashboard</p>', unsafe_allow_html=True)

# Sidebar for data source selection
st.sidebar.title("Data Source Configuration")

data_source = st.sidebar.radio(
    "Select Data Source:",
    ('Upload File', 'MySQL Database')
)

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
if 'date_columns' not in st.session_state:
    st.session_state.date_columns = []

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
    """Load data from MySQL database"""
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
        return df
    except Exception as e:
        st.error(f"Error connecting to MySQL: {str(e)}")
        return None

# Function to automatically detect column types
def detect_column_types(df):
    """Automatically detect numeric, categorical, and date columns"""
    numeric_columns = []
    categorical_columns = []
    date_columns = []
    
    for col in df.columns:
        # Try to convert to numeric
        if df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
        elif df[col].dtype == 'object':
            # Check if it's a date
            try:
                pd.to_datetime(df[col].dropna().iloc[:5])  # Test first 5 non-null values
                date_columns.append(col)
            except:
                # Check if it's categorical (low cardinality)
                if df[col].nunique() <= min(20, len(df) * 0.1):  # Less than 10% unique values
                    categorical_columns.append(col)
                else:
                    # Try to convert to numeric
                    try:
                        pd.to_numeric(df[col].dropna())
                        numeric_columns.append(col)
                    except:
                        categorical_columns.append(col)
        else:
            categorical_columns.append(col)
    
    return numeric_columns, categorical_columns, date_columns

# Advanced analytics functions
def perform_clustering(df, numeric_columns, n_clusters=3):
    """Perform K-means clustering on numeric data"""
    if len(numeric_columns) < 2:
        return None, "Need at least 2 numeric columns for clustering"
    
    # Prepare data
    data = df[numeric_columns].dropna()
    if len(data) < n_clusters:
        return None, f"Not enough data points ({len(data)}) for {n_clusters} clusters"
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Calculate silhouette score
    try:
        sil_score = silhouette_score(scaled_data, cluster_labels)
    except:
        sil_score = -1
    
    # Add cluster labels to dataframe
    result_df = data.copy()
    result_df['Cluster'] = cluster_labels
    
    return result_df, sil_score

def detect_anomalies(df, numeric_columns):
    """Detect anomalies using Isolation Forest"""
    if len(numeric_columns) < 2:
        return None, "Need at least 2 numeric columns for anomaly detection"
    
    # Prepare data
    data = df[numeric_columns].dropna()
    if len(data) < 10:
        return None, "Need at least 10 data points for anomaly detection"
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Detect anomalies
    iso_forest = IsolationForest(random_state=42)
    anomaly_labels = iso_forest.fit_predict(scaled_data)
    
    # Add anomaly labels to dataframe
    result_df = data.copy()
    result_df['Anomaly'] = anomaly_labels  # -1 for anomalies, 1 for normal
    
    # Count anomalies
    anomaly_count = sum(anomaly_labels == -1)
    
    return result_df, anomaly_count

def perform_pca(df, numeric_columns, n_components=2):
    """Perform Principal Component Analysis"""
    if len(numeric_columns) < 2:
        return None, "Need at least 2 numeric columns for PCA"
    
    # Prepare data
    data = df[numeric_columns].dropna()
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
    # Convert to proper format for DataFrame columns
    result_df = pd.DataFrame(pca_result)
    # Set column names using assignment
    result_df.columns = list(pca_columns)
    
    # Add explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    return result_df, explained_variance

# Add new advanced analytics functions
def perform_regression_analysis(df, numeric_columns):
    """Perform simple regression analysis between pairs of numeric columns"""
    if len(numeric_columns) < 2:
        return None, "Need at least 2 numeric columns for regression analysis"
    
    results = []
    for i in range(min(5, len(numeric_columns))):  # Limit to first 5 columns
        for j in range(i+1, min(5, len(numeric_columns))):
            col1, col2 = numeric_columns[i], numeric_columns[j]
            # Drop rows with NaN values
            clean_data = df[[col1, col2]].dropna()
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
    """Generate comprehensive data profile"""
    profile = {
        'Dataset_Info': {
            'Total_Rows': len(df),
            'Total_Columns': len(df.columns),
            'Memory_Usage_MB': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
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
                    st.balloons()  # Add a fun animation when data is loaded
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
                df = load_mysql_data(host, user, password, database, query)
                if df is not None:
                    st.session_state.df = df
                    # Auto-detect column types
                    numeric_cols, categorical_cols, date_cols = detect_column_types(df)
                    st.session_state.numeric_columns = numeric_cols
                    st.session_state.categorical_columns = categorical_cols
                    st.session_state.date_columns = date_cols
                    st.success("Data loaded successfully from MySQL!")
                else:
                    st.error("Failed to connect to MySQL database. Please check your connection details.")
        else:
            st.warning("Please fill all connection fields")

# Main dashboard content
if st.session_state.df is not None:
    df = st.session_state.df
    numeric_columns = st.session_state.numeric_columns
    categorical_columns = st.session_state.categorical_columns
    date_columns = st.session_state.date_columns
    
    # Display raw data toggle
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(df, use_container_width=True)
    
    # Data info with animated metrics
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Total Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(numeric_columns)}</div>
            <div class="metric-label">Numeric Columns</div>
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Overview", "📊 Visualizations", "🔍 Insights", "⚙️ Custom Analysis", "🤖 Advanced Analytics", "📈 Data Profiling"])
    
    with tab1:
        st.subheader("Dataset Summary")
        
        # Basic statistics for numeric columns
        if numeric_columns:
            st.write("Numeric Columns Summary:")
            st.dataframe(df[numeric_columns].describe(), use_container_width=True)
        
        # Value counts for categorical columns
        if categorical_columns:
            st.write("Categorical Columns Summary:")
            selected_cat_col = st.selectbox("Select categorical column:", categorical_columns, key="tab1_cat_col")
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts().head(10)
                st.bar_chart(value_counts)
                st.write(f"Top 10 values in {selected_cat_col}:")
                st.dataframe(value_counts, use_container_width=True)

    with tab2:
        st.subheader("Data Visualizations")
        
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
                    fig_hist = px.histogram(df, x=selected_num_col, nbins=nbins, 
                                      title=f"Distribution of {selected_num_col}",
                                      color_discrete_sequence=['#FFD700'])
                    fig_hist.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF')
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
                            fig_scatter = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                                   title=f"{y_col} vs {x_col} (colored by {color_col})",
                                                   color_continuous_scale='Viridis')
                        else:
                            fig_scatter = px.scatter(df, x=x_col, y=y_col, 
                                                   title=f"{y_col} vs {x_col}",
                                                   color_discrete_sequence=['#FF4500'])
                        fig_scatter.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF')
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
            
            elif viz_type == "Box Plot":
                # Box plot for numeric columns
                selected_num_col = st.selectbox("Select numeric column for box plot:", numeric_columns, key="box_col")
                if categorical_columns:
                    category_col = st.selectbox("Select category column (optional):", [None] + categorical_columns, key="box_category")
                    if category_col:
                        fig_box = px.box(df, x=category_col, y=selected_num_col,
                                       title=f"Distribution of {selected_num_col} by {category_col}",
                                       color=category_col,
                                       color_discrete_sequence=px.colors.qualitative.Set3)
                    else:
                        fig_box = px.box(df, y=selected_num_col,
                                       title=f"Distribution of {selected_num_col}",
                                       color_discrete_sequence=['#FFD700'])
                    fig_box.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF')
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    fig_box = px.box(df, y=selected_num_col,
                                   title=f"Distribution of {selected_num_col}",
                                   color_discrete_sequence=['#FFD700'])
                    fig_box.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF')
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
                        corr_subset = df[selected_cols]
                        # Ensure we're working with a DataFrame
                        if not isinstance(corr_subset, pd.DataFrame):
                            corr_subset = pd.DataFrame(corr_subset)
                        corr_data = corr_subset.corr()
                        fig_heatmap = px.imshow(corr_data, 
                                              title="Correlation Heatmap",
                                              color_continuous_scale='RdBu_r',
                                              aspect="auto")
                        fig_heatmap.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF')
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
                        fig_line = px.line(df, x=x_col, y=y_col,
                                         title=f"{y_col} over {x_col}",
                                         color_discrete_sequence=['#FFD700'])
                        fig_line.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF')
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
                        fig_area = px.area(df, x=x_col, y=y_col,
                                         title=f"{y_col} over {x_col}",
                                         color_discrete_sequence=['#FF4500'])
                        fig_area.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF')
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
                            fig_3d = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                                                 title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}",
                                                 color_continuous_scale='Viridis')
                        else:
                            fig_3d = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                                                 title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}",
                                                 color_discrete_sequence=['#FFD700'])
                        fig_3d.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF')
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
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF')
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
                                  color_discrete_sequence=['#FFD700'],
                                  marginal="box")
                fig_dist.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF')
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
                           color_discrete_sequence=['#FF4500'])
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF')
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
                    selected_range = st.slider(f"Select range for {col}:", 
                                             min_val, max_val, 
                                             (min_val, max_val))
                    filtered_df = filtered_df[
                        (filtered_df[col] >= selected_range[0]) & 
                        (filtered_df[col] <= selected_range[1])
                    ]
                elif col in categorical_columns:
                    # Categorical filter
                    unique_values = df[col].unique().tolist()
                    selected_values = st.multiselect(f"Select values for {col}:", 
                                                   unique_values, 
                                                   default=unique_values,
                                                   key="cat_filter_values")
                    if selected_values:
                        # Convert to Series if needed
                        col_series = pd.Series(filtered_df[col])
                        mask = col_series.isin(selected_values)
                        filtered_df = filtered_df[mask]
                elif col in date_columns:
                    # Date filter
                    # Convert to Series first to avoid attribute access issues
                    col_series = pd.Series(df[col])
                    min_date = pd.to_datetime(col_series.min())
                    max_date = pd.to_datetime(col_series.max())
                    selected_dates = st.date_input(f"Select date range for {col}:", 
                                                 value=(pd.to_datetime(min_date).date(), pd.to_datetime(max_date).date()))
                    if len(selected_dates) == 2:
                        filtered_df = filtered_df[
                            (filtered_df[col] >= pd.Timestamp(selected_dates[0])) & 
                            (filtered_df[col] <= pd.Timestamp(selected_dates[1]))
                        ]
            
            # Show filtered results
            st.write(f"### Filtered Results ({len(filtered_df)} rows)")
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
                            st.dataframe(numeric_filtered_df.describe(), use_container_width=True)
                    else:
                        # Single column case
                        col = numeric_cols_in_filtered[0]
                        series = filtered_df[col]
                        if pd.api.types.is_numeric_dtype(series):
                            # Convert to Series if it's not already
                            if not isinstance(series, pd.Series):
                                series = pd.Series(series)
                            desc = series.describe()
                            # Convert to DataFrame for display
                            desc_df = pd.DataFrame(desc)
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
                                 color_discrete_sequence=['#FFD700'])
                    fig_group.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF')
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
                                    color_continuous_scale='Viridis'
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
                                    color_continuous_scale='RdBu'
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
                                    color_continuous_scale='Viridis'
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
                                    color_continuous_scale='Viridis'
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
                                       color_discrete_sequence=['#FFD700'])
                        fig_ts.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF')
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)
                        
                        # Moving averages
                        st.write("### Moving Averages")
                        window = st.slider("Select moving average window:", 3, 30, 7)
                        ts_data[f'MA_{window}'] = ts_data[value_col].rolling(window=window).mean()
                        
                        fig_ma = px.line(ts_data, 
                                       title=f"Moving Average (Window: {window})",
                                       color_discrete_sequence=['#FFD700', '#FF4500'])
                        fig_ma.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF')
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
                
                # Data quality metrics
                st.write("### Data Quality Metrics")
                quality_metrics = {
                    'Duplicate_Rows': len(df) - len(df.drop_duplicates()),
                    'Duplicate_Percentage': round((len(df) - len(df.drop_duplicates())) / len(df) * 100, 2),
                    'Unique_Columns': len([col for col in df.columns if df[col].nunique() == len(df)])
                }
                quality_df = pd.DataFrame([quality_metrics]).T
                quality_df.columns = ['Value']
                st.dataframe(quality_df, use_container_width=True)

else:
    st.info("Please upload an Excel file or connect to a MySQL database to begin analysis.")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", 
             caption="Universal Data Analytics Dashboard", use_column_width=True)
    
    st.markdown("""
    ### How to Use This Dashboard
    
    1. **Select Data Source**: Choose between uploading an Excel file or connecting to a MySQL database
    2. **Upload Data**: 
       - For Excel: Upload any Excel file with your data
       - For MySQL: Enter connection details and SQL query
    3. **Automatic Analysis**: The dashboard will automatically detect:
       - Numeric columns
       - Categorical columns
       - Date columns
    4. **Explore**: Use different tabs for various insights:
       - Overview: Basic statistics and data summary
       - Visualizations: Charts and graphs
       - Insights: Correlations and key findings
       - Custom Analysis: Filter and analyze specific subsets
    
    ### Features
    
    - **Universal Compatibility**: Works with ANY dataset structure
    - **Auto Column Detection**: Automatically identifies data types
    - **Interactive Visualizations**: Dynamic charts and graphs
    - **Flexible Filtering**: Analyze specific data subsets
    - **Export Ready**: Download reports and visualizations
    """)

# Run the app
if __name__ == "__main__":
    pass