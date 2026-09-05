# <div align="center">

<br/>

```
██╗   ██╗███╗   ██╗██╗██╗   ██╗███████╗██████╗ ███████╗ █████╗ ██╗     
██║   ██║████╗  ██║██║██║   ██║██╔════╝██╔══██╗██╔════╝██╔══██╗██║     
██║   ██║██╔██╗ ██║██║██║   ██║█████╗  ██████╔╝███████╗███████║██║     
██║   ██║██║╚██╗██║██║╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║██╔══██║██║     
╚██████╔╝██║ ╚████║██║ ╚████╔╝ ███████╗██║  ██║███████║██║  ██║███████╗
 ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝
```

<br/>

<h2>Universal Data Analysis & Machine Learning Intelligence Dashboard</h2>

<p><em>No predefined schema required. Connect any MySQL database or upload Excel, CSV, JSON, Parquet, or Feather.<br/>Instant auto-profiling, interactive 3D visualizations, and unsupervised machine learning in seconds.</em></p>

<br/>

![Version](https://img.shields.io/badge/⚡_VERSION-2.5-D4AF37?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly_3D-Express_%26_Objects-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML_Clustering_%26_PCA-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-Live_Connector-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Design](https://img.shields.io/badge/Theme-Luxury_Gold_Glassmorphism-D4AF37?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

<br/>

[![GitHub Repo](https://img.shields.io/badge/📦_GITHUB_REPO-Universal--Data--Analysis--Dashboard-181717?style=for-the-badge&logo=github)](https://github.com/Tusharsinghoffical/Universal-Data-Analysis-Dashboard)
[![Developer](https://img.shields.io/badge/👨‍💻_LEAD_ARCHITECT-Tushar_Singh-6366F1?style=for-the-badge)](https://codewithmrsingh.me/)

<br/>

<div align="center">

### ╔═══════════════════════════════════════════════════════════════════════════════════╗
### ║  Navigate:  [Overview](#-overview)  •  [Formats](#-supported-data-sources)  •  [Capabilities](#-core-capabilities)  •  [ML Engine](#-advanced-machine-learning-suite)  •  [Architecture](#-data-pipeline-architecture)  •  [Quick Start](#-quick-start) ║
### ╚═══════════════════════════════════════════════════════════════════════════════════╝

</div>

---

## 🎯 Overview

> **Universal Data Analysis Dashboard** is a zero-configuration, production-grade exploratory data analysis (EDA) and unsupervised machine learning platform built on Streamlit, Plotly 3D, Scikit-Learn, and Pandas.

Most business intelligence dashboards require fragile data contracts, fixed SQL queries, or manual ETL schemas before a user can inspect a single chart. The Universal Data Analysis Dashboard eliminates schema lock-in: **it ingests ANY tabular dataset, auto-detects column data types, executes statistical profiling, and generates interactive visualizations and machine learning models on the fly.**

- 🌟 **Universal Schema Agnostic**: Ingests messy datasets without manual mapping.
- 📁 **Multi-Format Ingestion**: Excel (`.xlsx`, `.xls`), CSV (`.csv`), JSON (`.json`), Parquet (`.parquet`), Feather (`.feather`), or direct live MySQL database pools.
- 🤖 **Automated Data Type Inference**: Instantly distinguishes between continuous numeric, discrete ordinal, categorical, datetime, and free text variables.
- 📈 **Data Quality & Health Profiling**: Identifies missing values, duplicate records, outliers, skewness, and memory footprint.
- 🔬 **Built-in Machine Learning**: K-Means clustering with silhouette scoring, Isolation Forest anomaly detection, and PCA dimensionality reduction.
- 🎨 **Executive Luxury Design**: Ultra-premium frosted glassmorphism interface accented with executive gold (`#D4AF37`).

---

## 🔥 The Problem & The Solution

<div align="center">

```
TRADITIONAL BI TOOLS (TABLEAU / EXCEL)       UNIVERSAL DATA DASHBOARD
──────────────────────────────────────       ───────────────────────────────────
❌ Rigid Schema Lock-In                      ✅ 100% Universal Dataset Compatibility
   (Breaks when columns change or shift)        (Detects and conforms to ANY file structure)

❌ Limited to Basic Flat Files               ✅ Multi-Format + Direct MySQL Ingestion
   (Tedious import / export conversions)        (Excel, CSV, JSON, Parquet, Feather, MySQL)

❌ Manual Statistical Profiling              ✅ Instant Automated Health Profiling
   (Hours spent computing IQR, skew, missing)   (Sub-second quality audit & memory analysis)

❌ External ML Toolchains Needed             ✅ Native Scikit-Learn Machine Learning
   (Must export data to Jupyter/Python)         (1-click K-Means, Isolation Forest, 3D PCA)

❌ Slow, Clunky Legacy UI                    ✅ Luxury Gold Frosted Glassmorphism
   (Boring grey grids and static charts)        (GPU-accelerated Plotly 3D interactive HUD)
```

</div>

---

## 📁 Supported Data Sources

The dashboard seamlessly ingests and unifies diverse enterprise storage formats:

| Format | File Extensions | Read Engine | Typical Enterprise Use Case |
|:-------|:----------------|:------------|:----------------------------|
| **Excel Workbooks** | `.xlsx`, `.xls` | `openpyxl` / `xlrd` | Corporate spreadsheets, financial reports, operational rosters |
| **Delimited Text** | `.csv`, `.tsv`, `.txt` | `pandas.read_csv` | Machine exports, web analytics dumps, sensor logs |
| **Hierarchical JSON** | `.json` | `json.loads` + normalize | REST API responses, NoSQL database document exports |
| **Apache Parquet** | `.parquet` | `pyarrow.parquet` | Big data lakehouses, high-performance compressed columnar data |
| **Apache Arrow Feather**| `.feather` | `pyarrow.feather` | Ultra-fast memory-mapped IPC data science exchanges |
| **Relational MySQL** | Native Connection | `mysql-connector-python` | Live transactional enterprise databases (`SELECT * FROM table`) |

---

## 🚀 Core Capabilities

### `01` 🤖 Auto Column Detection & Data Typing
Upon ingestion, the platform scans columns and categorizes them automatically:
```
Continuous Numeric  ──► Float / Int (e.g. Salary, Revenue, Age, Duration)
Categorical / Class ──► Object / String / Category (e.g. Department, Status, City)
Temporal Datetime   ──► Timestamps / Dates (e.g. Created_At, Transaction_Date)
Free Text / Hash    ──► High-cardinality strings (e.g. UUID, Comments, Descriptions)
```
Interactive dropdowns across all visualization tabs dynamically adapt to show only relevant column types, preventing chart errors.

---

### `02` 📊 Dynamic Interactive Visualizations (Plotly Express & Objects)
Every chart is fully interactive — supporting zoom, pan, hover tooltips, and SVG/PNG image export:

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│  📊 Distribution Analytics           │  🔗 Relational & Correlation         │
├──────────────────────────────────────┼──────────────────────────────────────┤
│  • Histograms with dynamic bin width │  • 2D Scatter with Trendlines        │
│  • Box & Whisker with Outlier Dots   │  • 3D Scatter with Color Dimensions  │
│  • Violin Plots with Kernel Density  │  • Pearson Correlation Heatmap Matrix│
│  • Cumulative Area & Step Charts     │  • Categorical Group Sunburst Charts │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

### `03` 📈 Data Quality & Profiling Suite
Instant audit of dataset health before analytics or modeling:
- **Missing Value Matrix**: Percentage and count of missing records per column with heatmap visualization.
- **Duplicate Record Detector**: Identifies and filters duplicate rows with one-click deduplication.
- **Statistical Moments**: Calculates Mean, Median, Mode, Standard Deviation, Variance, Skewness, and Kurtosis.
- **Memory Consumption**: Displays memory breakdown per column type with optimization recommendations.

---

### `04` 🎛️ Multi-Dimensional Interactive Filters
Drill down into complex subsets without writing SQL:
- **Numerical Sliders**: Range-based boundaries (e.g. `Revenue: $50,000 – $250,000`).
- **Multi-Select Categorical Filters**: Filter by one or more categories simultaneously.
- **Date Range Pickers**: Slice data within temporal boundaries.
- **Synchronous Recalculation**: All visual tabs instantly recalculate based on the active filtered subset.

---

## 🔬 Advanced Machine Learning Suite

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNSUPERVISED MACHINE LEARNING ENGINE                     │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│  1. K-Means Clustering   │  2. Isolation Forest     │  3. PCA 3D Projection │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│  • Automated K Selection │  • Contamination Factor  │  • Eigenvalue Variance│
│  • Silhouette Score      │  • Outlier Isolation     │  • 2D / 3D Projection │
│  • Cluster Scatter Plot  │  • Anomaly Highlighting  │  • Dimensionality Red.│
└──────────────────────────┴──────────────────────────┴───────────────────────┘
```

</div>

### 1. K-Means Clustering & Silhouette Evaluation
- Select arbitrary numeric feature combinations (e.g. `Income vs Spending vs Age`).
- Configurable cluster count ($K = 2 \text{ to } 10$).
- Automated **Silhouette Score** calculation to evaluate cluster separation quality.
- Interactive cluster assignment scatter plots with distinct cluster centroid markers.

### 2. Isolation Forest Anomaly Detection
- Unsupervised anomaly detection algorithm isolating multi-variable outliers.
- Configurable **Contamination Rate** slider (1% to 15%).
- Outliers highlighted with high-contrast red warning indicators on charts.

### 3. Principal Component Analysis (PCA)
- Reduces high-dimensional datasets down to the 2 or 3 principal components explaining maximum variance.
- Interactive 3D Plotly rotation allowing inspection of latent patterns and data geometry.

---

## 🧠 AI Data Intelligence Suite

Universal Data Analysis Dashboard features a built-in AI intelligence layer powered by **Google Gemini** (`gemini-2.5-flash` / `gemini-1.5-flash`) with an automatic offline-capable **Intelligent Heuristic Analytics Engine** fallback:

### 1. 💬 Conversational Data Analyst ("Chat with your Data")
- Converse with any dataset in natural language.
- Compute real-time aggregations, rank categorical segments, identify maximums/minimums, and cross-reference variables.
- Powered by Google Gemini with an optimized schema/statistics digest, returning factual, data-grounded answers.

### 2. 📋 Executive Intelligence Briefing
- Single-click automated generation of C-suite briefings:
  - **Executive Synopsis**: High-level dataset perimeter, scope, and health metrics.
  - **Key Discoveries**: Quantified statistical patterns, dominant drivers, and categorical rankings.
  - **Anomalies & Risks**: Critical missing data flags, skewness irregularities, and duplicate records.
  - **Strategic Recommendations**: Concrete next steps for analysts and business stakeholders.
- One-click export and download as Markdown (`.md`).

### 3. 🎨 Smart Visualization & NL Chart Generator
- Auto-suggests the top 4 visualizations tailored to the loaded dataset's column topology.
- Convert plain text requests (e.g., *"Box plot of Salary by Department"* or *"Scatter plot of Experience vs Performance Score"*) directly into interactive Plotly charts.

### 4. 🧹 AI Data Quality & Cleaning Advisor
- Automated diagnostic audit evaluating missing rates, duplicate records, and IQR-based outliers.
- Actionable 1-click in-session remediation buttons (e.g. *Drop Duplicates*, *Impute Numeric Missing*, *Cap Outliers*) that immediately clean the working session dataset across all tabs.

### 5. 🔮 Metric Driver & Impact Analyzer
- Identifies the strongest statistical correlates and categorical variance spreads for any target variable.
- Provides plain-English driver attribution highlighting which factors move the metric most.

---

## ⚡ Data Pipeline Architecture

<div align="center">

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                        DATA INGESTION LAYER                      │
   │   Excel (.xlsx) · CSV · JSON · Parquet · Feather · MySQL DB       │
   └────────────────────────────────┬─────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼─────────────────────────────────┐
   │               AUTOMATIC COLUMN TYPE INFERENCE ENGINE             │
   │    Numeric Types · Categorical Strings · Datetimes · Text        │
   └────────────────────────────────┬─────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
 ┌──────▼─────────────┐      ┌──────▼─────────────┐      ┌──────▼─────────────┐
 │  Data Profiling &  │      │  Dynamic Plotly    │      │  Scikit-Learn ML   │
 │  Quality Audit     │      │  Visualization     │      │  Analytics Engine  │
 ├────────────────────┤      ├────────────────────┤      ├────────────────────┤
 │ • Missing Values   │      │ • 2D / 3D Scatter  │      │ • K-Means (K=2-10) │
 │ • Duplicates Check │      │ • Heatmap Matrix   │      │ • Silhouette Score │
 │ • Skewness & IQR   │      │ • Histograms & Box │      │ • Isolation Forest │
 │ • Memory Footprint │      │ • Area & Sunburst  │      │ • 3D PCA Projection│
 └────────────────────┘      └────────────────────┘      └────────────────────┘
                                    │
   ┌────────────────────────────────▼─────────────────────────────────┐
   │                    EXPORT & REPORT GENERATION                    │
   │          Download Processed Excel · Filtered CSV · Charts        │
   └──────────────────────────────────────────────────────────────────┘
```

</div>

---

## 🛠️ Technology Stack

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   UNIVERSAL DASHBOARD TECH SPECIFICATION                    │
├──────────────────────────┬──────────────────────────────────────────────────┤
│  Frontend & App Runtime  │  Streamlit 1.30+ with Custom Glassmorphism CSS   │
│  Data Processing Engine  │  Pandas 2.0+ & NumPy                             │
│  Interactive Charting    │  Plotly Express & Plotly Graph Objects (2D & 3D) │
│  Machine Learning        │  Scikit-Learn (KMeans, IsolationForest, PCA)     │
│  Relational Database     │  MySQL Connector Python (Custom Query Pooling)   │
│  Spreadsheet Engine      │  OpenPyXL & XLRD (Multi-Sheet Excel Support)     │
│  Columnar Big Data       │  PyArrow (Apache Parquet & Apache Arrow Feather) │
│  Statistical Plots       │  Seaborn & Matplotlib Core                       │
│  UI Color Aesthetics     │  Executive Luxury Gold (#D4AF37) Glassmorphism   │
└──────────────────────────┴──────────────────────────────────────────────────┘
```

</div>

---

## 📁 Project Structure

```
Universal-Data-Analysis-Dashboard-main/
│
├── 📄 app.py                         🚀 Main Streamlit application & interactive UI
├── 📄 ai_engine.py                   🧠 Gemini AI & Heuristic Intelligence Engine
├── 📄 config.py                      ⚙️ MySQL database configuration & defaults
├── 📄 utils.py                       🛠️ Statistical computations, filters & helper functions
├── 📄 database_schema.sql            🗄️ MySQL database schema & sample table seeder
├── 📄 requirements.txt               📋 Locked Python dependencies
├── 📄 employee_data_sample.xlsx      📊 Pre-configured enterprise sample dataset
├── 📄 sample_data.py                 🧪 Synthetic dataset generator script
├── 📄 STYLE_GUIDE.md                 🎨 Design consistency & gold glassmorphism tokens
├── 📄 PROJECT_SUMMARY.md             📝 High-level functional specification
├── 📄 PROJECT_MEMORY.md              🧠 Architecture notes & feature milestones
└── 📄 .env.example                   🔐 Environment variables template
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python ≥ 3.10 (Python 3.11 or 3.12 recommended)
Git
MySQL Server (Optional — only if using live MySQL database connection)
```

### 1 · Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/Tusharsinghoffical/Universal-Data-Analysis-Dashboard.git
cd Universal-Data-Analysis-Dashboard-main

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 2 · Install Dependencies
```bash
pip install -r requirements.txt
```

### 3 · (Optional) Configure Environment & Database
Create a `.env` file from the template:
```bash
copy .env.example .env     # Windows
cp .env.example .env       # Linux / macOS
```
Edit `.env` with your Gemini API key and MySQL credentials:
```env
GEMINI_API_KEY=your_gemini_api_key_here
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_actual_password
MYSQL_DATABASE=employee_db
MYSQL_PORT=3306
```
*(You can also seed a sample database using `mysql -u root -p < database_schema.sql`)*

### 4 · Launch Streamlit Dashboard
```bash
streamlit run app.py
```
The application will open automatically in your browser at: `http://localhost:8501`

---

## 💡 How to Use

1. **Choose Data Source in Sidebar**:
   - **Upload File**: Drag-and-drop any `.xlsx`, `.csv`, `.json`, `.parquet`, or `.feather` file.
   - **MySQL Database**: Enter your database host, port, credentials, and custom SQL query.
   - **Sample Data**: Click **Load Sample Data** to explore pre-loaded enterprise HR data immediately.
2. **Review Data Overview**:
   - Inspect dataset shape, column datatypes, and first 10 rows.
3. **Explore Visualizations**:
   - Switch between **Distributions**, **Relationships**, **Categorical**, and **Time Series** tabs.
4. **Run Machine Learning**:
   - Open **Advanced Analytics** to trigger K-Means clustering, Outlier isolation, or 3D PCA.
5. **Export Results**:
   - Click **Download Cleaned Dataset** or export interactive charts as high-resolution images.

---

## 📬 Contact & Support

<div align="center">

| Channel | Details |
|:--------|:--------|
| 👨‍💻 **Lead Architect** | [Tushar Singh](https://codewithmrsingh.me/) — `codewithmrsingh.me` |
| 📧 **Contact Email** | [tusharsingh.dev@gmail.com](mailto:tusharsingh.dev@gmail.com) |
| 📦 **GitHub Repository** | [github.com/Tusharsinghoffical/Universal-Data-Analysis-Dashboard](https://github.com/Tusharsinghoffical/Universal-Data-Analysis-Dashboard) |
| 🏢 **Headquarters** | Delhi / Pune, India |

</div>

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for complete terms.

---

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   🌟  UNIVERSAL DATA DASHBOARD  v2.5 — Schema-Free BI Platform  │
│                                                                 │
│   Crafted with ❤️ by Tushar Singh (codewithmrsingh.me)          │
│   Delhi / Pune, India  ·  AI & Machine Learning Engineering     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

</div>
