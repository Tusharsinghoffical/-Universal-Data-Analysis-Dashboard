# Universal Data Analysis Dashboard

A single-file Streamlit application for exploring any tabular dataset through interactive visualizations and ML-powered analytics — no coding required.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**

---

## MySQL Setup (required for database mode)

The most common error — `Unknown database 'employee_db'` — means the database hasn't been created yet. Run these steps once before connecting.

### Step 1 — Log in to MySQL

```bash
mysql -u root -p
# enter your password when prompted
```

### Step 2 — Create the database and tables

```sql
CREATE DATABASE IF NOT EXISTS employee_db;
```

Then exit MySQL and run the full schema:

```bash
mysql -u root -p < database_schema.sql
```

This creates the `employee_db` database, the `employees` / `departments` / `performance_reviews` tables, and inserts 100,000 sample rows automatically.

> **Note:** The bulk insert procedure generates 100 K rows and may take 30–60 seconds depending on your machine. Reduce the number at the bottom of `database_schema.sql` if you want a smaller dataset.

### Step 3 — Configure credentials

Copy the environment template and fill in your password:

```bash
cp .env.example .env
```

`.env` (edit this file):

```
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=1230
MYSQL_DATABASE=employee_db
MYSQL_PORT=3306
```

The app reads this file automatically on startup. The password default in `config.py` is also set to `1230` — change both if your MySQL password is different.

### Step 4 — Connect in the dashboard

1. Select **MySQL Database** in the sidebar
2. Confirm the pre-filled host / username / database
3. Enter your password
4. Keep the default query `SELECT * FROM employees` or write your own
5. Click **Connect to MySQL**

---

## File Upload (no database needed)

Select **Upload File** in the sidebar and drop in any of these formats:

| Format | Extension |
|--------|-----------|
| Excel | `.xlsx`, `.xls` |
| CSV | `.csv` |
| JSON | `.json` |
| Parquet | `.parquet` |
| Feather | `.feather` |

A ready-made sample file is included: `employee_data_sample.xlsx`

---

## Dashboard Sections

Once data is loaded, six analysis tabs activate:

| Tab | What it does |
|-----|-------------|
| **Overview** | Descriptive stats for numeric columns; value counts for categorical columns |
| **Visualizations** | Histogram, Scatter, Box Plot, Heatmap, Line, Area, 3D Scatter — all configurable |
| **Insights** | Correlation matrix; strong-correlation table (|r| > 0.7); distribution stats; bar + pie for categoricals |
| **Custom Analysis** | Multi-column filtering (range sliders, multi-select, date pickers); group-by aggregation |
| **Advanced Analytics** | K-Means clustering · Isolation Forest anomaly detection · PCA · Linear regression · Time series with trend |
| **Data Profiling** | Row/column count, memory usage, missing data %, duplicate count |

---

## Project Structure

```
├── app.py                     # Streamlit application (single entry point)
├── config.py                  # MySQL defaults — reads from .env
├── database_schema.sql        # Creates employee_db + inserts sample data
├── sample_data.py             # Generates employee_data_sample.xlsx
├── employee_data_sample.xlsx  # Ready-to-use sample file
├── requirements.txt           # Pinned dependencies
├── utils.py                   # Export / import / cleaning helpers
├── test_dashboard.py          # Test suite
├── verify_installation.py     # Checks all dependencies are installed
├── .env.example               # Credential template — copy to .env
├── .env                       # Your local credentials (never commit this)
├── STYLE_GUIDE.md             # v3 Instrument Panel design specification
├── PROJECT_MEMORY.md          # Living project doc (PRD, architecture, UI/UX, dev plan)
└── PROJECT_SUMMARY.md         # Original project overview
```

---

## Requirements

- Python 3.10+
- MySQL 8.0+ (only needed for database mode)

All Python dependencies are in `requirements.txt`. Key packages:

| Package | Purpose |
|---------|---------|
| streamlit 1.38+ | UI framework |
| pandas | Data loading and manipulation |
| plotly | Interactive charts |
| scikit-learn | Clustering, PCA, anomaly detection, regression |
| scipy | Statistical analysis, time series trend |
| mysql-connector-python | MySQL connectivity |
| openpyxl / pyarrow | Excel / Parquet file support |
| python-dotenv | `.env` credential loading |

---

## Troubleshooting

### `Unknown database 'employee_db'`
The database does not exist yet. Run `database_schema.sql` as shown in **MySQL Setup → Step 2** above.

### `Access denied for user 'root'`
Wrong password. Update `MYSQL_PASSWORD` in your `.env` file to match your MySQL root password.

### `Can't connect to MySQL server`
MySQL service is not running. Start it:
- **Windows:** `net start MySQL80` (or open Services and start MySQL)
- **macOS:** `brew services start mysql`
- **Linux:** `sudo systemctl start mysql`

### `ModuleNotFoundError`
Run `pip install -r requirements.txt` again. If a specific package fails, install it individually: `pip install <package-name>`.

### Charts not rendering / blank tabs
The dataset needs at least one numeric column for most visualizations. Check the **Auto-detected Column Types** expander on the main page to see what was detected.

### `use_column_width` error
Ensure Streamlit is version 1.38 or newer: `pip install --upgrade streamlit`

---

## Security

- Never commit `.env` to version control — it is listed in `.gitignore`
- Always set `MYSQL_PASSWORD` via `.env`, not by editing `config.py` directly
- The dashboard is read-only — it does not write back to the database

---

## License

MIT License — open source, free to use and modify.
