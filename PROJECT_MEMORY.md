# Universal Data Analysis Dashboard — Project Memory

> **Auto-updated** by Kiro on every `app.py` save.
> Last updated: 2026-08-25 (MySQL error handling fix)

---

## 1. PRD — Product Requirements Document

### Overview
A self-contained, browser-based data analysis dashboard that lets any user load a dataset and immediately explore it through interactive visualizations and ML-powered analytics — with zero coding required.

### Target Users
- Business analysts and data analysts who need quick EDA on arbitrary datasets.
- Data engineers who want a lightweight SQL query viewer.
- Non-technical stakeholders who need charts and summaries from uploaded files.

### Core Value Propositions
| # | Value |
|---|-------|
| 1 | **Zero-config EDA** — upload a file and get instant stats, charts, and insights |
| 2 | **Universal file support** — Excel, CSV, JSON, Parquet, Feather |
| 3 | **Live database querying** — connect to MySQL and run custom SQL |
| 4 | **ML analytics without code** — clustering, anomaly detection, PCA, regression |
| 5 | **Professional UI** — dark enterprise theme, responsive layout |

### Success Metrics
- Dashboard loads and displays data within 3 seconds for files up to 50 MB.
- All six analysis tabs render without errors for any valid tabular dataset.
- Zero Python knowledge required to operate the dashboard.

---

## 2. Requirements

### Functional Requirements

#### Data Ingestion
- FR-01: Upload files in `.xlsx`, `.xls`, `.csv`, `.json`, `.parquet`, `.feather` formats.
- FR-02: Connect to a MySQL database via host, port, user, password, database, and custom SQL query.
- FR-03: Auto-detect column types (numeric, categorical, datetime) on load.
- FR-04: Cache loaded data in Streamlit session state to avoid re-reads on widget interaction.

#### Analysis Tabs
- FR-05 **Overview** — descriptive statistics (`describe()`) for numeric columns; top-10 value counts for categorical columns.
- FR-06 **Visualizations** — Histogram, Scatter, Box Plot, Heatmap, Line Chart, Area Chart, 3D Scatter; all chart types configurable via dropdowns.
- FR-07 **Insights** — Correlation matrix heatmap; strong-correlation table (|r| > 0.7); per-column distribution stats (mean, median, std, skewness); bar + pie for categorical columns.
- FR-08 **Custom Analysis** — Multi-column filtering (numeric range sliders, categorical multi-select, date range pickers); group-by aggregation with selectable aggregation function.
- FR-09 **Advanced Analytics** — K-Means clustering with silhouette score; Isolation Forest anomaly detection; PCA with explained variance; linear regression between column pairs; time series with moving average and trend (scipy linregress).
- FR-10 **Data Profiling** — row/column count, memory usage, column type breakdown, missing-data % per column, duplicate row count.
- FR-14 **AI Data Intelligence** — Conversational Data Analyst ("Chat with your Data") with real-time computation; Executive Intelligence Briefing with markdown download; Smart Chart suggestions & NL prompt-to-chart rendering; AI Quality & Cleaning Advisor with in-session remediation; Metric Driver & Impact Analyzer with correlation attribution. Powered by Google Gemini (`gemini-3.7-flash`, `gemini-flash-lite-latest`, `gemini-3.5-flash-lite`) with automatic rate-limit failover and offline-capable Intelligent Heuristic Engine fallback.
- **UI Architecture & Fixes (2026-09-05)**:
  - Eliminated raw HTML block wrapping around Markdown content in `st.markdown` which caused unrendered CommonMark tags and trailing `</div>` artifacts.
  - Redesigned Tab 7 into segmented sub-tabs (`st.tabs`) with styled card containers (`st.container(border=True)`) and native `st.chat_message` streams.
  - Implemented representative sampling (25k-50k rows) for statistical profiles, quantiles, and skewness calculations to ensure multi-million row datasets (e.g. 2,740,000 MySQL rows) process in <100ms without timeouts.
  - Fixed `TypeError: Cannot compare Timestamp with datetime.date` in Tab 4 Custom Analysis date filtering by coercing column values via `pd.to_datetime` and comparing against `pd.Timestamp`.
  - Fixed `StreamlitDuplicateElementKey: cat_filter_values` in Tab 4 Custom Analysis by scoping multi-select keys dynamically (`cat_filter_values_{col}`) and added high-cardinality text search protection for text columns.

#### UI / UX
- FR-11: Sidebar holds all data-source and AI configuration controls; main area is exclusively analysis output.
- FR-12: No balloon/confetti animation on file load.
- FR-13: Responsive layout using `use_container_width=True` on all charts and tables.

### Non-Functional Requirements
- NFR-01: Python 3.12 compatible.
- NFR-02: All secrets (DB password) loaded from `.env`; never hard-coded in committed files.
- NFR-03: `requirements.txt` pins every dependency to an exact version.
- NFR-04: App must be startable with a single command: `streamlit run app.py`.

### Out of Scope (v1)
- User authentication / multi-user sessions.
- Write-back to database.
- Scheduled / automated report generation.
- Support for non-tabular data (images, text corpora, audio).

---

## 3. Architecture

### Stack
| Layer | Technology |
|-------|-----------|
| UI Framework | Streamlit 1.62.0 (installed) — `requirements.txt` pins 1.38.0 |
| Data layer | pandas 2.2.2, numpy 1.24.3 |
| Visualizations | Plotly 5.22.0, Matplotlib 3.7.1, Seaborn 0.13.2 |
| ML / Stats | scikit-learn 1.3.0, scipy 1.11.0 |
| DB connector | mysql-connector-python 8.0.33, SQLAlchemy 2.0.30 |
| File formats | openpyxl 3.1.2, pyarrow 14.0.1, fastparquet 2023.10.0 |
| Config | python-dotenv 1.0.0, `config.py` |

### File Structure
```
Universal-Data-Analysis-Dashboard-main/
├── app.py                    # Streamlit application entry point & interactive UI
├── ai_engine.py              # Gemini AI & Heuristic Intelligence Engine
├── config.py                 # MySQL defaults + app constants (reads .env)
├── utils.py                  # Statistical computations, filters & helper functions
├── requirements.txt          # Pinned dependencies
├── .env.example              # Environment variable template
├── .env                      # Local secrets (git-ignored)
├── database_schema.sql       # Reference MySQL schema for employee_db
├── sample_data.py            # Script to generate sample data
├── employee_data_sample.xlsx # Sample dataset for testing
├── PROJECT_MEMORY.md         # ← this file
├── PROJECT_SUMMARY.md        # Original project overview
├── README.md                 # Setup instructions & complete documentation
└── STYLE_GUIDE.md            # Original style reference
```

### Data Flow
```
User Action
    │
    ▼
Sidebar (data source selection)
    │
    ├─ Upload File ──► @st.cache_data loader ──► pandas DataFrame
    │
    └─ MySQL ────────► mysql.connector ────────► pandas.read_sql ──► DataFrame
                                                        │
                                                        ▼
                                             detect_column_types()
                                             (numeric / categorical / date)
                                                        │
                                                        ▼
                                             st.session_state.df
                                                        │
                              ┌─────────────────────────┤
                              ▼                         ▼
                         Tab rendering            Advanced analytics
                         (Plotly charts)          (sklearn / scipy)
```

### Key Design Decisions
- **Single-file app** — all logic lives in `app.py` for simplicity; `config.py` is the only split-out module.
- **Session state caching** — `st.session_state` holds the DataFrame and detected column lists to prevent redundant re-processing on widget interactions. Also holds `mysql_error` string to persist connection errors across rerenders.
- **`@st.cache_data`** on all loaders — file parsing is expensive; caching prevents re-reads when the same file object is passed. MySQL loader returns `(df, error_str)` tuple — never calls `st.error()` inside the cached function.
- **No ORM** — direct `mysql.connector` + `pd.read_sql` keeps the DB layer simple and avoids SQLAlchemy complexity for read-only use.
- **Error isolation** — MySQL errors are stored in `st.session_state.mysql_error` and rendered outside the button callback, preventing stale error messages from re-appearing on every rerender.

---

## 4. UI/UX — v3 Instrument Panel

> Previous versions: v1 gold/glass-morphism (superseded), v2 dark enterprise blue (superseded). See `STYLE_GUIDE.md` for the full v3 spec.

### Design System

#### Color Tokens (CSS custom properties)
| Token | Hex | Usage |
|-------|-----|-------|
| `--bg` | `#0B0E14` | Page background — near-black ink, slightly blue |
| `--surface` | `#12161F` | Cards, tab bar, expanders, sidebar, readout strip |
| `--surface-raised` | `#171C27` | Hovered/active surfaces |
| `--border` | `#232A38` | All 1px borders; chart gridlines |
| `--accent-signal` | `#5EEAD4` | PRIMARY — active tab underline, metric values, primary chart series, focus rings, buttons |
| `--accent-warn` | `#FDBA74` | Anomalies, outliers, idle status dot — never decorative |
| `--accent-secondary` | `#A5B4FC` | Secondary chart series, moving averages |
| `--text-primary` | `#F1F5F9` | Headings, active labels |
| `--text-muted` | `#8B95A7` | Body, axis labels, metadata |

#### Typography
| Role | Font | Notes |
|------|------|-------|
| Display / headings | Space Grotesk 600–700 | Loaded via Google Fonts |
| Body | Inter 400–500 | Fallback: system-ui |
| **Numbers / readouts** | **IBM Plex Mono 400–600** | All numeric values, KPIs, metric cards, buttons, status bar |

Monospace numerics is the defining v3 characteristic — values read like instrument readouts.

#### Components

**Status Bar** — single line above title; IBM Plex Mono 0.72rem uppercase `--text-muted`; 6px blinking dot (`--accent-signal` loaded / `--accent-warn` idle); format: `DATASET LOADED · N ROWS · N COLS · N NUMERIC`.

**Live Readout Strip** — full-width 36px `--surface` bar between title and content; `--border` top/bottom; key stats left, SVG `<polyline>` waveform centre (drawn via `stroke-dashoffset` animation), `SYS READY` right; all in `--accent-signal`. Animation respects `prefers-reduced-motion`.

**Metric Cards** — `--surface` background, 1px `--border`, 4px radius (sharp). Label top: IBM Plex Mono 0.68rem uppercase `--text-muted`. Value bottom: IBM Plex Mono 2rem `--accent-signal`. Hover: border transitions to `--accent-signal` only — no glow, no lift.

**Tabs** — underline style; transparent background; active: `--text-primary` + `2px --accent-signal` bottom border. No pill, no rounded fill.

**Buttons** — transparent background, `1px --accent-signal` border, `--accent-signal` text, IBM Plex Mono uppercase. Hover: fill `--accent-signal`, text `--bg`.

**Column Type Tags** — transparent, `1px --accent-signal` border, IBM Plex Mono 0.75rem, 2px radius inline badges.

**Charts** — `plot_bgcolor` / `paper_bgcolor` transparent (`rgba(11,14,20,0)`); gridlines `--border`; font IBM Plex Mono `--text-muted`; primary series `#5EEAD4`; secondary `#A5B4FC`; anomalies `#FDBA74`; continuous scale `['#12161F','#5EEAD4']`.

#### Layout
- Max content width: 1280px
- Header zone: status bar → title (Space Grotesk) → readout strip → metric row (3 cards) → column-type expander → 6-tab panel
- Sidebar: `--surface`, `1px --border` right edge; data-source controls only

#### Empty / Idle State
- No hero image, no feature-card grid
- Full-width `--surface` panel with amber `NO DATASET LOADED · AWAITING INPUT` status line, plain-language instruction paragraph, format tag row, static SVG waveform at 25% opacity

#### Removed Anti-patterns
- ~~Animated gradient background~~
- ~~Gold/orange palette~~
- ~~Pill/rounded filled active tabs~~
- ~~Blue (#3b82f6) accent~~
- ~~Hero image on welcome screen~~
- ~~2-column feature-card welcome grid~~
- ~~`st.balloons()`~~
- ~~`use_column_width=True`~~

---

## 5. Development Plan

### Completed ✅
- [x] Core dashboard with file upload (Excel, CSV, JSON, Parquet, Feather)
- [x] MySQL database connectivity
- [x] Auto column-type detection
- [x] Six analysis tabs: Overview, Visualizations, Insights, Custom Analysis, Advanced Analytics, Data Profiling
- [x] Advanced ML: K-Means, Isolation Forest, PCA, Linear Regression, Time Series
- [x] Fixed `use_column_width` → `use_container_width` (Streamlit 1.62 compat)
- [x] Removed `st.balloons()` on file upload
- [x] v2: Professional dark enterprise UI (blue accent, flat design)
- [x] v3: Instrument Panel redesign — CSS token system, Space Grotesk + IBM Plex Mono fonts
- [x] v3: Monospace STATUS bar with blinking dot above main title
- [x] v3: Live readout strip with SVG waveform + key stats under title
- [x] v3: Metric cards — label top / value bottom, IBM Plex Mono, `--accent-signal` cyan
- [x] v3: Underline-style tabs (oscilloscope channel selector)
- [x] v3: All 14 Plotly chart layouts updated — signal cyan primary, violet secondary, amber anomalies, dark gridlines
- [x] v3: Idle/empty state rebuilt as instrument panel (no hero image, no feature cards)
- [x] STYLE_GUIDE.md fully rewritten for v3
- [x] Fixed `load_mysql_data` — removed `st.error()` from inside `@st.cache_data` function; now returns `(df, error_str)` tuple
- [x] MySQL error stored in `st.session_state.mysql_error`; displayed once outside button callback — no more repeated error on rerender
- [x] Initialised `mysql_error` key in session state block to prevent `KeyError` on first load
- [x] README.md rewritten — accurate MySQL setup steps (4-step guide with exact commands), troubleshooting section, v3 UI description, removed all v1 gold/glass-morphism references

### In Progress 🔄
- [ ] None currently

### Backlog 📋
- [ ] Export: download filtered data as CSV / Excel from Custom Analysis tab
- [ ] Chart export: save individual Plotly charts as PNG
- [ ] Dark/light theme toggle
- [ ] Column rename / drop utility before analysis
- [ ] Support for multi-sheet Excel files (sheet selector)
- [ ] Pagination for large DataFrames in Raw Data view
- [ ] Unit tests for all analytics functions (`test_dashboard.py` expansion)
- [ ] Docker + `docker-compose.yml` for one-command deployment
- [ ] MySQL connection pooling / timeout handling
- [ ] Progress bar for large file parsing

### Known Issues 🐛
- `requirements.txt` pins Streamlit to 1.38.0 but 1.62.0 is installed — consider updating the pin.
- `config.py` contains a default password string; should always be overridden via `.env`.

---

## Change Log

| Date | Change |
|------|--------|
| 2026-08-25 | Initial dashboard — file upload, MySQL, 6 analysis tabs, ML analytics |
| 2026-08-25 | Fixed `use_column_width` → `use_container_width` (Streamlit 1.62 compat) |
| 2026-08-25 | Removed `st.balloons()` on file upload success |
| 2026-08-25 | v2 UI: dark slate theme, blue accent (#3b82f6), professional color palette, static header, redesigned welcome screen |
| 2026-08-25 | v3 UI: Instrument Panel redesign — CSS token system, Space Grotesk + IBM Plex Mono fonts, STATUS bar, live readout strip, underline tabs, 14 chart layouts, idle empty state, STYLE_GUIDE.md rewritten |
| 2026-08-25 | Fixed MySQL connector — moved `st.error()` out of `@st.cache_data`; `load_mysql_data` now returns `(df, err)` tuple; error stored in `st.session_state.mysql_error` and shown once outside button callback |
| 2026-08-25 | Initialised `mysql_error` in session state block; prevents KeyError on first render |
| 2026-08-25 | README.md rewritten — 4-step MySQL setup guide, troubleshooting section, accurate v3 UI description |
