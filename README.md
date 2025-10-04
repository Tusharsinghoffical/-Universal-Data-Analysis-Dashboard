# Universal Data Analysis Dashboard

A dynamic, interactive dashboard built with Streamlit for analyzing ANY type of data from various file formats or MySQL databases. The dashboard automatically detects column types and provides relevant visualizations and insights with a premium, luxurious design.

![Dashboard Preview](https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80)

## Features

- 🌟 **Universal Compatibility**: Works with ANY dataset structure
- 📁 **Multi-Format Support**: Excel, CSV, JSON, Parquet, Feather files
- 🤖 **Auto Column Detection**: Automatically identifies numeric, categorical, and date columns
- 📊 **Interactive Visualizations**: Dynamic charts and graphs
- 🔍 **Smart Insights**: Automatic correlation analysis and key findings
- ⚙️ **Flexible Filtering**: Analyze specific data subsets
- 🎨 **Premium UI Design**: Gold accents, gradient effects, glass-morphism
- 🗄️ **MySQL Connectivity**: Connect to any MySQL database
- 📤 **Export Ready**: Download reports and visualizations
- 🤖 **Advanced Analytics**: Machine learning capabilities
- 📈 **Data Profiling**: Comprehensive dataset analysis

## Supported File Formats

- **Excel**: .xlsx, .xls
- **CSV**: .csv
- **JSON**: .json
- **Parquet**: .parquet
- **Feather**: .feather

## Advanced Analytics Features

- **Clustering**: K-Means clustering with silhouette score evaluation
- **Anomaly Detection**: Isolation Forest for outlier detection
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Regression Analysis**: Relationship discovery between variables
- **Data Profiling**: Comprehensive dataset quality analysis
- **Time Series Analysis**: Trend detection and forecasting

## Premium Design Features

- **Gold Accents**: Luxurious color scheme with gold highlights
- **Gradient Effects**: Beautiful gradient backgrounds and elements
- **Glass-Morphism**: Modern frosted glass UI elements
- **Sophisticated Animations**: Smooth transitions and hover effects
- **Consistent Styling**: Uniform design across all dashboard sections
- **Vibrant Color Scheme**: Avoids black-and-white appearance
- **Responsive Layout**: Works on all device sizes

## Enhanced Visualizations

### 📊 Data Visualizations
- **Histograms**: Distribution analysis with customizable bins
- **Scatter Plots**: 2D and 3D scatter plots with color dimensions
- **Box Plots**: Distribution comparison across categories
- **Heatmaps**: Correlation matrices with color coding
- **Line Charts**: Time series and trend visualization
- **Area Charts**: Cumulative data representation
- **Bar Charts**: Category comparison and distribution

### 🔍 Insights
- **Correlation Analysis**: Strong relationship detection
- **Distribution Analysis**: Skewness, kurtosis, and statistical measures
- **Categorical Analysis**: Pie charts and bar graphs
- **Pattern Recognition**: Automated insight generation

### ⚙️ Custom Analysis
- **Multi-Column Filtering**: Complex data filtering
- **Group Analysis**: Aggregation by categories
- **Range Selection**: Numeric and date range filters
- **Statistical Summaries**: Filtered data statistics

### 🤖 Advanced Analytics
- **Clustering**: K-Means with cluster visualization
- **Anomaly Detection**: Outlier identification
- **PCA**: Dimensionality reduction
- **Regression Analysis**: Variable relationship discovery
- **Time Series Analysis**: Trend detection and moving averages

### 📈 Data Profiling
- **Dataset Overview**: Size, memory usage, structure
- **Quality Metrics**: Missing data, duplicates, outliers
- **Pattern Analysis**: Uniqueness ratios and distributions
- **Data Cleaning**: Duplicate removal and data optimization

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- MySQL Connector Python
- SQLAlchemy
- OpenPyXL
- PyArrow
- FastParquet
- Scikit-learn
- Python-dotenv
- SciPy

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Security Best Practices

For database connections, we recommend using environment variables instead of hardcoding credentials:

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Update the `.env` file with your actual database credentials

3. Never commit the `.env` file to version control

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. The dashboard will open in your default web browser.

## Design Consistency

This dashboard follows a comprehensive style guide to ensure visual consistency:

- **Color Scheme**: Gold accents with vibrant gradient backgrounds
- **Typography**: Consistent font hierarchy and styling
- **UI Components**: Uniform card, button, and input designs
- **Animations**: Smooth transitions and hover effects
- **Layout**: Consistent spacing and grid system

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for detailed design specifications.

## How It Works

The dashboard automatically analyzes your dataset and:

1. **Detects Column Types**: Identifies numeric, categorical, and date columns
2. **Provides Relevant Visualizations**: Shows appropriate charts for each data type
3. **Generates Insights**: Calculates correlations and key statistics
4. **Enables Custom Analysis**: Allows filtering and deep dives into specific data
5. **Performs Advanced Analytics**: Applies machine learning techniques
6. **Profiles Data Quality**: Analyzes dataset completeness and structure

## Data Format

Works with ANY file structure. The system automatically detects:

- **Numeric Columns**: For histograms, scatter plots, and statistical analysis
- **Categorical Columns**: For bar charts and value counts
- **Date Columns**: For time series analysis

## MySQL Connection

To connect to a MySQL database:
1. Select "MySQL Database" in the sidebar
2. Enter your database connection details:
   - Host (e.g., localhost or localhost:3306)
   - Username
   - Password
   - Database name
3. Enter a SQL query (default: `SELECT * FROM employees`)
4. Click "Connect to MySQL"

## Dashboard Sections

### Overview
- Basic statistics for numeric columns
- Value counts for categorical columns
- Auto-detected column types

### Visualizations
- Interactive histograms for numeric data
- Scatter plots for relationship analysis
- Box plots for distribution comparison
- Heatmaps for correlation analysis
- Line charts for time series
- Area charts for cumulative data
- 3D scatter plots for multi-dimensional analysis

### Insights
- Correlation matrix heatmap
- Distribution analysis with statistics
- Categorical analysis with pie/bar charts
- Strong relationship detection
- Pattern recognition

### Custom Analysis
- Multi-column filtering with range selection
- Group analysis with aggregation functions
- Statistical summaries for filtered data
- Date range filtering

### Advanced Analytics
- **Clustering**: Group similar data points using K-Means
- **Anomaly Detection**: Identify outliers in your data
- **PCA**: Reduce data dimensions while preserving variance
- **Regression Analysis**: Discover relationships between variables
- **Time Series Analysis**: Trend detection and moving averages

### Data Profiling
- **Dataset Information**: Size, memory usage, structure
- **Column Analysis**: Data types and distributions
- **Missing Data**: Identification and visualization
- **Quality Metrics**: Duplicates, uniqueness, completeness
- **Outlier Analysis**: Statistical outlier detection
- **Data Cleaning**: Duplicate removal tools

## Sample Data Generation

Run `python sample_data.py` to generate a new sample dataset.

## Database Setup

1. Create a MySQL database using the schema in `database_schema.sql`
2. Update connection details in `config.py` as needed

## Utility Functions

The `utils.py` file contains helpful functions for:
- Exporting/importing data to/from various formats
- Exporting/importing data to/from MySQL
- Cleaning and validating data
- Generating performance reports

## Project Structure

```
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── database_schema.sql    # MySQL database schema
├── employee_data_sample.xlsx  # Sample data file
├── requirements.txt       # Python dependencies
├── sample_data.py         # Sample data generator
├── utils.py               # Utility functions
├── test_dashboard.py      # Test script
├── verify_installation.py # Installation verification
├── .env.example          # Environment variables template
├── STYLE_GUIDE.md        # Design consistency guide
├── README.md             # This file
└── PROJECT_SUMMARY.md    # Project overview
```

## Customization

You can customize the dashboard by modifying `app.py`:
- Change color schemes in the CSS section
- Add new visualization types
- Modify metrics calculations
- Extend data source options
- Add new machine learning algorithms

## License

This project is open source and available under the MIT License.