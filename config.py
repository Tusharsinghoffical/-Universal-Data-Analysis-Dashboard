import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database Configuration
# This file contains default configuration settings for database connections

# MySQL Default Configuration (with environment variable support)
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', 'NewStrongPassword123!'),  # Change this to your actual password
    'database': os.getenv('MYSQL_DATABASE', 'employee_db'),
    'port': int(os.getenv('MYSQL_PORT', 3306))
}

# Default SQL Query
DEFAULT_QUERY = "SELECT * FROM employees"

# Excel File Configuration
EXCEL_CONFIG = {
    'default_sheet': 'Sheet1',
    'header_row': 0
}

# Performance Score Configuration
PERFORMANCE_CONFIG = {
    'min_score': 0,
    'max_score': 100,
    'high_performance_threshold': 80,
    'low_performance_threshold': 60
}

# Department Configuration
DEPARTMENTS = [
    'Engineering',
    'Marketing',
    'Sales',
    'HR',
    'Finance',
    'Operations',
    'IT',
    'R&D'
]