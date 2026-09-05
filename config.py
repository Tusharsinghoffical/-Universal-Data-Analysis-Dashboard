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
    'password': os.getenv('MYSQL_PASSWORD', '1230'),  # Change this to your actual password
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

# AI Configuration (Google Gemini & Heuristic Fallback)
AI_CONFIG = {
    'api_key': os.getenv('GEMINI_API_KEY', ''),
    'model': os.getenv('GEMINI_MODEL', 'gemini-flash-lite-latest'),
    'temperature': float(os.getenv('GEMINI_TEMPERATURE', '0.2')),
    'max_output_tokens': int(os.getenv('GEMINI_MAX_TOKENS', '8192'))
}