"""
Verification script for the Employee Performance Analysis Dashboard
This script checks that all required packages are installed and components work correctly
"""

import sys
import importlib

def check_package(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def main():
    print("Verifying Employee Performance Analysis Dashboard Installation...\n")
    
    # List of required packages
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'mysql.connector',
        'openpyxl',
        'sqlalchemy'
    ]
    
    # Check each package
    all_installed = True
    for package in required_packages:
        if not check_package(package):
            all_installed = False
    
    print("\n" + "="*50)
    
    if all_installed:
        print("✓ All required packages are installed!")
        print("\nYou can now run the dashboard with:")
        print("streamlit run app.py")
    else:
        print("✗ Some packages are missing. Please install them with:")
        print("pip install -r requirements.txt")
    
    print("\n" + "="*50)
    print("Testing data files...")
    
    try:
        import pandas as pd
        df = pd.read_excel('employee_data_sample.xlsx')
        print(f"✓ Sample data file loaded successfully ({df.shape[0]} rows)")
    except Exception as e:
        print(f"✗ Error loading sample data: {str(e)}")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main()