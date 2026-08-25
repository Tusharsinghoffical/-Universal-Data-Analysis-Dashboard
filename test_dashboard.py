"""
Test script for the Employee Performance Analysis Dashboard
This script verifies that all components work correctly
"""

import pandas as pd
import streamlit as st
from utils import clean_and_validate_data, generate_performance_report

def test_data_loading():
    """Test loading sample data"""
    try:
        df = pd.read_excel('employee_data_sample.xlsx')
        print(f"✓ Data loading test passed. Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"✗ Data loading test failed: {str(e)}")
        return None

def test_data_cleaning(df):
    """Test data cleaning function"""
    try:
        cleaned_df = clean_and_validate_data(df)
        print(f"✓ Data cleaning test passed. Cleaned data shape: {cleaned_df.shape}")
        return cleaned_df
    except Exception as e:
        print(f"✗ Data cleaning test failed: {str(e)}")
        return None

def test_report_generation(df):
    """Test report generation"""
    try:
        result = generate_performance_report(df, 'test_report.xlsx')
        if result:
            print("✓ Report generation test passed")
        else:
            print("✗ Report generation test failed")
        return result
    except Exception as e:
        print(f"✗ Report generation test failed: {str(e)}")
        return False

def main():
    print("Running Employee Performance Dashboard Tests...\n")
    
    # Test 1: Data loading
    df = test_data_loading()
    if df is None:
        return
    
    # Test 2: Data cleaning
    cleaned_df = test_data_cleaning(df)
    if cleaned_df is None:
        return
    
    # Test 3: Report generation
    test_report_generation(cleaned_df)
    
    print("\nAll tests completed!")
    print("\nTo run the dashboard:")
    print("streamlit run app.py")
    
    print("\nTo generate a performance report:")
    print("python -c \"from utils import generate_performance_report; import pandas as pd; df = pd.read_excel('employee_data_sample.xlsx'); generate_performance_report(df)\"")

if __name__ == "__main__":
    main()