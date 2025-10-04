import pandas as pd
import mysql.connector
from config import MYSQL_CONFIG
import warnings
warnings.filterwarnings('ignore')

def export_dataframe_to_excel(df, filename):
    """
    Export a DataFrame to Excel file
    """
    try:
        df.to_excel(filename, index=False)
        print(f"Data exported successfully to {filename}")
        return True
    except Exception as e:
        print(f"Error exporting to Excel: {str(e)}")
        return False

def import_excel_to_dataframe(filename):
    """
    Import data from Excel file to DataFrame
    """
    try:
        df = pd.read_excel(filename)
        print(f"Data imported successfully from {filename}")
        return df
    except Exception as e:
        print(f"Error importing from Excel: {str(e)}")
        return None

def export_dataframe_to_mysql(df, table_name, if_exists='replace'):
    """
    Export DataFrame to MySQL database
    """
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        
        # Create table if not exists
        columns = ', '.join([f"{col} VARCHAR(255)" for col in df.columns])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        cursor.execute(create_table_query)
        
        # Export data
        for _, row in df.iterrows():
            placeholders = ', '.join(['%s'] * len(row))
            columns_str = ', '.join(df.columns)
            insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            cursor.execute(insert_query, tuple(row))
        
        connection.commit()
        cursor.close()
        connection.close()
        print(f"Data exported successfully to MySQL table '{table_name}'")
        return True
    except Exception as e:
        print(f"Error exporting to MySQL: {str(e)}")
        return False

def import_mysql_to_dataframe(query):
    """
    Import data from MySQL database to DataFrame
    """
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        df = pd.read_sql(query, connection)
        connection.close()
        print("Data imported successfully from MySQL")
        return df
    except Exception as e:
        print(f"Error importing from MySQL: {str(e)}")
        return None

def clean_and_validate_data(df):
    """
    Clean and validate the employee data
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values for different possible column names
    fill_values = {}
    
    # Performance score
    if 'Performance_Score' in df.columns:
        fill_values['Performance_Score'] = df['Performance_Score'].mean()
    elif 'performance_score' in df.columns:
        fill_values['performance_score'] = df['performance_score'].mean()
    
    # Experience years
    if 'Experience_Years' in df.columns:
        fill_values['Experience_Years'] = 0
    elif 'experience_years' in df.columns:
        fill_values['experience_years'] = 0
    
    # Salary
    if 'Salary' in df.columns:
        fill_values['Salary'] = df['Salary'].median()
    elif 'salary' in df.columns:
        fill_values['salary'] = df['salary'].median()
    
    df = df.fillna(fill_values)
    
    # Ensure data types
    if 'Performance_Score' in df.columns:
        df['Performance_Score'] = pd.to_numeric(df['Performance_Score'], errors='coerce')
    elif 'performance_score' in df.columns:
        df['performance_score'] = pd.to_numeric(df['performance_score'], errors='coerce')
    
    if 'Experience_Years' in df.columns:
        df['Experience_Years'] = pd.to_numeric(df['Experience_Years'], errors='coerce')
    elif 'experience_years' in df.columns:
        df['experience_years'] = pd.to_numeric(df['experience_years'], errors='coerce')
    
    if 'Salary' in df.columns:
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    elif 'salary' in df.columns:
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
    
    if 'Join_Date' in df.columns:
        df['Join_Date'] = pd.to_datetime(df['Join_Date'], errors='coerce')
    elif 'join_date' in df.columns:
        df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce')
    
    return df

def generate_performance_report(df, output_filename='performance_report.xlsx'):
    """
    Generate a comprehensive performance report
    """
    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # Determine column names (handle both cases)
            performance_col = 'Performance_Score' if 'Performance_Score' in df.columns else 'performance_score' if 'performance_score' in df.columns else None
            department_col = 'Department' if 'Department' in df.columns else 'department' if 'department' in df.columns else None
            salary_col = 'Salary' if 'Salary' in df.columns else 'salary' if 'salary' in df.columns else None
            experience_col = 'Experience_Years' if 'Experience_Years' in df.columns else 'experience_years' if 'experience_years' in df.columns else None
            
            # Overall summary
            summary_data = {
                'Metric': ['Total Employees', 'Average Performance Score', 'Highest Score', 
                          'Lowest Score', 'Departments', 'Average Salary']
            }
            
            # Fill values based on available columns
            values = [str(len(df))]
            if performance_col:
                values.extend([str(round(df[performance_col].mean(), 2)), str(df[performance_col].max()), str(df[performance_col].min())])
            else:
                values.extend(['0', '0', '0'])
                
            if department_col:
                values.append(str(df[department_col].nunique()))
            else:
                values.append('0')
                
            if salary_col:
                values.append(str(round(df[salary_col].mean(), 2)))
            else:
                values.append('0')
                
            summary_data['Value'] = values
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Department analysis (only if department column exists)
            if department_col and performance_col and salary_col and experience_col:
                dept_analysis = df.groupby(department_col).agg({
                    performance_col: ['mean', 'count'],
                    salary_col: 'mean',
                    experience_col: 'mean'
                }).round(2)
                dept_analysis.columns = ['Avg_Performance', 'Employee_Count', 'Avg_Salary', 'Avg_Experience']
                dept_analysis.to_excel(writer, sheet_name='Department_Analysis')
            
            # Top performers (only if performance column exists)
            if performance_col:
                top_performers = df.nlargest(20, performance_col)
                top_performers.to_excel(writer, sheet_name='Top_Performers', index=False)
            
            # Performance distribution (only if performance column exists)
            if performance_col:
                performance_dist = df[performance_col].value_counts(bins=10).sort_index()
                performance_dist_df = pd.DataFrame({
                    'Score_Range': performance_dist.index.astype(str),
                    'Count': performance_dist.values
                })
                performance_dist_df.to_excel(writer, sheet_name='Performance_Distribution', index=False)
        
        print(f"Performance report generated: {output_filename}")
        return True
    except Exception as e:
        print(f"Error generating performance report: {str(e)}")
        return False

if __name__ == "__main__":
    print("Utility functions for Employee Performance Analysis Dashboard")
    print("Available functions:")
    print("- export_dataframe_to_excel(df, filename)")
    print("- import_excel_to_dataframe(filename)")
    print("- export_dataframe_to_mysql(df, table_name)")
    print("- import_mysql_to_dataframe(query)")
    print("- clean_and_validate_data(df)")
    print("- generate_performance_report(df, output_filename)")