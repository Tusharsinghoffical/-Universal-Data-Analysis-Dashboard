import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Generate sample employee data
def generate_sample_data():
    # Departments
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations', 'IT', 'R&D']
    
    # Employee names
    first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 
                   'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
                   'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher', 'Nancy', 'Daniel', 'Lisa',
                   'Matthew', 'Betty', 'Anthony', 'Helen', 'Mark', 'Sandra']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                  'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
                  'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 
                  'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson']
    
    # Positions
    positions = ['Manager', 'Developer', 'Analyst', 'Designer', 'Coordinator', 'Specialist', 
                 'Director', 'Associate', 'Consultant', 'Supervisor']
    
    # Generate 500 sample employees
    num_employees = 500
    employee_data = []
    
    for i in range(num_employees):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        employee_name = f"{first_name} {last_name}"
        
        department = random.choice(departments)
        position = random.choice(positions)
        
        # Generate performance score (normally distributed around 75 with std dev of 15)
        performance_score = max(0, min(100, np.random.normal(75, 15)))
        
        # Generate join date (within last 5 years)
        days_ago = random.randint(0, 365*5)
        join_date = datetime.now() - timedelta(days=days_ago)
        
        # Generate years of experience (0-20 years)
        experience_years = min(20, max(0, int(np.random.normal(5, 3))))
        
        # Generate salary based on experience and department
        base_salary = 30000 + (experience_years * 2000)
        department_multiplier = {'Engineering': 1.2, 'Sales': 1.1, 'Finance': 1.15, 
                               'Marketing': 1.0, 'HR': 0.95, 'Operations': 1.05, 
                               'IT': 1.25, 'R&D': 1.3}
        salary = int(base_salary * department_multiplier.get(department, 1.0) * np.random.normal(1, 0.1))
        
        employee_data.append({
            'Employee_ID': i + 1,
            'Employee_Name': employee_name,
            'Department': department,
            'Position': position,
            'Performance_Score': round(performance_score, 2),
            'Join_Date': join_date.strftime('%Y-%m-%d'),
            'Experience_Years': experience_years,
            'Salary': salary
        })
    
    # Create DataFrame
    df = pd.DataFrame(employee_data)
    
    # Save to Excel
    df.to_excel('employee_data_sample.xlsx', index=False)
    print("Sample data generated and saved to 'employee_data_sample.xlsx'")
    return df

if __name__ == "__main__":
    df = generate_sample_data()
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"\nDepartments: {df['Department'].unique()}")
    print(f"\nAverage Performance Score: {df['Performance_Score'].mean():.2f}")