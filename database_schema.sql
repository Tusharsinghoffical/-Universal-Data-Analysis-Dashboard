-- Employee Performance Database Schema

-- Create database
CREATE DATABASE IF NOT EXISTS employee_db;
USE employee_db;

-- Create employees table
CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_name VARCHAR(100) NOT NULL,
    department VARCHAR(50) NOT NULL,
    position VARCHAR(50),
    performance_score DECIMAL(5,2),
    join_date DATE,
    experience_years INT,
    salary DECIMAL(10,2)
);

-- Insert sample data
INSERT INTO employees (employee_name, department, position, performance_score, join_date, experience_years, salary) VALUES
('John Smith', 'Engineering', 'Software Engineer', 85.50, '2020-03-15', 3, 75000),
('Mary Johnson', 'Marketing', 'Marketing Manager', 92.25, '2019-07-01', 5, 82000),
('Robert Brown', 'Sales', 'Sales Representative', 78.00, '2021-01-10', 2, 55000),
('Linda Davis', 'HR', 'HR Specialist', 88.75, '2018-11-20', 6, 68000),
('Michael Wilson', 'Finance', 'Financial Analyst', 91.30, '2020-05-30', 4, 72000),
('Jennifer Taylor', 'Operations', 'Operations Manager', 87.60, '2019-02-14', 5, 78000),
('David Anderson', 'IT', 'IT Specialist', 83.40, '2021-08-22', 2, 65000),
('Susan Thomas', 'R&D', 'Research Scientist', 89.90, '2018-04-05', 7, 92000),
('James Jackson', 'Engineering', 'Senior Engineer', 94.20, '2017-09-12', 8, 95000),
('Patricia White', 'Marketing', 'Content Creator', 80.15, '2022-01-18', 1, 50000);

-- Create performance_reviews table
CREATE TABLE performance_reviews (
    review_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT,
    review_date DATE,
    reviewer_name VARCHAR(100),
    comments TEXT,
    goals TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

-- Create departments table
CREATE TABLE departments (
    department_id INT PRIMARY KEY AUTO_INCREMENT,
    department_name VARCHAR(50) UNIQUE NOT NULL,
    manager_name VARCHAR(100),
    budget DECIMAL(12,2)
);

-- Insert department data
INSERT INTO departments (department_name, manager_name, budget) VALUES
('Engineering', 'James Jackson', 2000000),
('Marketing', 'Mary Johnson', 500000),
('Sales', 'Robert Brown', 750000),
('HR', 'Linda Davis', 300000),
('Finance', 'Michael Wilson', 400000),
('Operations', 'Jennifer Taylor', 600000),
('IT', 'David Anderson', 350000),
('R&D', 'Susan Thomas', 1500000);

-- Sample queries for the dashboard
-- 1. Get all employees with department info
SELECT e.*, d.manager_name, d.budget 
FROM employees e 
JOIN departments d ON e.department = d.department_name;

-- 2. Get average performance by department
SELECT department, AVG(performance_score) as avg_performance, COUNT(*) as employee_count
FROM employees 
GROUP BY department;

-- 3. Get top performers
SELECT * FROM employees 
ORDER BY performance_score DESC 
LIMIT 10;