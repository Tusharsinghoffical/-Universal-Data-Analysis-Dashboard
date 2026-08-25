-- =========================================
-- 1. DATABASE SETUP
-- =========================================
CREATE DATABASE IF NOT EXISTS employee_db;
USE employee_db;

-- =========================================
-- 2. TABLES
-- =========================================

CREATE TABLE IF NOT EXISTS employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_name VARCHAR(100) NOT NULL,
    department VARCHAR(50) NOT NULL,
    position VARCHAR(50),
    performance_score DECIMAL(5,2),
    join_date DATE,
    experience_years INT,
    salary DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS performance_reviews (
    review_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT,
    review_date DATE,
    reviewer_name VARCHAR(100),
    comments TEXT,
    goals TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE IF NOT EXISTS departments (
    department_id INT PRIMARY KEY AUTO_INCREMENT,
    department_name VARCHAR(50) UNIQUE NOT NULL,
    manager_name VARCHAR(100),
    budget DECIMAL(12,2)
);

-- =========================================
-- 3. INSERT DEPARTMENTS
-- =========================================
INSERT INTO departments (department_name, manager_name, budget) VALUES
('Engineering', 'James Jackson', 2000000),
('Marketing', 'Mary Johnson', 500000),
('Sales', 'Robert Brown', 750000),
('HR', 'Linda Davis', 300000),
('Finance', 'Michael Wilson', 400000),
('Operations', 'Jennifer Taylor', 600000),
('IT', 'David Anderson', 350000),
('R&D', 'Susan Thomas', 1500000);

-- =========================================
-- 4. PERFORMANCE OPTIMIZATION SETTINGS
-- =========================================
SET autocommit = 0;
SET unique_checks = 0;
SET foreign_key_checks = 0;

-- =========================================
-- 5. RANDOM NAME FUNCTION
-- =========================================
DELIMITER $$

CREATE FUNCTION random_name() RETURNS VARCHAR(100)
DETERMINISTIC
BEGIN
    RETURN ELT(FLOOR(1 + (RAND() * 10)),
        'Amit Sharma','Rohit Verma','Priya Singh','Neha Gupta',
        'Rahul Mehta','Ankit Jain','Pooja Kapoor','Vikas Yadav',
        'Sneha Reddy','Arjun Nair');
END$$

DELIMITER ;

-- =========================================
-- 6. BULK INSERT PROCEDURE
-- =========================================
DELIMITER $$

CREATE PROCEDURE insert_dummy_employees(IN total_rows INT)
BEGIN
    DECLARE i INT DEFAULT 0;

    START TRANSACTION;

    WHILE i < total_rows DO
        
        INSERT INTO employees 
        (employee_name, department, position, performance_score, join_date, experience_years, salary)
        VALUES
        (
            random_name(),
            ELT(FLOOR(1 + (RAND() * 8)),
                'Engineering','Marketing','Sales','HR',
                'Finance','Operations','IT','R&D'),
            ELT(FLOOR(1 + (RAND() * 5)),
                'Engineer','Manager','Analyst','Executive','Specialist'),
            ROUND(60 + (RAND() * 40), 2),
            DATE_SUB(CURDATE(), INTERVAL FLOOR(RAND()*2000) DAY),
            FLOOR(RAND()*10),
            ROUND(30000 + (RAND()*100000), 2)
        );

        SET i = i + 1;

        -- Commit every 10,000 rows
        IF i % 10000 = 0 THEN
            COMMIT;
            START TRANSACTION;
        END IF;

    END WHILE;

    COMMIT;
END$$

DELIMITER ;

-- =========================================
-- 7. RUN DATA GENERATION
-- =========================================

-- Start small (recommended)
CALL insert_dummy_employees(100000);   -- 100K

-- Scale gradually
-- CALL insert_dummy_employees(1000000);    -- 1 Million
-- CALL insert_dummy_employees(10000000);   -- 10 Million
-- CALL insert_dummy_employees(100000000);  -- 100 Million (Heavy)

-- =========================================
-- 8. RESTORE SETTINGS
-- =========================================
SET autocommit = 1;
SET unique_checks = 1;
SET foreign_key_checks = 1;

-- =========================================
-- 9. SAMPLE ANALYTICS QUERIES
-- =========================================

-- Employees with department info
SELECT e.*, d.manager_name, d.budget 
FROM employees e 
JOIN departments d ON e.department = d.department_name;

-- Avg performance per department
SELECT department, AVG(performance_score) AS avg_performance, COUNT(*) AS employee_count
FROM employees
GROUP BY department;

-- Top performers
SELECT * FROM employees
ORDER BY performance_score DESC
LIMIT 10;