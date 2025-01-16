https://www.geeksforgeeks.org/sql-query-interview-questions/

https://www.geeksforgeeks.org/mysql-interview-questions/?ref=lbp
Here's a comprehensive breakdown of essential SQL topics, including brief explanations, code examples, and real-life applications:

---

### **1. SQL Basics**
- **What it is:** SQL (Structured Query Language) is a standard language for managing and manipulating relational databases.
- **Commands Overview:** SQL commands include **DML** (Data Manipulation Language), **DDL** (Data Definition Language), **DCL** (Data Control Language), and **TCL** (Transaction Control Language).

**Example Commands:**
```sql
-- Creating a table (DDL)
CREATE TABLE Employees (
    ID INT PRIMARY KEY,
    Name VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting data (DML)
INSERT INTO Employees (ID, Name, Age, Salary)
VALUES (1, 'John Doe', 30, 50000.00);

-- Selecting data (DML)
SELECT * FROM Employees;

-- Updating data (DML)
UPDATE Employees
SET Salary = 55000.00
WHERE ID = 1;

-- Deleting data (DML)
DELETE FROM Employees
WHERE ID = 1;
```

**Real-life Example:** Storing employee records in a company database.

### **2. Advanced SELECT Statements**
- **What it is:** Advanced queries use filters, joins, grouping, and sorting to retrieve complex data sets.

**Example Query:**
```sql
-- Filter with WHERE clause
SELECT Name, Age
FROM Employees
WHERE Age > 25;

-- Sort data with ORDER BY
SELECT Name, Age
FROM Employees
ORDER BY Age DESC;

-- Grouping with GROUP BY
SELECT Age, COUNT(*) AS NumberOfEmployees
FROM Employees
GROUP BY Age;

-- Filter groups with HAVING
SELECT Age, COUNT(*) AS NumberOfEmployees
FROM Employees
GROUP BY Age
HAVING COUNT(*) > 1;
```

**Real-life Example:** Retrieving customers based on their age or sorting products based on price.

### **3. Joins**
- **What it is:** SQL joins combine rows from two or more tables based on a related column.

**Types of Joins:**
1. **INNER JOIN**: Returns records that have matching values in both tables.
2. **LEFT JOIN** (or LEFT OUTER JOIN): Returns all records from the left table and matching records from the right table.
3. **RIGHT JOIN** (or RIGHT OUTER JOIN): Returns all records from the right table and matching records from the left table.
4. **FULL JOIN** (or FULL OUTER JOIN): Returns all records when there is a match in either left or right table.

**Example Query:**
```sql
-- INNER JOIN Example
SELECT Employees.Name, Departments.DepartmentName
FROM Employees
INNER JOIN Departments ON Employees.DepartmentID = Departments.ID;

-- LEFT JOIN Example
SELECT Employees.Name, Departments.DepartmentName
FROM Employees
LEFT JOIN Departments ON Employees.DepartmentID = Departments.ID;
```

**Real-life Example:** Retrieving customer orders and their details, such as order ID, customer name, and product name.

### **4. Subqueries and Nested Queries**
- **What it is:** A subquery is a query nested inside another query to retrieve more complex information.

**Example Query:**
```sql
-- Subquery Example
SELECT Name, Salary
FROM Employees
WHERE Salary > (SELECT AVG(Salary) FROM Employees);
```

**Real-life Example:** Finding employees with a salary above the average salary in the company.

### **5. Normalization**
- **What it is:** Organizing data to reduce redundancy and improve data integrity.

**Normal Forms (1NF to 5NF):**
- **1NF:** Eliminate duplicate columns from the same table.
- **2NF:** Remove subsets of data that apply to multiple rows.
- **3NF:** Remove columns not dependent on the primary key.
- **BCNF:** Ensure no overlapping candidate keys.
- **4NF/5NF:** Focus on multi-valued dependencies.

**Example Problem:** Breaking down a "CustomerOrder" table into separate "Customers," "Orders," and "Products" tables.

**Real-life Example:** Optimizing a large e-commerce database to ensure data consistency and storage efficiency.

### **6. ACID Properties**
- **What it is:** Properties that ensure reliable database transactions.

- **A**tomicity: Transactions are all-or-nothing.
- **C**onsistency: Transactions move the database from one valid state to another.
- **I**solation: Transactions are independent of each other.
- **D**urability: Once a transaction is committed, it remains in the system.

**Example Problem:** Ensuring that a bank transfer transaction completes successfully or rolls back entirely.

### **7. Transactions and Concurrency Control**
- **What it is:** Managing multiple transactions concurrently to avoid conflicts and ensure data integrity.

**Example Commands:**
```sql
-- Start a transaction
BEGIN TRANSACTION;

-- Perform SQL operations
UPDATE Accounts SET Balance = Balance - 100 WHERE AccountID = 1;
UPDATE Accounts SET Balance = Balance + 100 WHERE AccountID = 2;

-- Commit transaction
COMMIT;

-- Rollback if an error occurs
ROLLBACK;
```

**Real-life Example:** Handling simultaneous booking requests on a ticket booking website without double-booking seats.

### **8. Indexing**
- **What it is:** Creating indexes to speed up data retrieval based on specific columns.

**Example Command:**
```sql
CREATE INDEX idx_salary ON Employees (Salary);
```

**Real-life Example:** Creating an index on a "Products" table’s "Price" column for faster search in an e-commerce application.

### **9. Views**
- **What it is:** Virtual tables based on SQL queries that simplify complex joins or hide sensitive data.

**Example Command:**
```sql
CREATE VIEW EmployeeSalaries AS
SELECT Name, Salary
FROM Employees
WHERE Salary > 50000;
```

**Real-life Example:** Creating a view to show only high-salary employees for the management.

### **10. Stored Procedures and Functions**
- **What it is:** Predefined SQL routines to perform repetitive tasks or calculations.

**Example Command:**
```sql
-- Stored Procedure Example
CREATE PROCEDURE GetHighSalaries()
BEGIN
    SELECT Name, Salary
    FROM Employees
    WHERE Salary > 50000;
END;

-- Calling the procedure
CALL GetHighSalaries();
```

**Real-life Example:** Creating a procedure to calculate employee bonuses or a function to compute product discounts.

### **11. SQL Triggers**
- **What it is:** SQL code that automatically executes in response to certain events on a table.

**Example Trigger:**
```sql
CREATE TRIGGER SalaryCheck
BEFORE UPDATE ON Employees
FOR EACH ROW
BEGIN
    IF NEW.Salary < 30000 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Salary cannot be less than 30,000';
    END IF;
END;
```

**Real-life Example:** Preventing invalid salary updates or automatically updating inventory stock on a sale.

### **12. SQL Injection and Security**
- **What it is:** Techniques to prevent malicious SQL code execution.

**Best Practices:**
- Use **prepared statements** to prevent SQL Injection.
- Implement **user access controls** using GRANT and REVOKE.

**Example Problem:** Protecting a login page against SQL injection.

### **13. Advanced SQL Functions**
- **What it is:** Functions such as `COUNT()`, `SUM()`, `AVG()`, `MIN()`, `MAX()`, etc., and window functions like `ROW_NUMBER()`, `RANK()`, `PARTITION BY`.

**Example Query:**
```sql
-- Using aggregate functions
SELECT DepartmentID, AVG(Salary) AS AverageSalary
FROM Employees
GROUP BY DepartmentID;

-- Window function example
SELECT Name, Salary, RANK() OVER (ORDER BY Salary DESC) AS SalaryRank
FROM Employees;
```

**Real-life Example:** Calculating sales totals by region or ranking employees by performance scores.

---

If you master these topics, you'll gain a solid understanding of SQL and its practical applications, allowing you to tackle complex database-related tasks in real-world scenarios.








1- https://www.geeksforgeeks.org/how-to-print-duplicate-rows-in-a-table/
