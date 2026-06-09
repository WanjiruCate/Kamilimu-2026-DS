# DBeaver Practice — SQL Exercises
**Database:** `student_practice.db`

---

## The Dataset

You are working with a fictional company's database. It has **5 tables**:

```
departments   — company departments and their locations
employees     — staff records including salary and department
products      — items the company sells
orders        — sales orders placed by employees
order_items   — individual line items within each order
```

### Schema Diagram

**Entity Relationship Diagram (ERD):**

```
┌─────────────┐     ┌───────────┐     ┌────────┐     ┌─────────────┐     ┌──────────┐
│ departments │────<│ employees │────<│ orders │────<│ order_items │>────│ products │
└─────────────┘     └───────────┘     └────────┘     └─────────────┘     └──────────┘
     1                    N              1        N        N         M         1
```

**Relationship Key:**
- `1 ────< N` : One-to-Many relationship (one department has many employees)
- `N >──── 1` : Many-to-One relationship (many order_items reference one product)
- **Primary Keys (PK)**: Unique identifier for each table
- **Foreign Keys (FK)**: References to related tables

**Detailed Relationships:**
1. **departments → employees**: One department has many employees (`department_id` FK in employees)
2. **employees → orders**: One employee can create many orders (`employee_id` FK in orders)
3. **orders → order_items**: One order contains many line items (`order_id` FK in order_items)
4. **products → order_items**: One product can appear in many order items (`product_id` FK in order_items)

**Table Structure Summary:**
```
departments
├── department_id (PK)
├── department_name
└── location

employees
├── employee_id (PK)
├── first_name
├── last_name
├── email
├── hire_date
├── job_title
├── salary
└── department_id (FK → departments)

orders
├── order_id (PK)
├── order_date
├── employee_id (FK → employees)
└── status

order_items
├── item_id (PK)
├── order_id (FK → orders)
├── product_id (FK → products)
├── quantity
└── unit_price

products
├── product_id (PK)
├── product_name
├── category
├── stock_quantity
└── unit_price

```

---

## How to Load the File in DBeaver

1. Open DBeaver → click **New Database Connection**
2. Select **SQLite** → click **Next**
3. Under **Path**, click **Open** and browse to `student_practice.db`
4. Click **Finish**


---

## How to open SQL Queries in DBeaver

1. Expand the connection in the left panel → open **SQL Editor → New SQL Script**
2. Write your SQL query.
3. Click the Run button (or press `Ctrl + Enter`) to execute the query.

---

## Exercises

### Level 1 — Basics: SELECT, WHERE, ORDER BY

**Exercise 1.1** — List all employees (first name, last name, job title).

```sql
SELECT first_name, last_name, job_title
FROM employees;
```

**Exercise 1.2** — Show only employees in department 1.

```sql
SELECT first_name, last_name, job_title
FROM employees
WHERE department_id = 1;
```

**Exercise 1.3** — List all products priced above 100, ordered by price (highest first).

```sql
SELECT product_name, category, unit_price
FROM products
WHERE unit_price > 100
ORDER BY unit_price DESC;
```

**Exercise 1.4** — Find all orders that were delivered.

```sql
SELECT order_id, order_date, status
FROM orders
WHERE status = 'Delivered';
```

---

### Level 2 — Aggregations: COUNT, SUM, AVG, GROUP BY

**Exercise 2.1** — How many employees are in each department?

```sql
SELECT department_id, COUNT(*) AS employee_count
FROM employees
GROUP BY department_id;
```

**Exercise 2.2** — What is the average salary by job title?

```sql
SELECT job_title, ROUND(AVG(salary), 2) AS avg_salary
FROM employees
GROUP BY job_title
ORDER BY avg_salary DESC;
```

**Exercise 2.3** — What is the total revenue from all delivered orders?

```sql
SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'Delivered';
```

**Exercise 2.4** — How many orders does each status category have?

```sql
SELECT status, COUNT(*) AS order_count
FROM orders
GROUP BY status;
```

---

### Level 3 — Joins

**Exercise 3.1** — Show each employee with their department name.

```sql
SELECT e.first_name, e.last_name, d.department_name, d.location
FROM employees e
JOIN departments d ON e.department_id = d.department_id;
```

**Exercise 3.2** — Show all orders with the name of the employee who placed them.

```sql
SELECT o.order_id, o.order_date, o.status,
       e.first_name || ' ' || e.last_name AS placed_by
FROM orders o
JOIN employees e ON o.employee_id = e.employee_id;
```

**Exercise 3.3** — List the products in each order along with quantities.

```sql
SELECT o.order_id, p.product_name, oi.quantity, oi.unit_price,
       ROUND(oi.quantity * oi.unit_price, 2) AS line_total
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN products p ON oi.product_id = p.product_id
ORDER BY o.order_id;
```

**Exercise 3.4** — Which department has the highest total salary bill?

```sql
SELECT d.department_name, ROUND(SUM(e.salary), 2) AS total_salary
FROM employees e
JOIN departments d ON e.department_id = d.department_id
GROUP BY d.department_name
ORDER BY total_salary DESC
LIMIT 1;
```

---

### Level 4 — Subqueries & Filtering

**Exercise 4.1** — Find employees who earn above the company average salary.

```sql
SELECT first_name, last_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees)
ORDER BY salary DESC;
```

**Exercise 4.2** — Which products have never been ordered?

```sql
SELECT product_name, category
FROM products
WHERE product_id NOT IN (SELECT DISTINCT product_id FROM order_items);
```

**Exercise 4.3** — Find the top 3 best-selling products by total quantity sold.

```sql
SELECT p.product_name, SUM(oi.quantity) AS total_sold
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.product_name
ORDER BY total_sold DESC
LIMIT 3;
```

---

### Challenge Exercises

**Challenge 1** — Show each employee's name, department, and whether their salary is above or below their department's average.

<details>
<summary>💡 Click to reveal solution</summary>

```sql
SELECT
    e.first_name || ' ' || e.last_name AS employee,
    d.department_name,
    e.salary,
    ROUND(AVG(e.salary) OVER (PARTITION BY e.department_id), 2) AS dept_avg,
    CASE
        WHEN e.salary > AVG(e.salary) OVER (PARTITION BY e.department_id) THEN 'Above Average'
        ELSE 'Below Average'
    END AS vs_dept_avg
FROM employees e
JOIN departments d ON e.department_id = d.department_id;
```

</details>

**Challenge 2** — For each month in 2024, show the total order value (delivered orders only).

<details>
<summary>💡 Click to reveal solution</summary>

```sql
SELECT
    strftime('%Y-%m', o.order_date) AS month,
    COUNT(DISTINCT o.order_id)      AS orders_count,
    ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'Delivered'
GROUP BY month
ORDER BY month;
```

</details>

---

## Exploration Tasks (No hints!)

Try these on your own using what you've learned:

1. Which employee has placed the most orders?
2. List all products in the **Electronics** category with fewer than 100 units in stock.
3. What is the most expensive single order (by total value)?
4. Which location (city) has the most employees across all departments?
5. Find all employees hired after 2020 who earn more than 80,000.

---

*Happy querying! Run each query with Ctrl+Enter (Windows/Linux) or Cmd+Enter (macOS).*
