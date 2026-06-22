# 💾 SQL for Data Science

> **Prerequisites**: [Feature Engineering](./14-Feature-Engineering.md) | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 📋 Table of Contents

1. [Why SQL is Mandatory for Data Science](#1-why-sql-is-mandatory-for-data-science)
2. [The Core Query Structure](#2-the-core-query-structure)
3. [Filtering and Aggregating Data](#3-filtering-and-aggregating-data)
4. [Relational Algebra (JOINs)](#4-relational-algebra-joins)
5. [Advanced Analytics (Window Functions)](#5-advanced-analytics-window-functions)
6. [What's Next](#6-whats-next)

---

## 1. Why SQL is Mandatory for Data Science

### 🟢 Beginner

> 💡 *For a full SQL syntax tutorial including database creation, see [00-Prerequisites/04-SQL-For-Data-Science.md](../00-Prerequisites/04-SQL-For-Data-Science.md).*

If you are a Data Scientist at a tech company, your company's data does not live in an Excel file. It lives in a massive database system (PostgreSQL, Snowflake, BigQuery, Redshift). 

**SQL (Structured Query Language)** is how you ask the database for data. 

**Why not just use Pandas?**
If the company has 500 million rows of purchase history, loading that into Pandas on your laptop will instantly crash your computer. You must use SQL to filter, aggregate, and narrow the data down to a manageable size *before* you bring it into Python.

---

## 2. The Core Query Structure

### 🟢 Beginner

Every analytical SQL query follows the same structure. The order in which you WRITE the code is different from the order in which the database EXECUTES it.

**Writing Order:**
1. `SELECT` (What columns do you want?)
2. `FROM` (What table is it in?)
3. `WHERE` (Filter the rows)
4. `GROUP BY` (Aggregate)
5. `HAVING` (Filter the aggregations)
6. `ORDER BY` (Sort the final output)

**Execution Order (How the database thinks):**
1. `FROM` -> 2. `WHERE` -> 3. `GROUP BY` -> 4. `HAVING` -> 5. `SELECT` -> 6. `ORDER BY`

```sql
-- "Show me the top 5 countries where we sold more than 10,000 units"
SELECT 
    country, 
    SUM(sales_volume) as total_volume
FROM 
    global_sales
WHERE 
    year = 2023
GROUP BY 
    country
HAVING 
    SUM(sales_volume) > 10000
ORDER BY 
    total_volume DESC
LIMIT 5;
```

---

## 3. Filtering and Aggregating Data

### 🟡 Intermediate

Data Scientists use SQL primarily to reduce massive tables into smaller statistical summaries.

**1. Filtering with WHERE**
```sql
SELECT user_id, age, subscription_plan
FROM users
WHERE age BETWEEN 18 AND 35
  AND subscription_plan IN ('Premium', 'Enterprise')
  AND last_login IS NOT NULL;
```

**2. Aggregations (The Math)**
The moment you use an aggregation function (`COUNT`, `SUM`, `AVG`), you **must** include a `GROUP BY` clause for any column that is NOT aggregated.

```sql
SELECT 
    department,
    COUNT(employee_id) as total_employees,
    ROUND(AVG(salary), 2) as avg_salary,
    MAX(salary) as highest_salary
FROM 
    employees
GROUP BY 
    department;
```

**3. The CASE Statement (If/Else Logic)**
This is extremely useful for Feature Engineering directly inside the database.

```sql
SELECT 
    customer_id,
    total_spent,
    CASE 
        WHEN total_spent > 1000 THEN 'Whale'
        WHEN total_spent > 500 THEN 'Gold'
        ELSE 'Standard'
    END as customer_segment
FROM 
    customers;
```

---

## 4. Relational Algebra (JOINs)

### 🟡 Intermediate

Data is normalized (split) across multiple tables to save space. You must use `JOIN` to stitch it back together.

*   `users` table: Has `user_id` and `name`
*   `orders` table: Has `order_id`, `user_id`, and `amount`

```sql
-- Get the name of the user and how much they spent on each order
SELECT 
    u.name, 
    o.order_id, 
    o.amount
FROM 
    users u
INNER JOIN 
    orders o ON u.user_id = o.user_id;
```

**Types of Joins:**
- **INNER JOIN**: Only keeps rows where the ID exists in BOTH tables.
- **LEFT JOIN**: Keeps ALL rows from the first table. If they don't have an order, the order amount will just be `NULL`. *(Data Scientists use LEFT JOIN 90% of the time to avoid accidentally deleting data).*
- **RIGHT JOIN**: Keeps ALL rows from the second table.
- **FULL OUTER JOIN**: Keeps absolutely everything from both tables.

---

## 5. Advanced Analytics (Window Functions)

### 🔴 Advanced

Window functions are the secret weapon of Data Science in SQL. They allow you to perform calculations across a set of table rows related to the current row, **without collapsing the rows** (unlike `GROUP BY`).

They use the `OVER()` keyword.

**Example 1: Running Totals (Cumulative Sum)**
```sql
SELECT 
    date,
    daily_sales,
    SUM(daily_sales) OVER (ORDER BY date) as cumulative_revenue
FROM 
    sales_data;
```

**Example 2: Ranking (Who was the top earner in each department?)**
```sql
SELECT 
    employee_name,
    department,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank
FROM 
    employees;
```
*(You could wrap this in a subquery to `WHERE dept_rank = 1` to get only the highest paid person per department).*

**Example 3: Moving Averages**
```sql
SELECT 
    date,
    stock_price,
    AVG(stock_price) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as 7_day_moving_avg
FROM 
    stock_history;
```

---

## 6. What's Next

You now have the technical skills to acquire, clean, visualize, test, and engineer data. But before you unleash your models on the world, you must understand the responsibility that comes with this power.

| Next Topic | Why |
|------------|-----|
| [Data Ethics](./16-Data-Ethics.md) | Learn how to identify algorithmic bias, ensure privacy, and build responsible AI systems that don't harm society. |

---

[← Feature Engineering](14-Feature-Engineering.md) | [Back to Index](../README.md) | [Next: Data Ethics and Responsible AI →](16-Data-Ethics.md)
