# 🗄️ SQL for Data Science

> **Prerequisites**: None | **Difficulty**: ⭐⭐☆☆☆ Elementary

---

## 📋 Table of Contents
1. [Why SQL is Critical for ML](#1-why-sql-is-critical-for-ml)
2. [Basic Retrieval & Filtering](#2-basic-retrieval--filtering)
3. [Aggregations (GROUP BY)](#3-aggregations-group-by)
4. [Joins](#4-joins)
5. [Window Functions](#5-window-functions)
6. [CTEs (Common Table Expressions)](#6-ctes-common-table-expressions)

---

## 1. Why SQL is Critical for ML

While many courses teach Machine Learning using pre-cleaned CSV files loaded via `pandas.read_csv()`, in the real world, **your data lives in a database or a data warehouse** (e.g., PostgreSQL, Snowflake, BigQuery). 

Before you can train a model, you must extract, join, and aggregate millions of rows of raw data. Doing this in Pandas or Python memory is often impossible. **SQL (Structured Query Language)** is the primary tool used to prepare data for ML pipelines.

---

## 2. Basic Retrieval & Filtering

The core of SQL is the `SELECT` statement.

```sql
SELECT 
    user_id, 
    age, 
    signup_date
FROM users
WHERE age >= 18 
  AND status = 'active'
ORDER BY signup_date DESC
LIMIT 100;
```
- `SELECT`: Which columns to return.
- `FROM`: Which table to read.
- `WHERE`: Row-level filtering.
- `ORDER BY` / `LIMIT`: Sorting and truncating the output.

---

## 3. Aggregations (GROUP BY)

To build features for machine learning (e.g., "total spent by user"), you must aggregate row-level data.

```sql
SELECT 
    user_id,
    COUNT(order_id) AS total_orders,
    SUM(amount) AS total_spent,
    AVG(amount) AS avg_order_value,
    MAX(order_date) AS last_order_date
FROM orders
WHERE order_date >= '2023-01-01'
GROUP BY user_id
HAVING COUNT(order_id) > 5;
```
- `GROUP BY`: Determines the level of aggregation.
- `HAVING`: Filters the results **after** aggregation (unlike `WHERE` which filters before).

---

## 4. Joins

In relational databases, data is normalized across multiple tables. You use `JOIN` to bring them together.

```sql
SELECT 
    u.user_id,
    u.age,
    SUM(o.amount) AS total_spent
FROM users u
LEFT JOIN orders o 
    ON u.user_id = o.user_id
GROUP BY u.user_id, u.age;
```
- **INNER JOIN**: Keeps only rows that match in both tables.
- **LEFT JOIN**: Keeps all rows from the left table (`users`), filling with NULLs if there are no matching orders. (The most common join for building ML datasets).

---

## 5. Window Functions

Window functions perform calculations across a set of rows related to the current row. Unlike `GROUP BY`, they **do not collapse** the rows. They are essential for time-series features (e.g., moving averages, row numbering).

```sql
SELECT 
    user_id,
    order_date,
    amount,
    -- Calculate a running total for each user over time
    SUM(amount) OVER (
        PARTITION BY user_id 
        ORDER BY order_date
    ) AS running_total_spent,
    -- Rank the user's orders sequentially
    ROW_NUMBER() OVER (
        PARTITION BY user_id 
        ORDER BY order_date
    ) AS order_sequence_number
FROM orders;
```

---

## 6. CTEs (Common Table Expressions)

When building complex ML datasets, your SQL queries can become hundreds of lines long. **CTEs** (using the `WITH` clause) allow you to break complex queries into readable, reusable blocks.

```sql
WITH ActiveUsers AS (
    SELECT user_id 
    FROM users 
    WHERE status = 'active'
),
UserSpend AS (
    SELECT user_id, SUM(amount) AS total_spent
    FROM orders
    GROUP BY user_id
)
-- Main Query combining the CTEs
SELECT 
    a.user_id,
    COALESCE(s.total_spent, 0) AS total_spent
FROM ActiveUsers a
LEFT JOIN UserSpend s 
    ON a.user_id = s.user_id;
```

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Write a SQL query using SQLite in Python (`sqlite3`) to extract a dataset and load it into a Pandas DataFrame.
- 🟡 **Intermediate**: Use a Window Function to calculate the "days since last order" feature for a hypothetical customer churn model.

### What's Next
| Next | Why |
|------|-----|
| [What Is Data Science And ML](../01-Data-Science-Foundations/01-What-Is-Data-Science-And-ML.md) | Begin the core Data Science Foundations module. |

---

[← Probability And Statistics](./03-Probability-And-Statistics.md) | [Back to Index](../README.md) | [Next: What Is Data Science And ML →](../01-Data-Science-Foundations/01-What-Is-Data-Science-And-ML.md)
