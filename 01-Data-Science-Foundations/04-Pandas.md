# 🐼 Pandas for Data Science

> **Prerequisites**: Python & NumPy Basics | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents

1. [What is Pandas?](#1-what-is-pandas)
2. [Series vs DataFrames](#2-series-vs-dataframes)
3. [Filtering & Selecting Data](#3-filtering--selecting-data)
4. [GroupBy & Aggregation](#4-groupby--aggregation)
5. [Merging & Joining](#5-merging--joining)
6. [Handling Missing Values](#6-handling-missing-values)

---

## 1. What is Pandas?

### 🟢 Beginner

**Simple Explanation**: If NumPy is a super-calculator, Pandas is Excel on steroids using Python. Whenever you have data that fits nicely into a table (like a CSV file or SQL database), Pandas is the tool you use to load it, inspect it, and manipulate it.

**Real-world Analogy**: Imagine you have a massive filing cabinet full of customer records. Searching for all customers over 30 who live in New York would take days by hand. Pandas is like a magical robotic assistant that scans millions of files instantly and hands you exactly the subset of records you asked for.

**Visual Intuition**:
```text
Raw Data (CSV):          Pandas DataFrame:
Name,Age,City            |   | Name  | Age | City  |
Alice,25,NY         =>   | 0 | Alice | 25  | NY    |
Bob,30,LA           =>   | 1 | Bob   | 30  | LA    |
```

### 🟡 Intermediate

**Concepts**: 
Pandas is built *on top* of NumPy. While NumPy is excellent for pure numbers in matrices, Pandas handles heterogeneous tabular data (numbers, strings, booleans, dates mixed together). Its two primary structures are the `Series` (1-dimensional) and the `DataFrame` (2-dimensional).

**Workflow & Practical Applications**:
Loading real datasets, cleaning out errors, pivoting tables, and preparing data for Machine Learning models or visualization.

**Code Implementation (Real Dataset example)**:
```python
import pandas as pd
import seaborn as sns

# Load a real built-in dataset (Titanic passengers)
df = sns.load_dataset('titanic')

# View the first 5 rows
print(df.head())
```

### 🔴 Advanced

**Mathematics & Statistical Reasoning**:
Pandas uses labeled indexing, which means mathematical operations align on the *index* implicitly. If you subtract Series A from Series B, Pandas aligns them by their index labels before subtracting, ensuring data integrity even if the rows are out of order.

**Industry Considerations**:
Pandas is strictly single-threaded and operates entirely in-memory (RAM). If your dataset exceeds your machine's RAM (e.g., 50GB dataset on a 16GB laptop), Pandas will crash. In industry, Big Data alternatives like **PySpark**, **Dask**, or **Polars** are used to scale tabular data processing.

---

## 2. Filtering & Selecting Data

### 🟢 Beginner

**Simple Explanation**: How do we grab specific rows and columns? Like asking the database for "Only show me Female passengers in 1st class."

### 🟡 Intermediate

**Code Implementation**:
```python
# Select a single column
ages = df['age']

# Filter rows based on a condition
adults = df[df['age'] >= 18]

# Complex filtering (AND operator is &, OR is |)
first_class_females = df[(df['pclass'] == 1) & (df['sex'] == 'female')]
print(f"Number of 1st class females: {len(first_class_females)}")
```

### 🔴 Advanced

**Performance Considerations**:
Using `.loc` and `.iloc` is the preferred, explicit way to access data. `.loc` accesses via label (names), `.iloc` accesses via integer position. Avoid chained indexing (e.g., `df[df['age'] > 18]['survived'] = 1`) as it can lead to the notorious `SettingWithCopyWarning`, meaning you might be modifying a temporary copy rather than the original DataFrame.

---

## 3. GroupBy & Aggregation

### 🟢 Beginner

**Simple Explanation**: Grouping data is like sorting your laundry into piles (lights, darks, colors) and then counting the number of shirts in each pile. 

### 🟡 Intermediate

**Concepts**: The Split-Apply-Combine strategy.
1. **Split** the data into groups based on some criteria.
2. **Apply** a function (mean, sum, count) to each group independently.
3. **Combine** the results into a new data structure.

**Code Implementation**:
```python
# What was the survival rate by passenger class?
survival_rates = df.groupby('pclass')['survived'].mean()
print(survival_rates)
# pclass
# 1    0.629630
# 2    0.472826
# 3    0.242363
```

### 🔴 Advanced

**Statistical Reasoning**:
Aggregation provides descriptive statistics conditioned on categorical variables. This is the foundation of hypothesis generation. If `groupby('gender')['salary'].mean()` shows a disparity, it warrants a statistical hypothesis test (like a T-test) to confirm if the difference is statistically significant.

---

## 4. Merging & Joining

### 🟢 Beginner

**Simple Explanation**: You have a spreadsheet of Customers, and another of Orders. You want to combine them so you can see which customer made which order. Merging stitches them together using a common ID column.

### 🟡 Intermediate

**Workflow**: Similar to SQL JOINs.
- `how='inner'`: Keep only matching rows.
- `how='left'`: Keep all rows from the left table, and match the right.

**Code Implementation**:
```python
customers = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
orders = pd.DataFrame({'order_id': [101, 102], 'cust_id': [1, 2], 'amount': [250, 50]})

# Merge (Join) tables
merged = pd.merge(customers, orders, left_on='id', right_on='cust_id', how='left')
print(merged)
```

### 🔴 Advanced

**Database Theory**: 
Merging in Pandas employs Hash Joins and Sort-Merge Joins under the hood. Understanding relational algebra and database normalization will make your data merging strategies significantly more robust and computationally efficient.

---

[← NumPy for Data Science](03-NumPy.md) | [Back to Index](../README.md) | [Next: Data Collection →](05-Data-Collection.md)
