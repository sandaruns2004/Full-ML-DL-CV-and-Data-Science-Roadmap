# 📊 Exploratory Data Analysis (EDA)

> **Prerequisites**: Python Essentials, What Is Data Science and ML | **Difficulty**: ⭐☆☆☆☆ Beginner

---

## 📋 Table of Contents

1. [What Is EDA?](#1-what-is-eda)
2. [The EDA Workflow Checklist](#2-the-eda-workflow-checklist)
3. [Univariate Analysis](#3-univariate-analysis)
4. [Bivariate & Multivariate Analysis](#4-bivariate--multivariate-analysis)
5. [Missing Values & Outlier Detection](#5-missing-values--outlier-detection)
6. [Automated EDA Profiling Deep Dive](#6-automated-eda-profiling-deep-dive)
7. [Real-World Case Study: Retail Data](#7-real-world-case-study-retail-data)
8. [Descriptive Statistics Deep Dive](#8-descriptive-statistics-deep-dive)
9. [Data Storytelling & Reporting](#9-data-storytelling--reporting)
10. [Project Ideas](#10-project-ideas)
11. [What's Next](#11-whats-next)

---

## 1. What Is EDA?

Exploratory Data Analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods. 

> **"You can't build a good model on data you don't understand."**

EDA helps you:
1. Uncover underlying structure and hidden patterns.
2. Identify anomalies, outliers, and missing data.
3. Test assumptions required for machine learning algorithms.
4. Determine optimal features for machine learning (Feature Selection/Engineering).

---

## 2. The EDA Workflow Checklist

Before building any models, run through this checklist with your DataFrame:

1. **Shape and Structure**: How many rows/columns? What are the data types?
2. **First Look**: `df.head()`, `df.tail()`, `df.sample()`. Does the data look as expected?
3. **Missing Values**: How many missing values per column? Is there a pattern to the missingness?
4. **Summary Statistics**: `df.describe()`. Look at means, min/max, and percentiles.
5. **Categorical Distributions**: What are the unique classes and their frequencies?
6. **Target Variable**: If doing ML, what is the distribution of the thing you are trying to predict?

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample dataset for demonstration
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'age': np.random.normal(35, 12, n).clip(18, 80),
    'income': np.random.lognormal(10.5, 0.8, n),
    'education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], n, p=[0.4, 0.4, 0.15, 0.05]),
    'credit_score': np.random.normal(650, 80, n).clip(300, 850),
    'churned': np.random.binomial(1, 0.2, n)
})
# Inject some missing values
df.loc[np.random.choice(n, 50), 'income'] = np.nan

# ==========================================
# 1. Basic Structure
# ==========================================
print(f"Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")

# ==========================================
# 2. Missing Values Analysis
# ==========================================
missing = df.isnull().sum()
print(f"\nMissing Values:\n{missing[missing > 0]}")

# ==========================================
# 3. Numerical Summary
# ==========================================
print(f"\nNumerical Summary:\n{df.describe().round(2)}")

# ==========================================
# 4. Target Variable Analysis
# ==========================================
churn_rate = df['churned'].mean()
print(f"\nOverall Churn Rate: {churn_rate:.2%}")
print(df['churned'].value_counts())
```

---

## 3. Univariate Analysis

Univariate analysis looks at **one variable at a time**.

### For Numerical Variables:
- **Histograms / KDE**: To see the distribution shape (normal, skewed, bimodal).
- **Boxplots**: To identify outliers and quartiles.

### For Categorical Variables:
- **Countplots / Bar charts**: To see the frequency of each category.
- **Pie charts**: (Use sparingly, only for 2-3 categories to show parts of a whole).

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Histogram + KDE for Income (Numerical)
sns.histplot(df['income'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Income Distribution (Right Skewed)')
axes[0].set_xscale('log') # Log scale because income is highly skewed

# 2. Boxplot for Age (Numerical)
sns.boxplot(y=df['age'], ax=axes[1], color='lightgreen')
axes[1].set_title('Age Distribution (Boxplot)')

# 3. Countplot for Education (Categorical)
sns.countplot(data=df, x='education', ax=axes[2], palette='viridis', 
              order=['High School', 'Bachelors', 'Masters', 'PhD'])
axes[2].set_title('Education Level Frequencies')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

---

## 4. Bivariate & Multivariate Analysis

Bivariate analysis looks at the relationship between **two variables**. Multivariate looks at **three or more**.

### Numerical vs Numerical
- **Scatterplots**: Shows correlation and non-linear patterns.
- **Correlation Heatmap**: Pearson/Spearman correlation across all numeric columns.

### Categorical vs Numerical
- **Boxplots / Violinplots**: Comparing distributions across categories.
- **Groupby Means**: Average value of a numeric column per category.

### Categorical vs Categorical
- **Grouped Bar Charts / Stacked Bar Charts**
- **Crosstabs / Contingency Tables**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Scatterplot: Age vs Credit Score (colored by Churn)
sns.scatterplot(data=df, x='age', y='credit_score', hue='churned', alpha=0.6, ax=axes[0])
axes[0].set_title('Age vs Credit Score')

# 2. Violinplot: Credit Score by Education
sns.violinplot(data=df, x='education', y='credit_score', ax=axes[1], palette='Set2',
               order=['High School', 'Bachelors', 'Masters', 'PhD'])
axes[1].set_title('Credit Score by Education Level')
axes[1].tick_params(axis='x', rotation=45)

# 3. Correlation Heatmap
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[2])
axes[2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.show()
```

---

## 5. Missing Values & Outlier Detection

EDA is where you *detect* data quality issues. (You will *fix* them in the Data Preprocessing stage).

### Detecting Missing Values
Use the `missingno` library for beautiful missing data visualizations.

```python
import missingno as msno

# Visualizes where missing values occur in the dataset
msno.matrix(df, figsize=(10, 4))
plt.show()
```

### Detecting Outliers
- **Z-Score Method**: Any point further than 3 standard deviations from the mean. (Assumes normal distribution).
- **IQR Method (Tukey)**: Any point outside $[Q1 - 1.5 \times IQR, \ Q3 + 1.5 \times IQR]$.

```python
# IQR Method for Outlier Detection
Q1 = df['credit_score'].quantile(0.25)
Q3 = df['credit_score'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['credit_score'] < lower_bound) | (df['credit_score'] > upper_bound)]
print(f"Detected {len(outliers)} outliers in Credit Score.")
```

---

## 6. Automated EDA Profiling Deep Dive

Instead of writing dozens of lines of code for basic EDA, you can generate comprehensive HTML reports automatically. 

**Popular Tools:**
1. **ydata-profiling** (formerly `pandas-profiling`): The industry standard.
2. **Sweetviz**: Great for comparing two datasets (e.g., train vs test).
3. **AutoViz**: Very fast for massive datasets.

```python
# Example using ydata-profiling
# pip install ydata-profiling

from ydata_profiling import ProfileReport

# Generate the report with explorative mode (calculates more interactions)
profile = ProfileReport(df, title="Customer Dataset Profiling Report", explorative=True)

# Save to HTML
# profile.to_file("report.html")
```

**What `ydata-profiling` gives you out of the box:**
- **Overview**: Dataset statistics, variable types, missing cells.
- **Variables**: Quantile statistics (min, max, median, IQR), descriptive stats (mean, standard dev, skewness, kurtosis), and a histogram.
- **Interactions**: Scatter plots for every pair of numeric variables.
- **Correlations**: Spearman, Pearson, and Kendall matrices.
- **Missing Values**: Matrices and dendrograms.
- **Alerts**: It automatically flags High Cardinality, High Correlation, Missing Data, and Imbalance.

*Note: Automated profiling is great for a first glance, but it does NOT replace deep, manual exploration driven by business logic and domain knowledge.*

---

## 7. Real-World Case Study: Retail Data

When dealing with real datasets (like an E-Commerce log), the EDA process becomes messier and requires more context.

**Scenario**: You are given a dataset of online retail transactions.
- `InvoiceNo`: The transaction ID.
- `StockCode`: The item ID.
- `Description`: Text description of the item.
- `Quantity`: Number of items bought.
- `InvoiceDate`: Timestamp.
- `UnitPrice`: Cost per item.
- `CustomerID`: Unique customer ID.
- `Country`: Customer location.

**Real-world EDA Questions to Ask:**
1. **Sanity Checks**: Are there negative `Quantity` values? (Yes, these might be returns!). Are there $0 `UnitPrice` items? (Promotions or missing data?).
2. **Feature Engineering via EDA**: We have `Quantity` and `UnitPrice`, but the business cares about `Revenue`. We must create `df['Revenue'] = df['Quantity'] * df['UnitPrice']` during our EDA to plot revenue distributions.
3. **Time Series Patterns**: By parsing `InvoiceDate`, we can group by month or day of the week. "Do we sell more on weekends?"
4. **Cohort Analysis**: Grouping `CustomerID` by their first purchase date to see how retention drops over time.

```python
# Real-world EDA snippet
import pandas as pd

# Load data
# df_retail = pd.read_csv('online_retail.csv')

# 1. Handle Returns (Negative Quantities)
# returns = df_retail[df_retail['Quantity'] < 0]
# sales = df_retail[df_retail['Quantity'] > 0]

# 2. Time-based Exploration
# df_retail['InvoiceDate'] = pd.to_datetime(df_retail['InvoiceDate'])
# df_retail['Month'] = df_retail['InvoiceDate'].dt.month
# monthly_sales = df_retail.groupby('Month')['Quantity'].sum()
```

In real-world data science, 80% of your time is spent in this phase—cleaning the data as you discover anomalies during EDA.

---

## 8. Descriptive Statistics Deep Dive

### Measures of Shape

1. **Skewness**: Measures the asymmetry of the probability distribution.
   - Normal distribution has a skew of 0.
   - **Right-skewed** (Positive skew): Tail is on the right (e.g., Income). Mean > Median.
   - **Left-skewed** (Negative skew): Tail is on the left (e.g., Age at retirement). Mean < Median.

2. **Kurtosis**: Measures the "tailedness" of the distribution.
   - **Mesokurtic** (Kurtosis ≈ 3): Normal distribution.
   - **Leptokurtic** (Kurtosis > 3): Heavy tails, more outliers.
   - **Platykurtic** (Kurtosis < 3): Light tails, fewer outliers.

```python
from scipy import stats

income_skew = df['income'].dropna().skew()
income_kurt = df['income'].dropna().kurtosis()

print(f"Income Skewness: {income_skew:.2f} (Highly right-skewed)")
print(f"Income Kurtosis: {income_kurt:.2f} (Heavy tails)")
```

---

## 9. Data Storytelling & Reporting

The ultimate goal of EDA is often to present findings to stakeholders. 

### The Data Storytelling Framework

1. **CONTEXT**: What's the business question? (e.g., "Why are customers churning?")
2. **EXPLORE**: What does the data show? (EDA phase).
3. **INSIGHT**: What's the key finding? (e.g., "Customers with low credit scores and high school education churn 3x more.")
4. **ACTION**: What should we do about it? (e.g., "Create a targeted retention campaign for this demographic.")
5. **VISUALIZE**: How do we communicate this? (Clean, annotated, un-cluttered charts).

### Visualization for Presentation
When presenting, remove all non-essential elements (chart junk). Add explicit titles and annotations that state the insight directly.

```python
# A presentation-ready chart
plt.figure(figsize=(10, 6))

# Group data for visualization
churn_by_edu = df.groupby('education')['churned'].mean().sort_values(ascending=False) * 100

bars = plt.bar(churn_by_edu.index, churn_by_edu.values, color=['#e74c3c', '#bdc3c7', '#bdc3c7', '#bdc3c7'])

# Annotations
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", 
             ha='center', va='bottom', fontweight='bold')

# Styling for presentation
plt.title('High School Educated Customers Drive Churn', fontsize=16, fontweight='bold', pad=20, loc='left')
plt.ylabel('Churn Rate (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()
```

---

## 10. Project Ideas

### 🟢 Project 1: Automated EDA Pipeline (Beginner)
Write a Python script `analyze_data.py` that accepts any CSV file. The script should automatically print shape, missing values, numerical summary, and generate a correlation heatmap image `correlation.png`.

### 🟡 Project 2: Kaggle Dataset Deep Dive (Intermediate)
Pick a classic dataset (like the Titanic, or Ames Housing) and perform a comprehensive manual EDA without using automated tools like `ydata-profiling`. Focus on finding multivariate relationships and documenting them with markdown text answering *why* a relationship exists.

### 🔴 Project 3: Interactive EDA Dashboard (Advanced)
Use **Streamlit** or **Dash** to build an interactive web app. Users can upload a CSV, select columns from dropdowns, and the app dynamically generates histograms, scatterplots, and summary statistics.

---

## 11. What's Next

| Next Topic | Why |
|------------|-----|
| [Statistical Inference](./03-Statistical-Inference.md) | How do we prove that the patterns we saw in EDA are statistically significant? |
| [Data Preprocessing](./06-Data-Preprocessing.md) | Now that we found the missing values and outliers, how do we fix them? |
| [Feature Engineering](./07-Feature-Engineering.md) | Using insights from EDA to create powerful new variables for Machine Learning. |

---

[← What Is Data Science And ML](./01-What-Is-Data-Science-And-ML.md) | [Back to Index](../README.md) | [Next: Statistical Inference →](./03-Statistical-Inference.md)
