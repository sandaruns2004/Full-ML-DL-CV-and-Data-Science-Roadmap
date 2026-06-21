# 📊 Exploratory Data Analysis (EDA)

> **Prerequisites**: [Data Cleaning](./04-Data-Cleaning.md) | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 📋 Table of Contents

1. [What is EDA?](#1-what-is-eda)
2. [Univariate Analysis (One Variable)](#2-univariate-analysis-one-variable)
3. [Bivariate Analysis (Two Variables)](#3-bivariate-analysis-two-variables)
4. [Multivariate Analysis & Correlation](#4-multivariate-analysis--correlation)
5. [Automated EDA Profiling](#5-automated-eda-profiling)
6. [What's Next](#6-whats-next)

---

## 1. What is EDA?

### 🟢 Beginner

**Exploratory Data Analysis (EDA)** is the process of acting like a detective with your data. Before you build complex machine learning models, you need to understand the shape, distribution, and relationships hidden within your dataset.

**The goals of EDA:**
1. Maximize insight into a dataset
2. Uncover underlying structure and hidden patterns
3. Extract important variables (Feature Selection prep)
4. Detect anomalies and outliers
5. Test underlying assumptions (e.g., Is this data normally distributed?)

> *"EDA is an attitude, a state of flexibility, a willingness to look for those things that we believe are not there, as well as those we believe to be there."* — John Tukey (The father of EDA)

---

## 2. Univariate Analysis (One Variable)

### 🟡 Intermediate

Univariate analysis looks at **one variable at a time** to understand its distribution.

We divide variables into two types:
1. **Numerical (Continuous)**: Age, Height, Salary
2. **Categorical (Discrete)**: Gender, Eye Color, City

**For Numerical Data:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = sns.load_dataset('titanic')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Histogram (Shows the distribution shape)
sns.histplot(df['fare'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Passenger Fares')

# 2. Boxplot (Shows outliers and quartiles)
sns.boxplot(x=df['fare'], ax=axes[1])
axes[1].set_title('Boxplot of Fares (Outliers visible!)')

plt.show()
```

**For Categorical Data:**

```python
# 1. Count Plot (Bar chart for categories)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='class', palette='Set2')
plt.title('Passenger Class Count')
plt.show()

# 2. Value Counts (Text output)
print(df['class'].value_counts(normalize=True)) 
# normalize=True gives percentages instead of raw counts
```

---

## 3. Bivariate Analysis (Two Variables)

### 🟡 Intermediate

Bivariate analysis explores the **relationship between two variables**. Does an increase in Age lead to an increase in Salary?

**1. Numerical vs. Numerical (Scatter Plots)**

```python
# Scatter plot to find linear/non-linear relationships
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='age', y='fare', alpha=0.5)
plt.title('Age vs Fare')
plt.show()
```

**2. Categorical vs. Numerical (Boxplots / Violin Plots)**
How does the numerical distribution change across different categories?

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot: Compares distributions
sns.boxplot(data=df, x='class', y='age', ax=axes[0])
axes[0].set_title('Age Distribution by Passenger Class')

# Violin Plot: Combines boxplot with kernel density estimation
sns.violinplot(data=df, x='survived', y='age', ax=axes[1])
axes[1].set_title('Age Distribution by Survival')

plt.show()
```

**3. Categorical vs. Categorical (Stacked Bars / Crosstabs)**
Are two categories dependent on each other?

```python
# Create a crosstab table
cross_tab = pd.crosstab(df['class'], df['survived'], normalize='index')
print(cross_tab)

# Plot as stacked bar chart
cross_tab.plot(kind='bar', stacked=True, figsize=(8, 5))
plt.title('Survival Rate by Class')
plt.ylabel('Percentage')
plt.show()
```

---

## 4. Multivariate Analysis & Correlation

### 🔴 Advanced

When dealing with 3 or more variables, we use Multivariate Analysis.

**1. The Correlation Matrix**
Correlation measures how strongly two numerical variables are related.
*   **1.0**: Perfect positive correlation (A goes up, B goes up)
*   **-1.0**: Perfect negative correlation (A goes up, B goes down)
*   **0.0**: No linear correlation

```python
import numpy as np

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Calculate Pearson correlation
corr = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
# cmap='coolwarm' clearly shows positive (red) and negative (blue)
# annot=True shows the actual numbers
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.show()
```

> ⚠️ **WARNING**: Correlation does NOT imply causation! Ice cream sales and shark attacks are highly correlated, but one does not cause the other (they are both caused by a third variable: summer weather).

**2. Pair Plots**
A Pair Plot creates a grid showing the relationship between *every* combination of numerical variables, color-coded by a categorical variable.

```python
# Drop missing values for pairplot
clean_df = df[['age', 'fare', 'survived', 'class']].dropna()

# hue='survived' colors the dots based on survival
sns.pairplot(clean_df, hue='survived', diag_kind='kde')
plt.show()
```

---

## 5. Automated EDA Profiling

### 🔴 Advanced

Writing all the matplotlib code above takes time. In professional environments, Data Scientists use Automated EDA libraries to generate comprehensive HTML reports in 2 lines of code.

The most popular library is `ydata-profiling` (formerly `pandas-profiling`).

```python
# pip install ydata-profiling
from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv('your_dataset.csv')

# Generate the report
profile = ProfileReport(df, title="Dataset Exploratory Report", explorative=True)

# Save as interactive HTML file
profile.to_file("eda_report.html")
```

The generated report automatically calculates:
- Missing value percentages
- Mean, Median, Min, Max
- Skewness and Kurtosis
- Histograms for every column
- Correlation matrices (Pearson, Spearman)
- High cardinality warnings

---

## 6. What's Next

EDA requires a solid understanding of statistical metrics. To truly understand distributions, variance, and correlations, we need to dive deep into Statistics.

| Next Topic | Why |
|------------|-----|
| [Descriptive Statistics](./06-Descriptive-Statistics.md) | Understand the mathematics behind the mean, median, standard deviation, and skewness that we just visualized. |

---

[← Previous: Data Cleaning](./04-Data-Cleaning.md) | [Back to Main Index](../README.md) | [Next: Descriptive Statistics →](./06-Descriptive-Statistics.md)
