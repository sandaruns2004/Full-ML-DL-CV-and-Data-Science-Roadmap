# 📈 Descriptive Statistics

> **Prerequisites**: [Exploratory Data Analysis](./05-Exploratory-Data-Analysis.md) | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 📋 Table of Contents

1. [What are Descriptive Statistics?](#1-what-are-descriptive-statistics)
2. [Measures of Central Tendency](#2-measures-of-central-tendency)
3. [Measures of Dispersion (Spread)](#3-measures-of-dispersion-spread)
4. [Shape of Data: Skewness & Kurtosis](#4-shape-of-data-skewness--kurtosis)
5. [Percentiles & Quartiles](#5-percentiles--quartiles)
6. [What's Next](#6-whats-next)

---

## 1. What are Descriptive Statistics?

### 🟢 Beginner

If you have a dataset with 1,000,000 rows, you cannot look at every single number. **Descriptive Statistics** are mathematical summaries that describe the entire dataset using just a few numbers.

Descriptive statistics do exactly what they say: they *describe* the data you currently have. (They do NOT make predictions about future data — that is *Inferential* Statistics, which we cover later).

There are three main categories of descriptive statistics:
1. **Central Tendency**: Where is the "middle" of the data?
2. **Dispersion**: How spread out is the data?
3. **Shape**: Is the data symmetric or lopsided?

---

## 2. Measures of Central Tendency

### 🟢 Beginner

These metrics try to find the single "typical" value for a dataset.

| Metric | Definition | Python (Pandas) | When to use |
|--------|------------|-----------------|-------------|
| **Mean** | The mathematical average | `df['col'].mean()` | Symmetric data without outliers |
| **Median** | The exact middle value when sorted | `df['col'].median()` | Skewed data or data with outliers |
| **Mode** | The most frequent value | `df['col'].mode()[0]` | Categorical/text data |

### 🟡 Intermediate

**The problem with the Mean (Why we need the Median):**

Imagine a bar with 9 people. Each person makes $50,000 a year.
- **Mean Income**: $50,000
- **Median Income**: $50,000

Suddenly, Elon Musk walks into the bar. He makes $1,000,000,000 a year.
- **New Mean Income**: $100,045,000
- **New Median Income**: $50,000

*The Mean is highly sensitive to outliers. The Median is robust.* If your data has extreme outliers (like income, house prices, or viral video views), **always use the Median**.

---

## 3. Measures of Dispersion (Spread)

### 🟡 Intermediate

Central tendency alone is not enough. 
- Dataset A: [50, 50, 50, 50, 50] (Mean = 50)
- Dataset B: [0, 25, 50, 75, 100] (Mean = 50)

They have the same mean, but completely different spreads! We need metrics to quantify this spread.

**1. Range**
The simplest measure. Max value minus Min value.
```python
range_val = df['col'].max() - df['col'].min()
```

**2. Variance ($s^2$)**
The average of the *squared* differences from the Mean. Because it's squared, it gives heavy weight to outliers.
$$s^2 = \frac{\sum (x_i - \bar{x})^2}{n-1}$$
```python
variance = df['col'].var()
```

**3. Standard Deviation ($s$ or $\sigma$)**
The square root of Variance. We use this more often because it is in the *same units* as the original data. If you are measuring height in inches, variance is "inches squared" (useless), but standard deviation is "inches" (useful).
$$s = \sqrt{\text{Variance}}$$
```python
std_dev = df['col'].std()
```

---

## 4. Shape of Data: Skewness & Kurtosis

### 🔴 Advanced

**1. Skewness (Asymmetry)**

Skewness measures whether the data is pulled to one side.
- **Zero Skew**: Symmetrical (Mean = Median). Looks like a bell curve.
- **Positive (Right) Skew**: Tail pulls to the right (Mean > Median). Examples: Income, house prices.
- **Negative (Left) Skew**: Tail pulls to the left (Mean < Median). Examples: Age of retirement, exam scores where the test was easy.

```python
skewness = df['col'].skew()
# > 1 : Highly positive skew
# < -1 : Highly negative skew
# between -0.5 and 0.5 : Fairly symmetrical
```

**2. Kurtosis ("Tail Heaviness")**

Kurtosis measures how many extreme outliers are in the tails of the distribution compared to a normal bell curve.
- **Mesokurtic (Kurtosis ~ 3)**: Normal distribution.
- **Leptokurtic (Kurtosis > 3)**: "Fat tails." Lots of extreme outliers (highly unpredictable, common in financial stock returns).
- **Platykurtic (Kurtosis < 3)**: "Thin tails." Very few outliers, data is tightly grouped.

```python
# Pandas uses excess kurtosis (Fisher's definition), where normal = 0
kurt = df['col'].kurtosis() 
```

---

## 5. Percentiles & Quartiles

### 🟡 Intermediate

**Percentiles** tell you what percentage of data falls below a certain value.
If you score in the 90th percentile on a test, you scored better than 90% of the people.

**Quartiles** divide the data into 4 equal chunks (25% each):
- **Q1 (25th Percentile)**: Lower quartile
- **Q2 (50th Percentile)**: The Median!
- **Q3 (75th Percentile)**: Upper quartile

**Interquartile Range (IQR)**: The distance between Q3 and Q1. This represents the "middle 50%" of your data and is the standard metric used to detect outliers in Boxplots.
$$IQR = Q3 - Q1$$

```python
# Calculating IQR in Python
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1

# The standard formula for identifying an Outlier:
lower_outlier_fence = Q1 - 1.5 * IQR
upper_outlier_fence = Q3 + 1.5 * IQR
```

---

## 6. What's Next

Descriptive Statistics allow us to summarize data, but they heavily rely on understanding the *shape* of the data. The most important shape in all of statistics is the Normal Distribution.

| Next Topic | Why |
|------------|-----|
| [Probability & Distributions](./07-Probability-and-Distributions.md) | Master the Normal curve, Binomial distributions, and Bayes' Theorem. |

---

[← Previous: Exploratory Data Analysis](./05-Exploratory-Data-Analysis.md) | [Back to Main Index](../README.md) | [Next: Probability & Distributions →](./07-Probability-and-Distributions.md)
