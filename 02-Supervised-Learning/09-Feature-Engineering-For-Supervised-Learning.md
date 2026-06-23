# 🛠️ Feature Engineering for Supervised Learning

> **Prerequisites:** Data Science Foundations
>
> **Difficulty:** ⭐⭐⭐☆☆
>
> **Estimated Reading Time:** 15 minutes

*"Applied machine learning is basically feature engineering."* — Andrew Ng

Algorithms are important, but the quality of the data you feed into them usually dictates the ceiling of your model's performance. 

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [General Workflow](#4-general-workflow)
5. [Advantages & Limitations](#5-advantages--limitations)
6. [Hyperparameters (Techniques)](#6-hyperparameters-techniques)
7. [Industry Applications](#7-industry-applications)
8. [Exercises](#8-exercises)

---

# 1. What Problem Does This Solve?

### 🟢 Beginner
Algorithms are important, but they cannot understand raw data like text, timestamps, or categories natively. Feature Engineering solves this problem by translating raw data into clear, mathematical numbers that machine learning algorithms can easily understand and learn from.

### 🟡 Intermediate
Feature Engineering is the process of using domain knowledge to extract new variables (features) from raw data, or to transform existing features. You use it in almost every real-world project, particularly when your model is underfitting or when handling non-numerical data (text, dates, strings). You generally do *not* need it as heavily when utilizing Deep Learning, since Neural Networks are designed to do automated feature extraction from raw data (pixels, audio waves).

### 🔴 Advanced
The quality of the features you feed into your algorithms dictates the ceiling of your model's performance. Proper feature engineering maps raw inputs into an optimal feature space where the underlying patterns are linearly separable (or easily split by trees), significantly reducing the complexity required by the estimator and improving generalization.

---

# 2. Intuition

### Real-World Example
Imagine trying to predict if a flight will be delayed. Your raw dataset gives you `Departure_Time = 2023-11-23 18:45:00`.
A linear regression algorithm looks at that string of text and sees absolute garbage. It cannot do math on a timestamp.

**Feature Engineering** transforms that single column into:
- `Is_Weekend`: 0 (Thursday)
- `Hour_of_Day`: 18
- `Is_Holiday_Week`: 1 (Thanksgiving)
- `Time_block`: "Evening Rush"

Suddenly, the algorithm can clearly see that "Evening Rush" flights during "Holiday Weeks" have a massive correlation with delays. You didn't change the algorithm; you just translated the data into a language it understands.

---

# 3. Mathematical Foundations

### Scaling and Normalization
Algorithms based on distance (KNN, SVM) or gradient descent (Linear/Logistic Regression) are highly sensitive to the scale of the data. 

**Standardization (Z-Score Normalization)**:
Rescales data to have a mean ($\mu$) of 0 and standard deviation ($\sigma$) of 1.
$$ z = \frac{x - \mu}{\sigma} $$

**Min-Max Scaling (Normalization)**:
Rescales data to fit exactly between 0 and 1.
$$ x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}} $$

### Log Transforms
If a feature is highly skewed (e.g., Income, where most make \$50k but a few make \$5M), it ruins linear models. Applying a natural logarithm "squashes" the long tail, making the distribution closer to a Normal distribution.
$$ x_{new} = \log(x + 1) $$

---

# 4. General Workflow

1. **Exploratory Data Analysis (EDA)**: Plot histograms and correlations to understand what the data looks like.
2. **Imputation**: Handle missing values (fill with mean/median, or use advanced KNN imputers).
3. **Encoding Categoricals**: Convert text categories into numbers (One-Hot Encoding, Label Encoding).
4. **Feature Creation**: Combine features (e.g., `Total_Rooms / Total_Bedrooms = Rooms_per_Bedroom`) or extract dates.
5. **Scaling**: Standardize numerical features.
6. **Feature Selection**: Drop redundant, highly correlated, or useless features to prevent the Curse of Dimensionality.

---

# 5. Advantages & Limitations

### Advantages
- **Massive Performance Gains**: Good features can make a simple Linear Regression outperform a complex Random Forest trained on bad features.
- **Injects Domain Knowledge**: Allows human experts to encode their business logic directly into the model.
- **Simplifies Models**: A well-engineered feature can reduce a model's complexity, making it faster and less prone to overfitting.

### Limitations
- **Time Consuming**: It is an iterative, manual, and often tedious process that can take up 80% of a data scientist's time.
- **Data Leakage Risk**: If you scale your data *before* doing a Train/Test split, information from the test set (like the global mean) "leaks" into the training set, invalidating your results.
- **Dimensionality Explosion**: Over-using One-Hot Encoding on a column with 10,000 unique ZIP codes will create 10,000 new sparse columns, destroying model efficiency.

---

# 6. Hyperparameters (Techniques)

Feature engineering involves choosing the right *techniques* rather than tuning hyperparameters:

- **Missing Values**: Mean, Median, Mode, or Constant value (`-999`).
- **Categoricals**: 
  - *Nominal* (no order, e.g., Color): One-Hot Encoding.
  - *Ordinal* (ordered, e.g., Size S/M/L): Ordinal Encoding (1, 2, 3).
  - *High Cardinality* (e.g., ZIP codes): Target Encoding or Frequency Encoding.
- **Binning**: Converting a continuous variable (Age: 34) into a categorical bin (Age_Group: 30-40).

---

# 7. Industry Applications

- **Banking/Fraud**: Creating velocity features (e.g., "Number of transactions in the last 10 minutes") is the primary way fraud models detect stolen cards.
- **Retail**: Creating features like "Days since last purchase" or "Average basket size" to predict customer churn.
- **NLP**: Converting raw text documents into TF-IDF numerical matrices to feed into Naive Bayes or SVM classifiers.

---

# 8. Exercises

### Easy
- **Encoding**: Use Pandas `pd.get_dummies()` to One-Hot Encode a dataframe containing a "Department" column (Sales, Engineering, HR).

### Medium
- **Date Extraction**: Load a dataset with a standard "Date" column. Write a function that extracts the Year, Month, Day of Week, and a boolean `Is_Weekend` column.

### Hard
- **Custom Transformer**: Write a custom Scikit-Learn Transformer class (inheriting from `BaseEstimator, TransformerMixin`) that automatically finds highly skewed numerical columns and applies a log transformation to them.

---

[← Naive Bayes](08-Naive-Bayes.md) | [Back to Index](../README.md) | [Next: Regularization →](10-Regularization.md)
