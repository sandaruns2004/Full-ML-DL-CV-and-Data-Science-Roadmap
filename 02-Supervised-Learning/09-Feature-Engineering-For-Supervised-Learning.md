# 🛠️ Feature Engineering for Supervised Learning

> **Prerequisites**: Data Science Foundations | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

*"Applied machine learning is basically feature engineering."* — Andrew Ng

Algorithms are important, but the quality of the data you feed into them usually dictates the ceiling of your model's performance. 

---

## 1. Introduction

### What is Feature Engineering?
Feature Engineering is the process of using domain knowledge to extract new variables (features) from raw data, or to transform existing features into formats that make it easier for machine learning algorithms to understand the underlying patterns.

### When to use it
- Always. Every real-world machine learning project requires some level of feature engineering.
- When your model is underfitting and needs more complex information.
- When you have non-numerical data (text, dates, categorical strings) that must be fed into mathematical algorithms.

### When NOT to use it
- When utilizing Deep Learning (Neural Networks are designed to do automated feature extraction from raw data, like pixels or audio waves).

---

## 2. Intuition

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

## 3. Mathematical Foundations

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

## 4. General Workflow

1. **Exploratory Data Analysis (EDA)**: Plot histograms and correlations to understand what the data looks like.
2. **Imputation**: Handle missing values (fill with mean/median, or use advanced KNN imputers).
3. **Encoding Categoricals**: Convert text categories into numbers (One-Hot Encoding, Label Encoding).
4. **Feature Creation**: Combine features (e.g., `Total_Rooms / Total_Bedrooms = Rooms_per_Bedroom`) or extract dates.
5. **Scaling**: Standardize numerical features.
6. **Feature Selection**: Drop redundant, highly correlated, or useless features to prevent the Curse of Dimensionality.

---

## 5. Advantages

- **Massive Performance Gains**: Good features can make a simple Linear Regression outperform a complex Random Forest trained on bad features.
- **Injects Domain Knowledge**: Allows human experts to encode their business logic directly into the model.
- **Simplifies Models**: A well-engineered feature can reduce a model's complexity, making it faster and less prone to overfitting.

---

## 6. Limitations

- **Time Consuming**: It is an iterative, manual, and often tedious process that can take up 80% of a data scientist's time.
- **Data Leakage Risk**: If you scale your data *before* doing a Train/Test split, information from the test set (like the global mean) "leaks" into the training set, invalidating your results.
- **Dimensionality Explosion**: Over-using One-Hot Encoding on a column with 10,000 unique ZIP codes will create 10,000 new sparse columns, destroying model efficiency.

---

## 7. Hyperparameters (Techniques)

Feature engineering involves choosing the right *techniques* rather than tuning hyperparameters:

- **Missing Values**: Mean, Median, Mode, or Constant value (`-999`).
- **Categoricals**: 
  - *Nominal* (no order, e.g., Color): One-Hot Encoding.
  - *Ordinal* (ordered, e.g., Size S/M/L): Ordinal Encoding (1, 2, 3).
  - *High Cardinality* (e.g., ZIP codes): Target Encoding or Frequency Encoding.
- **Binning**: Converting a continuous variable (Age: 34) into a categorical bin (Age_Group: 30-40).

---

## 8. Industry Applications

- **Banking/Fraud**: Creating velocity features (e.g., "Number of transactions in the last 10 minutes") is the primary way fraud models detect stolen cards.
- **Retail**: Creating features like "Days since last purchase" or "Average basket size" to predict customer churn.
- **NLP**: Converting raw text documents into TF-IDF numerical matrices to feed into Naive Bayes or SVM classifiers.

---

## 9. Interview Preparation

### Beginner Questions
**Q: What is One-Hot Encoding?**
> A: It is a process of converting categorical text variables into binary (1 or 0) columns. For a column `Color` with values `Red, Blue`, it creates two new columns `Is_Red` and `Is_Blue`.

**Q: Why do we need to scale numerical data?**
> A: Algorithms like KNN or SVM calculate geometric distances. If `Age` ranges from 0-100 and `Income` ranges from 0-100,000, the algorithm will mathematically treat `Income` as 1,000 times more important than `Age` simply because the numbers are bigger. Scaling puts them on a level playing field.

### Intermediate Questions
**Q: What is Data Leakage in the context of Feature Engineering?**
> A: Data Leakage occurs when you use information from outside the training dataset to create the model. The most common mistake is applying `StandardScaler.fit()` to the *entire* dataset before splitting. The scaler "sees" the test data's mean and variance, which gives the model an unfair advantage and results in overly optimistic test scores. You must `fit()` only on the training data, and `transform()` the test data.

**Q: When should you use Min-Max scaling vs. Standardization (Z-score)?**
> A: Use Min-Max scaling when the algorithm strictly requires bounds (e.g., Neural Networks expecting 0-1 inputs) or when the data doesn't have extreme outliers. Use Standardization when the data has outliers or follows a Gaussian distribution, as it is more robust to extreme values.

### Advanced Questions
**Q: Explain Target Encoding and its primary danger.**
> A: Target encoding replaces a categorical value (e.g., "City") with the mean of the target variable for that category (e.g., the average house price in that City). The danger is severe overfitting (Target Leakage), especially for categories with very few samples. It must be implemented using smoothing techniques and cross-validation folds.

---

## 10. Exercises

### Easy
- **Encoding**: Use Pandas `pd.get_dummies()` to One-Hot Encode a dataframe containing a "Department" column (Sales, Engineering, HR).

### Medium
- **Date Extraction**: Load a dataset with a standard "Date" column. Write a function that extracts the Year, Month, Day of Week, and a boolean `Is_Weekend` column.

### Hard
- **Custom Transformer**: Write a custom Scikit-Learn Transformer class (inheriting from `BaseEstimator, TransformerMixin`) that automatically finds highly skewed numerical columns and applies a log transformation to them.

---

## 11. Further Reading

### Books
- *Feature Engineering for Machine Learning* by Alice Zheng & Amanda Casari
- *Hands-On Machine Learning with Scikit-Learn* by Aurélien Géron (Chapter 2 - Data Preparation)

### Documentation
- [Scikit-Learn Preprocessing API](https://scikit-learn.org/stable/modules/preprocessing.html)

---

[← Naive Bayes](08-Naive-Bayes.md) | [Back to Index](../README.md) | [Next: Regularization →](10-Regularization.md)
