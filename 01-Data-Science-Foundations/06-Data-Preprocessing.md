# 🧹 Data Preprocessing

> **Prerequisites**: Exploratory Data Analysis | **Difficulty**: ⭐⭐☆☆☆ Elementary

---

## 📋 Table of Contents

1. [Why Preprocess Data?](#1-why-preprocess-data)
2. [Handling Missing Values](#2-handling-missing-values)
3. [Handling Outliers](#3-handling-outliers)
4. [Encoding Categorical Variables](#4-encoding-categorical-variables)
5. [Feature Scaling](#5-feature-scaling)
6. [Handling Imbalanced Data](#6-handling-imbalanced-data)
7. [The Scikit-Learn Pipeline](#7-the-scikit-learn-pipeline)
8. [Data Leakage (Crucial Concept)](#8-data-leakage-crucial-concept)
9. [Project Ideas](#9-project-ideas)
10. [What's Next](#10-whats-next)

---

## 1. Why Preprocess Data?

> **"Garbage In, Garbage Out."**

Machine Learning models are essentially math equations. They do not understand missing values (`NaN`), text strings like `"Red"`, or the fact that income is measured in dollars (0 to 1,000,000) while age is measured in years (0 to 100). 

**Data Preprocessing** is the pipeline of transforming raw data into a clean, standardized numeric format that an algorithm can learn from effectively.

---

## 2. Handling Missing Values

Missing data is a reality in almost all real-world datasets. 

### Types of Missing Data
1. **MCAR (Missing Completely At Random)**: The missingness has nothing to do with the data (e.g., a sensor briefly lost power).
2. **MAR (Missing At Random)**: Missingness depends on another observed variable (e.g., Men are less likely to fill out the "depression scale" survey).
3. **MNAR (Missing Not At Random)**: Missingness depends on the missing value itself (e.g., extremely wealthy people leave "Income" blank).

### Strategies

| Strategy | When to use | Pros | Cons |
|----------|-------------|------|------|
| **Drop Rows** | < 5% of data is missing, MCAR | Easy, preserves distribution | Loss of information |
| **Drop Column** | > 60% of data is missing | Removes useless feature | Might lose a weak signal |
| **Mean/Median Imputation** | Small amount of missing numeric data | Simple, fast | Reduces variance, distorts correlations |
| **Mode Imputation** | Missing categorical data | Simple | Can heavily skew categories |
| **Predictive Imputation (KNN/Iterative)** | Missing data has relationships with other columns | Highly accurate | Computationally expensive |
| **Add "Is_Missing" Indicator** | The missingness itself is informative | Captures MNAR | Adds dimensionality |

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50],
    'income': [50k, 60k, 80k, np.nan, 120k],
    'city': ['NY', 'LA', 'NY', np.nan, 'SF']
})

# 1. Simple Imputation (Mean for numeric)
num_imputer = SimpleImputer(strategy='mean')
df['age_imputed'] = num_imputer.fit_transform(df[['age']])

# 2. Simple Imputation (Constant/Mode for categorical)
cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
df['city_imputed'] = cat_imputer.fit_transform(df[['city']])

# 3. Advanced: KNN Imputer (Uses nearest neighbors)
knn_imputer = KNNImputer(n_neighbors=2)
df[['age_knn', 'income_knn']] = knn_imputer.fit_transform(df[['age', 'income']])
```

---

## 3. Handling Outliers

Outliers can severely skew models like Linear Regression and K-Means. (Tree-based models are robust to outliers).

### Strategies
1. **Drop them**: If they are clear data entry errors (e.g., Age = 999).
2. **Cap/Clip them (Winsorization)**: Limit extreme values to a specific percentile (e.g., 99th percentile).
3. **Transform them**: Applying a Log or Square Root transform often pulls outliers in.

```python
# Capping Outliers (Winsorization) using Pandas
lower_limit = df['income'].quantile(0.01)
upper_limit = df['income'].quantile(0.99)

df['income_clipped'] = df['income'].clip(lower=lower_limit, upper=upper_limit)
```

---

## 4. Encoding Categorical Variables

Algorithms need numbers. We must convert text categories into numeric representations.

### 4.1 Ordinal Encoding
Used when there is an **inherent order** in the categories (e.g., Low < Medium < High).

```python
from sklearn.preprocessing import OrdinalEncoder

sizes = pd.DataFrame({'size': ['Small', 'Large', 'Medium', 'Small']})
# Explicitly define the order!
encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
encoded = encoder.fit_transform(sizes) # 0, 2, 1, 0
```

### 4.2 One-Hot Encoding (OHE)
Used for **nominal** variables with NO order (e.g., Red, Green, Blue). Creates a binary column for each category.

```python
from sklearn.preprocessing import OneHotEncoder

colors = pd.DataFrame({'color': ['Red', 'Green', 'Blue']})
# drop='first' prevents perfect collinearity (Dummy Variable Trap)
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(colors)
```

### 4.3 Target Encoding
Used for high-cardinality categorical variables (e.g., Zip Codes). Replaces the category with the mean of the target variable for that category. *Must be done carefully to avoid data leakage!*

---

## 5. Feature Scaling

Distance-based algorithms (KNN, SVM, K-Means) and Gradient Descent-based algorithms (Linear Regression, Neural Networks) are highly sensitive to the scale of features. (Tree-based algorithms are NOT).

### 5.1 Min-Max Scaling (Normalization)
Scales data to a fixed range, usually [0, 1].
- **Formula**: $X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$
- **When to use**: Image pixels (0-255), Neural Networks, or when you know the strict upper and lower bounds.

### 5.2 Standardization (Z-Score Scaling)
Centers the data around a mean of 0 with a standard deviation of 1.
- **Formula**: $X_{std} = \frac{X - \mu}{\sigma}$
- **When to use**: Most ML algorithms (SVM, Logistic Regression, PCA). It handles outliers better than Min-Max.

### 5.3 Robust Scaling
Uses the median and Interquartile Range (IQR).
- **When to use**: When your data contains massive outliers that would ruin the mean/std calculations of Standard Scaler.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = pd.DataFrame({'income': [30000, 45000, 50000, 1000000]}) # Massive outlier

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

print(std_scaler.fit_transform(data))
print(minmax_scaler.fit_transform(data))
print(robust_scaler.fit_transform(data)) # Best for this outlier!
```

---

## 6. Handling Imbalanced Data

If you are predicting Fraud, and only 0.1% of transactions are fraud, a model that predicts "Not Fraud" 100% of the time is 99.9% accurate, but entirely useless.

### Strategies
1. **Undersampling**: Randomly remove examples from the majority class. (Loses data).
2. **Oversampling (SMOTE)**: Synthetically generate new examples of the minority class using K-Nearest Neighbors.
3. **Class Weights**: Tell the algorithm to penalize errors on the minority class heavily.

```python
from imblearn.over_sampling import SMOTE

# X_train, y_train have extreme imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## 7. The Scikit-Learn Pipeline

Applying all these steps manually to training data, and then remembering to apply the EXACT SAME steps to the test data, is a nightmare and highly prone to error.

Scikit-Learn `Pipeline` and `ColumnTransformer` bundle preprocessing and modeling together.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Define column groups
numeric_features = ['age', 'income', 'balance']
categorical_features = ['city', 'education']

# 1. Create Preprocessing Steps for Numeric Data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Create Preprocessing Steps for Categorical Data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 3. Combine them using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Append the Classifier to the end
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# 5. Fit the ENTIRE pipeline at once
# clf.fit(X_train, y_train)

# 6. Predict on unseen data (automatically runs through the preprocessor!)
# predictions = clf.predict(X_test)
```

---

## 8. Data Leakage (Crucial Concept)

> **Data Leakage occurs when information from outside the training dataset is used to create the model.** This results in overly optimistic performance during training that completely fails in production.

### The Golden Rule of Preprocessing
**Fit ONLY on the Training Data. Transform BOTH Training and Test Data.**

❌ **WRONG (Data Leakage)**:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Fitting on the whole dataset!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

✅ **RIGHT**:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Learn the Mean/Std from Train ONLY
X_test_scaled = scaler.transform(X_test)       # Apply that Mean/Std to Test
```
*(Using the Scikit-Learn `Pipeline` prevents this automatically!)*

---

## 9. Project Ideas

### 🟢 Project 1: The Pipeline Builder (Beginner)
Take the Titanic dataset. Build a complete, robust Scikit-Learn `Pipeline` that handles missing ages via Median, missing Embarked via Mode, scales the Fare, One-Hot Encodes the Sex, and fits a Logistic Regression model.

### 🟡 Project 2: Fraud Detection Pipeline (Intermediate)
Find a highly imbalanced credit card fraud dataset. Build two pipelines: one without SMOTE and one with SMOTE (using `imblearn.pipeline.Pipeline`). Compare the Precision-Recall curves of both.

---

## 10. What's Next

| Next Topic | Why |
|------------|-----|
| [Feature Engineering](./07-Feature-Engineering.md) | We've cleaned the data. Now let's invent new, powerful features from it. |
| [Linear Regression](../02-Supervised-Learning/01-Linear-Regression.md) | The first mathematical model that relies heavily on proper scaling. |

---

[← Probability Distributions](./05-Probability-Distributions.md) | [Back to Index](../README.md) | [Next: Feature Engineering →](./07-Feature-Engineering.md)
