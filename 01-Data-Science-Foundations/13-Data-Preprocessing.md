# ⚙️ Data Preprocessing for Machine Learning

> **Prerequisites**: [Data Cleaning](./06-Data-Cleaning.md) | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 📋 Table of Contents

1. [Cleaning vs. Preprocessing](#1-cleaning-vs-preprocessing)
2. [Handling Categorical Data (Encoding)](#2-handling-categorical-data-encoding)
3. [Feature Scaling (Normalization & Standardization)](#3-feature-scaling-normalization--standardization)
4. [Handling Date and Time Features](#4-handling-date-and-time-features)
5. [The ML Pipeline (Preventing Data Leakage)](#5-the-ml-pipeline-preventing-data-leakage)
6. [What's Next](#6-whats-next)

---

## 1. Cleaning vs. Preprocessing

### 🟢 Beginner

In **Data Cleaning**, we fixed human errors (missing values, typos, duplicates) so the data was accurate.
In **Data Preprocessing**, we transform that clean data into a format that a Machine Learning algorithm can physically read.

Machine Learning models (like Linear Regression or Neural Networks) are just math equations. **They cannot read text.** If you feed the word "New York" into a math equation, it crashes. We must translate everything into numbers.

---

## 2. Handling Categorical Data (Encoding)

### 🟡 Intermediate

Categorical data (text labels) comes in two forms, and we treat them differently.

**1. Nominal Data (No inherent order)**
Examples: City (NY, LA, SF), Color (Red, Blue, Green).
We cannot simply map Red=1, Blue=2, Green=3. The model will think Green is "greater" than Red, which makes no sense.
Instead, we use **One-Hot Encoding** (creating dummy variables).

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({'City': ['NY', 'LA', 'SF', 'NY']})

# Pandas method (Quick and easy)
df_dummies = pd.get_dummies(df, columns=['City'], drop_first=True)
# drop_first=True prevents "The Dummy Variable Trap" (Multicollinearity)

# Scikit-Learn method (Better for ML pipelines)
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cities = encoder.fit_transform(df[['City']])
```

**2. Ordinal Data (Inherent order)**
Examples: Size (Small, Medium, Large), Education (High School, Bachelors, Masters).
Because there is a mathematical hierarchy (Large > Medium > Small), we use **Ordinal Encoding** or **Label Encoding**.

```python
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({'Size': ['Small', 'Large', 'Medium', 'Small']})

# We must specify the order!
sizes = [['Small', 'Medium', 'Large']]
encoder = OrdinalEncoder(categories=sizes)

df['Size_Encoded'] = encoder.fit_transform(df[['Size']])
# Output: Small=0.0, Medium=1.0, Large=2.0
```

---

## 3. Feature Scaling (Normalization & Standardization)

### 🟡 Intermediate

Imagine a dataset predicting house prices with two features:
- **Number of Bedrooms**: 1 to 5
- **Square Footage**: 500 to 5,000

If we feed this into a distance-based algorithm (like K-Nearest Neighbors or K-Means), the model will think Square Footage is 1,000 times more important than Bedrooms simply because the numbers are bigger. We must scale them so they carry equal weight!

**1. Standardization (Z-Score Scaling)**
Centers data around 0 with a standard deviation of 1.
*Best for: Algorithms assuming normal distributions (Linear Regression, Logistic Regression, SVM).*

```python
from sklearn.preprocessing import StandardScaler

# Formula: (X - Mean) / Standard Deviation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Bedrooms', 'SqFt']])
```

**2. Normalization (Min-Max Scaling)**
Squashes all data into a fixed range, usually 0 to 1.
*Best for: Neural Networks, Image processing (pixel values 0-255).*

```python
from sklearn.preprocessing import MinMaxScaler

# Formula: (X - Min) / (Max - Min)
scaler = MinMaxScaler(feature_range=(0, 1))
df_normalized = scaler.fit_transform(df[['Bedrooms', 'SqFt']])
```

> **Rule of Thumb**: Tree-based models (Random Forest, XGBoost) do NOT require feature scaling. Almost everything else does.

---

## 4. Handling Date and Time Features

### 🟡 Intermediate

A datetime string like `"2023-12-25 08:30:00"` is useless to an ML model. We must extract numerical features from it.

```python
import pandas as pd

df = pd.DataFrame({'Timestamp': ['2023-12-25 08:30:00', '2023-07-04 20:15:00']})
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extracting temporal features
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday

# Extracting categorical/binary features
df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df['Is_Morning'] = df['Hour'].apply(lambda x: 1 if 6 <= x < 12 else 0)

# Now drop the original Timestamp column!
df = df.drop('Timestamp', axis=1)
```

---

## 5. The ML Pipeline (Preventing Data Leakage)

### 🔴 Advanced

**Data Leakage** is the #1 mistake junior Data Scientists make. It occurs when information from outside the training dataset is used to create the model.

**The Mistake:**
1. You have a dataset of 1,000 rows.
2. You run `StandardScaler().fit_transform()` on all 1,000 rows. (The scaler calculates the Mean of all 1,000 rows).
3. You split the data: 800 for training, 200 for testing.
4. *ERROR!* Your training data has been scaled using the Mean of the testing data. The test data has "leaked" into the training process!

**The Fix: Scikit-Learn Pipelines**
Always split your data FIRST. Then, `fit()` your preprocessing on the training data ONLY, and `transform()` the test data.

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. SPLIT FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Define the Pipeline steps
# The pipeline acts as a single object that handles all preprocessing and modeling
ml_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Fill missing values
    ('scaler', StandardScaler()),                  # Scale the data
    ('model', LogisticRegression())                # The ML algorithm
])

# 3. Fit on Training Data
# This calculates the median, the mean, the std dev, AND trains the model
ml_pipeline.fit(X_train, y_train)

# 4. Predict on Test Data
# This applies the exact same median/mean/std dev from the training data to the test data
predictions = ml_pipeline.predict(X_test)
```

Pipelines guarantee that you will never accidentally leak data, and they make deploying your model to production infinitely easier.

---

## 6. What's Next

Preprocessing gets the data ready for the model. But to get *great* performance, we need to creatively construct entirely new features that the original dataset didn't have.

| Next Topic | Why |
|------------|-----|
| [Feature Engineering](./14-Feature-Engineering.md) | Learn how to combine and transform basic columns into high-signal features that drastically improve model accuracy. |

---

[← Data Visualization](12-Data-Visualization.md) | [Back to Index](../README.md) | [Next: Feature Engineering →](14-Feature-Engineering.md)
