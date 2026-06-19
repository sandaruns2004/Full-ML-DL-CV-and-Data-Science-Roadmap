# ⚙️ Feature Engineering

> **Prerequisites**: Data Preprocessing | **Difficulty**: ⭐⭐☆☆☆ Elementary

---

## 📋 Table of Contents

1. [What Is Feature Engineering?](#1-what-is-feature-engineering)
2. [Feature Creation (Manual)](#2-feature-creation-manual)
3. [Feature Transformation](#3-feature-transformation)
4. [Feature Selection](#4-feature-selection)
5. [Feature Importance](#5-feature-importance)
6. [Domain-Specific Features](#6-domain-specific-features)
7. [Automated Feature Engineering (Featuretools)](#7-automated-feature-engineering-featuretools)
8. [Feature Stores (Feast)](#8-feature-stores-feast)
9. [Project Ideas](#9-project-ideas)
10. [What's Next](#10-whats-next)

---

## 1. What Is Feature Engineering?

> **"Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering."** — Andrew Ng

Feature engineering is the process of using **domain knowledge** to create, transform, and select features (input variables) that make ML algorithms work better. While Deep Learning can sometimes automate this, traditional ML (like Random Forests and XGBoost) relies heavily on high-quality features.

**The Hierarchy of Impact**:
```
More Data               ████████████████████████████ (Highest impact)
Feature Engineering     ██████████████████████ 
Algorithm Choice        ████████████████
Hyperparameter Tuning   ████████████
Ensembling              ████████
```

---

## 2. Feature Creation (Manual)

### 2.1 Mathematical Combinations

Creating new interactions between existing features.

```python
import pandas as pd
import numpy as np

# House price dataset
df = pd.DataFrame({
    'length': [30, 40, 25],
    'width': [20, 25, 15],
    'floors': [1, 2, 1],
    'bedrooms': [3, 4, 2],
    'bathrooms': [2, 3, 1],
    'price': [200000, 350000, 150000]
})

# Create new features from existing ones
df['area'] = df['length'] * df['width']                    # Interaction
df['total_area'] = df['area'] * df['floors']               # Multi-feature interaction
df['price_per_sqft'] = df['price'] / df['total_area']      # Ratio
df['rooms_ratio'] = df['bedrooms'] / df['bathrooms']       # Ratio
```

### 2.2 DateTime Features

Dates and times are useless as raw strings. They must be parsed into cyclical or categorical numbers.

```python
dates = pd.date_range('2023-01-01', periods=365, freq='D')
df = pd.DataFrame({'date': dates})

# Extract components
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek      # 0=Monday, 6=Sunday
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding (Because December 31 is close to January 1, but 12 and 1 are far numerically)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

### 2.3 Text & NLP Features

If you aren't using a massive Transformer model, simple text features still add huge value.

```python
df = pd.DataFrame({'review': ['This is amazing!', 'Terrible. Waste of money.']})

df['word_count'] = df['review'].str.split().str.len()
df['has_exclamation'] = df['review'].str.contains('!').astype(int)
df['uppercase_ratio'] = df['review'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x))
```

---

## 3. Feature Transformation

### 3.1 Unskewing Data (Log, Sqrt, Box-Cox)

Linear regression and Neural Networks perform better when variables are normally distributed. If a feature has a heavy right tail (like Income), transform it.

```python
import numpy as np
from scipy import stats

data = np.random.exponential(5, 1000)  # Heavily skewed data

log_transformed = np.log1p(data)       # Log(x + 1)
boxcox_transformed, _ = stats.boxcox(data + 0.01) # Automatic optimal transformation
```

### 3.2 Polynomial Features

Allows a linear model to capture non-linear relationships.
$\text{For features } [x_1, x_2]: \text{degree 2} \rightarrow [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]$

```python
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[2, 3], [4, 5]])
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

### 3.3 Binning (Discretization)

Converting continuous variables into categorical bins. (e.g., Age $\rightarrow$ Age Group).

```python
ages = pd.Series([18, 22, 25, 30, 35, 42, 48, 55, 62, 70])

# Equal-width binning
age_bins = pd.cut(ages, bins=[0, 25, 45, 65, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])

# Quantile binning (Equal-frequency)
age_quantiles = pd.qcut(ages, q=3, labels=['Low', 'Medium', 'High'])
```

---

## 4. Feature Selection

Adding too many features causes **the curse of dimensionality**, leading to overfitting. We must select only the best ones.

### 4.1 Filter Methods (Statistical)
Evaluate features independently of the ML model. Fast, but misses feature interactions.

```python
from sklearn.feature_selection import SelectKBest, f_classif

# ANOVA F-test (Select top 10 features)
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

### 4.2 Wrapper Methods (Iterative)
Train a model iteratively, adding or removing features. Very accurate but computationally expensive.

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)
selected_features = [f for f, s in zip(feature_names, rfe.support_) if s]
```

### 4.3 Embedded Methods (Model-Based)
The algorithm itself performs feature selection during training. (e.g., Lasso Regression drives useless feature weights exactly to 0).

```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5)
lasso.fit(X, y)

# Features with a coefficient of 0 were dropped by Lasso
important_features = np.where(lasso.coef_ != 0)[0]
```

---

## 5. Feature Importance

How do we explain our model to stakeholders?

### 5.1 Tree-Based Importance (Impurity)
Fast, but biased towards high-cardinality features (features with many unique values).

```python
rf = RandomForestClassifier()
rf.fit(X, y)
importances = rf.feature_importances_
```

### 5.2 Permutation Importance (The Gold Standard)
Measures the drop in model accuracy when a single feature is randomly shuffled. If accuracy drops massively, the feature was critical. Evaluated on the *test set*.

```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()[::-1]
```

---

## 6. Domain-Specific Features

Different industries rely on completely different core features.

### E-commerce
```python
df['total_spent'] = df['order_count'] * df['avg_order_value']
df['days_since_last_order'] = (pd.Timestamp.now() - df['last_order_date']).dt.days
```

### Finance (Credit Risk)
```python
df['debt_to_income'] = df['total_debt'] / df['annual_income']
df['credit_utilization'] = df['credit_used'] / df['credit_limit']
```

---

## 7. Automated Feature Engineering (Featuretools)

Manually writing code for hundreds of mathematical combinations is tedious. **Featuretools** automates this by performing Deep Feature Synthesis (DFS) across relational databases.

```python
# pip install featuretools
import featuretools as ft

# Assume we have Customers, Sessions, and Transactions dataframes
# We create an EntitySet
es = ft.EntitySet(id="customer_data")

# Add dataframes to the entity set
es = es.add_dataframe(dataframe_name="customers", dataframe=customers_df, index="customer_id")
es = es.add_dataframe(dataframe_name="transactions", dataframe=transactions_df, index="transaction_id")

# Define relationships (One customer -> Many transactions)
es = es.add_relationship("customers", "customer_id", "transactions", "customer_id")

# Automatically generate features!
# Generates things like: SUM(transactions.amount), MAX(transactions.amount), DAY(transactions.date)
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    max_depth=2  # How deep to stack operations (e.g., MEAN of the MAX of...)
)
```

---

## 8. Feature Stores (Feast)

In large organizations, Data Scientist A might write an SQL query to calculate a customer's `30_day_rolling_spend`. Data Scientist B might write a slightly different query for the same concept. 

When deploying to production, the engineering team has to rewrite the Pandas code in Java or Go. This causes **Training-Serving Skew**.

### What is a Feature Store?
A centralized hub for organizing, storing, and serving machine learning features. 

**Popular Tool**: **Feast** (Feature Store)
- **Offline Store**: (e.g., Snowflake, BigQuery) Used for training massive batches of data.
- **Online Store**: (e.g., Redis, DynamoDB) Used for serving features at ultra-low latency (<10ms) during real-time inference.

```python
# Conceptual Feast Implementation
# You define features in a central repository (YAML/Python)

from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Data Scientist fetches historical features for training
training_df = store.get_historical_features(
    entity_df=customer_ids_and_timestamps,
    features=[
        "customer_hourly_stats:conv_rate",
        "customer_hourly_stats:total_spend"
    ],
).to_df()

# ML Engineer fetches the EXACT SAME feature definition from the online store for real-time predictions
online_features = store.get_online_features(
    features=[
        "customer_hourly_stats:conv_rate",
        "customer_hourly_stats:total_spend"
    ],
    entity_rows=[{"customer_id": 1001}]
).to_dict()
```

---

## 9. Project Ideas

### 🟢 Project 1: Feature Engineering on Titanic (Beginner)
Create 20+ features from the Titanic dataset (Title from Name, Family Size, Cabin Prefix). See which feature improves a basic Random Forest the most.

### 🟡 Project 2: Auto Feature Generator (Intermediate)
Build a wrapper around `Featuretools` that automatically takes any Pandas DataFrame, generates interaction features, un-skews continuous variables, and returns a fully engineered matrix.

### 🔴 Project 3: Mini Feature Store (Advanced)
Set up a local `Feast` repository. Connect it to a local SQLite database (Offline) and a local Redis container (Online). Train a model using the offline store, and write a FastAPI endpoint that queries the online store for real-time predictions.

---

## 10. What's Next

| Next Topic | Why |
|------------|-----|
| [Machine Learning Fundamentals](../02-Supervised-Learning/README.md) | Time to start actually building predictive models. |
| [Linear Regression](../02-Supervised-Learning/01-Linear-Regression.md) | Apply your features to a real, interpretable model. |

---

[← Data Preprocessing](./06-Data-Preprocessing.md) | [Back to Index](../README.md) | [Next: Data Visualization Mastery →](./08-Data-Visualization-Mastery.md)
