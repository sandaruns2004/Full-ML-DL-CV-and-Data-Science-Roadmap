# 🏗️ Feature Engineering

> **Prerequisites**: [Data Preprocessing](./11-Data-Preprocessing.md) | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents

1. [What is Feature Engineering?](#1-what-is-feature-engineering)
2. [Polynomial and Interaction Features](#2-polynomial-and-interaction-features)
3. [Binning and Discretization](#3-binning-and-discretization)
4. [Mathematical Transformations](#4-mathematical-transformations)
5. [Text to Features (TF-IDF Basics)](#5-text-to-features-tf-idf-basics)
6. [What's Next](#6-whats-next)

---

## 1. What is Feature Engineering?

### 🟢 Beginner

> *"Applied machine learning is basically feature engineering."* — Andrew Ng

**Feature Engineering** is the art of using domain knowledge to create new variables (features) that make machine learning algorithms work better. 

While Deep Learning models can automatically extract features from raw data (like images or audio), classical Machine Learning (Random Forests, XGBoost, Linear Regression) working on tabular data relies heavily on human-engineered features.

**Example**: 
You have a dataset to predict if a loan will default. You have two columns: `Total_Debt` and `Income`.
Instead of feeding these raw numbers to the model, you create a new feature: `Debt_to_Income_Ratio = Total_Debt / Income`. 
This one new feature might be more predictive than the entire rest of the dataset combined.

---

## 2. Polynomial and Interaction Features

### 🟡 Intermediate

Sometimes the relationship between a feature and the target is not a straight line, or two features only become powerful when they interact.

**1. Polynomial Features**
If a relationship is curved (non-linear), linear models will fail. We can create polynomial features ($x^2, x^3$) to capture the curve.

```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.DataFrame({'Area': [500, 1000, 1500, 2000]})

# Create degree 2 polynomial features: (1, x, x^2)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['Area']])
# Result: [[500, 250000], [1000, 1000000], ...]
```

**2. Interaction Features**
When the effect of one feature depends on another feature.
*Example: Predicting house price. `Num_Bathrooms` is good. `Has_Pool` is good. `Num_Bathrooms * Has_Pool` might be an indicator of a luxury mansion.*

```python
df = pd.DataFrame({
    'Bedrooms': [2, 3, 4],
    'Bathrooms': [1, 2, 3]
})

# Manual interaction feature
df['Bed_Bath_Ratio'] = df['Bedrooms'] / df['Bathrooms']
df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']
```

---

## 3. Binning and Discretization

### 🟡 Intermediate

Sometimes, precise continuous numbers add "noise" to a model. We can group these numbers into discrete "bins" or "buckets."

*Example: Age. The difference between age 34 and 35 doesn't matter much for targeted marketing. But the difference between "Young Adult (18-35)" and "Middle Age (36-55)" is huge.*

```python
import pandas as pd

df = pd.DataFrame({'Age': [15, 22, 34, 45, 60, 72]})

# Define the bin edges and labels
bins = [0, 18, 35, 60, 100]
labels = ['Child', 'Young Adult', 'Middle Age', 'Senior']

# pd.cut creates bins of equal value ranges
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# pd.qcut creates bins containing an equal number of rows (Quantiles)
df['Age_Quantile'] = pd.qcut(df['Age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print(df[['Age', 'Age_Group']])
```

---

## 4. Mathematical Transformations

### 🔴 Advanced

Many ML models (especially Linear Regression, Logistic Regression, and Neural Networks) assume that your features follow a Normal (Gaussian) Distribution. If your data is heavily skewed (like Income), it degrades model performance.

We can apply mathematical transformations to un-skew the data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.DataFrame({'Income': np.random.exponential(scale=50000, size=1000)})

# 1. Log Transformation (Very common for financial/price data)
# Use log1p (log(1+x)) to handle zeros in the data safely
df['Log_Income'] = np.log1p(df['Income'])

# 2. Square Root Transformation (Good for count data)
df['Sqrt_Income'] = np.sqrt(df['Income'])

# 3. Box-Cox Transformation (The ultimate normalizer)
# Automatically finds the best power transformation. Data must be > 0.
df['BoxCox_Income'], _ = stats.boxcox(df['Income'] + 1)
```

> **Warning**: When you transform a feature, you lose interpretability. You can't easily explain to a business stakeholder that "for every 1 unit increase in the Box-Cox transformed log-income, sales go up by 2."

---

## 5. Text to Features (TF-IDF Basics)

### 🔴 Advanced

How do you feed a paragraph of text (like a product review) into an ML model? 
You extract numerical features from the text using **Natural Language Processing (NLP)** techniques.

**TF-IDF (Term Frequency - Inverse Document Frequency)**
- **TF**: How often does a word appear in this specific review?
- **IDF**: How rare is this word across ALL reviews?
- The combination gives a high score to words that are frequent in this document, but rare overall (finding the "keywords" of the document).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

reviews = [
    "The camera quality is amazing and battery life is great.",
    "Battery life is terrible, it died in one hour.",
    "Great camera, but the phone is too expensive."
]

# Initialize TF-IDF (removing common words like "is", "the", "and")
tfidf = TfidfVectorizer(stop_words='english')

# Transform text into a matrix of numbers
tfidf_matrix = tfidf.fit_transform(reviews)

# Look at the engineered features
feature_names = tfidf.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# The model can now use these numerical scores to predict sentiment!
print(df_tfidf)
```

---

## 6. What's Next

By generating interaction features, polynomials, and TF-IDF vectors, your dataset might now have 500+ columns! 
Having too many features causes the "Curse of Dimensionality" (slower training, overfitting). We need to filter them down to only the best ones.

| Next Topic | Why |
|------------|-----|
| [Feature Selection](./13-Feature-Selection.md) | Learn how to scientifically drop useless features and keep only the ones that matter, improving model speed and accuracy. |

---

[← Previous: Data Preprocessing](./11-Data-Preprocessing.md) | [Back to Main Index](../README.md) | [Next: Feature Selection →](./13-Feature-Selection.md)
