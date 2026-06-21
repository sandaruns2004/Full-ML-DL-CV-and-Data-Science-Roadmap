# ⚖️ Handling Imbalanced Data

> **Prerequisites**: [Feature Selection](./13-Feature-Selection.md) | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents

1. [The Accuracy Paradox](#1-the-accuracy-paradox)
2. [Better Evaluation Metrics](#2-better-evaluation-metrics)
3. [Resampling Data (Undersampling & Oversampling)](#3-resampling-data-undersampling--oversampling)
4. [SMOTE (Synthetic Minority Over-sampling)](#4-smote-synthetic-minority-over-sampling)
5. [Algorithmic Approaches (Class Weights)](#5-algorithmic-approaches-class-weights)
6. [What's Next](#6-whats-next)

---

## 1. The Accuracy Paradox

### 🟢 Beginner

Imagine you are building a Machine Learning model to detect Credit Card Fraud. 
In your dataset of 100,000 transactions:
- **99,000** are Legitimate (99%)
- **1,000** are Fraudulent (1%)

This is highly **Imbalanced Data**. 

You write a terrible ML model that consists of exactly one line of code:
`return "Legitimate"` (It just guesses Legitimate for everything).

What is the Accuracy of this model? **99%!!**
Your model looks like a massive success, but it failed to catch a single case of fraud. This is the **Accuracy Paradox**. When data is imbalanced, accuracy is a useless metric.

---

## 2. Better Evaluation Metrics

### 🟡 Intermediate

When dealing with imbalanced data, you must completely ignore "Accuracy" and use the Confusion Matrix metrics: Precision, Recall, and F1-Score.

*   **Precision**: When the model yells "FRAUD!", how often is it actually fraud? (Prevents false alarms/annoying the customer).
*   **Recall (Sensitivity)**: Out of all the ACTUAL frauds, how many did the model catch? (Prevents financial loss).
*   **F1-Score**: The harmonic mean of Precision and Recall. Use this as your primary metric.

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# After model.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---

## 3. Resampling Data (Undersampling & Oversampling)

### 🟡 Intermediate

The easiest way to fix imbalanced data is to change the data itself before you train the model.

**1. Random Undersampling (Shrink the majority)**
Delete random rows from the Legitimate class until it matches the Fraud class.
- *Pros*: Super fast, reduces training time.
- *Cons*: You are deleting 98,000 rows of perfectly good data!

**2. Random Oversampling (Duplicate the minority)**
Duplicate the Fraud rows until they match the Legitimate class.
- *Pros*: You don't lose any data.
- *Cons*: The model sees the exact same fraudulent rows over and over, leading to severe overfitting.

```python
from sklearn.utils import resample
import pandas as pd

# Separate majority and minority classes
df_majority = df[df['Class'] == 0] # Legitimate
df_minority = df[df['Class'] == 1] # Fraud

# Undersample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=1000,   # match minority class size
                                 random_state=42)

# Combine minority class with downsampled majority class
df_balanced = pd.concat([df_majority_downsampled, df_minority])
```

---

## 4. SMOTE (Synthetic Minority Over-sampling Technique)

### 🔴 Advanced

Since duplicating the minority class causes overfitting, what if we generated *brand new, synthetic* examples of the minority class?

**SMOTE** works by picking a fraud case, finding its nearest fraud neighbors, and drawing a line between them. It then randomly places a new, synthetic data point somewhere on that line.

```python
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

print("Before SMOTE:")
print(y_train.value_counts())
# 0: 99000
# 1: 1000

smote = SMOTE(random_state=42)

# Fit and apply ONLY ON THE TRAINING DATA
# NEVER run SMOTE on your test data! Your test data must represent reality.
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(y_train_smote.value_counts())
# 0: 99000
# 1: 99000
```

> ⚠️ **CRITICAL PIPELINE RULE**: You must split your data into Train/Test *BEFORE* you apply SMOTE. If you apply SMOTE to the whole dataset and then split, synthetic data will bleed into your test set, and your evaluation metrics will be fake.

---

## 5. Algorithmic Approaches (Class Weights)

### 🔴 Advanced

Instead of changing the data, you can change the algorithm to pay more attention to the minority class. 

You can pass a `class_weight` parameter to almost every Scikit-Learn model. This tells the math equation: "If you get a Legitimate transaction wrong, penalty is 1 point. If you get a Fraud transaction wrong, penalty is 100 points."

```python
from sklearn.ensemble import RandomForestClassifier

# 'balanced' automatically calculates weights inversely proportional to class frequencies
# Weight for class = Total_Samples / (Number_of_Classes * Samples_in_Class)
model = RandomForestClassifier(class_weight='balanced')

# OR define them manually
weights = {0: 1.0, 1: 99.0} # Legitimate has weight 1, Fraud has weight 99
model = RandomForestClassifier(class_weight=weights)

model.fit(X_train, y_train)
```

**Ensemble Methods for Imbalance:**
Certain algorithms like **Isolation Forests** (Anomaly Detection) or **Balanced Random Forests** are designed from the ground up to handle 99/1 splits without requiring SMOTE.

---

## 6. What's Next

Congratulations! You have completed the core data manipulation, statistics, visualization, and preprocessing syllabus. 

To round out your foundational knowledge, we must cover two essential topics that are heavily tested in interviews: SQL and Data Ethics.

| Next Topic | Why |
|------------|-----|
| [SQL for Data Science](./15-SQL-for-Data-Science.md) | Data lives in databases. If you can't write SQL, you can't be a Data Scientist. Learn how to write analytic queries. |

---

[← Previous: Feature Selection](./13-Feature-Selection.md) | [Back to Main Index](../README.md) | [Next: SQL for Data Science →](./15-SQL-for-Data-Science.md)
