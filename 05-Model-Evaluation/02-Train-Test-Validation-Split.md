# ✂️ Train, Test, and Validation Split

> **Prerequisites**: Intro to Model Evaluation | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Three Datasets](#1-the-three-datasets)
2. [Data Leakage](#2-data-leakage)
3. [Implementation in Scikit-Learn](#3-implementation)

---

## 1. The Three Datasets

### 🟢 Beginner
**Simple Explanation**: 
To properly train a model, we divide our data into three distinct buckets:
1. **Train Set (70%)**: The textbook. The model uses this to learn.
2. **Validation Set (15%)**: The practice quiz. We use this to tweak the model's settings (hyperparameters) and see how it's doing.
3. **Test Set (15%)**: The final exam. Used strictly *once* at the very end to report the final grade.

**Visual Intuition**:
```text
[================= Full Dataset =================]
[==== Train (70%) ====] [Val (15%)] [Test (15%)]
```

### 🟡 Intermediate
**Workflow and Practical Implementation**:
If you tune your hyperparameters based on the Test Set, your model is inadvertently "learning" the Test Set. This invalidates the evaluation. Therefore, a Validation set is strictly required when performing hyperparameter tuning (like adjusting the depth of a tree or the learning rate of a neural network).

**Python Implementation**:
```python
from sklearn.model_selection import train_test_split

# First split: Separate out the Test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Second split: Separate the remaining data into Train and Validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42) 
# 0.176 of 0.85 is ~0.15 of the total
```

### 🔴 Advanced
**Data Leakage & Statistical Independence**:
Data leakage occurs when information from outside the training dataset is used to create the model.
- **Temporal Leakage**: In time-series data, random splitting is disastrous. You must use chronological splitting (e.g., train on Jan-Oct, test on Nov-Dec).
- **Feature Leakage**: Applying `StandardScaler().fit_transform()` on the *entire* dataset before splitting leaks the global mean and variance into the training set. 
**Mathematical Rule**: Any statistic (mean, variance, PCA components, Imputation values) used for transformation must be calculated *strictly* on $\mathcal{D}_{train}$ and then applied to $\mathcal{D}_{val}$ and $\mathcal{D}_{test}$.
