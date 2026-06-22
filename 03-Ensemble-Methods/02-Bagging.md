# 🎒 Bagging (Bootstrap Aggregating)

> **Prerequisites**: Introduction to Ensembles | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 📋 Table of Contents
1. [Bootstrap Sampling & Parallel Learning](#1-bootstrap-sampling--parallel-learning)
2. [Why Bootstrap Works](#2-why-bootstrap-works)
3. [Out-of-Bag (OOB) Error](#3-out-of-bag-oob-error)
4. [Python Demonstration of Bootstrap](#4-python-demonstration-of-bootstrap)

---

## 1. Bootstrap Sampling & Parallel Learning

### 🟢 Beginner
**Simple Explanation**: "Bagging" stands for Bootstrap Aggregating. Imagine you have 100 students taking an exam. Instead of giving them all the exact same practice test, you give each student a slightly different, randomly selected mix of questions from the test bank (sampling with replacement). Then, they all take the final exam and you average their scores.

**Visual Intuition**:
```
Original Dataset
  ├── Sample 1 (with replacement) ──> Base Model 1 ──┐
  ├── Sample 2 (with replacement) ──> Base Model 2 ──┼──> Average/Majority Vote ──> Final Prediction
  └── Sample B (with replacement) ──> Base Model B ──┘
```

### 🟡 Intermediate
**Working Mechanism**: 
1. **Bootstrap**: Draw $N$ samples from the original dataset of size $N$ with replacement.
2. **Train**: Train a base model (usually an unpruned decision tree) independently and in parallel on each sample.
3. **Aggregate**: Use Majority Voting (Classification) or Averaging (Regression) to combine the predictions.

$$\hat{f}_{bag}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} \hat{f}_b(\mathbf{x})$$

---

## 2. Why Bootstrap Works

### 🔴 Advanced
**Variance Reduction**: If we have $B$ independent trees each with variance $\sigma^2$, the variance of the average is:

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B} f_b\right) = \frac{\sigma^2}{B}$$

In reality, the bootstrap samples are drawn from the same training set, so the trees are not perfectly independent. If they have correlation $\rho$, the variance of the bagged ensemble is:

$$\text{Var}_{bagging} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B \to \infty$, the second term goes to 0, and the variance approaches $\rho\sigma^2$. To reduce variance further, we must reduce the correlation $\rho$ between the trees (which is what Random Forest does!).

---

## 3. Out-of-Bag (OOB) Error

Since bootstrap sampling is done with replacement, some data points are selected multiple times in a sample, and some are not selected at all. 

For a dataset of size $N$, the probability of a specific point *not* being chosen in a single draw is $1 - \frac{1}{N}$.
For $N$ draws, the probability that it is never chosen is:

$$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.368$$

This means ~36.8% of instances are never seen by a given base model! These are called **Out-of-Bag (OOB)** samples.
We can use these OOB samples as a built-in validation set! We evaluate each model on the samples it didn't see, providing an unbiased validation score without needing a separate train-test split or cross-validation.

---

## 4. Python Demonstration of Bootstrap

Here is how we can visualize the bootstrap sampling process in Python:

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate bootstrap sampling
np.random.seed(42)
original_data = np.arange(10)
print(f"Original data: {original_data}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    bootstrap = np.random.choice(original_data, size=len(original_data), replace=True)
    oob = np.setdiff1d(original_data, bootstrap)
    
    # Visualize
    ax.bar(range(10), np.bincount(bootstrap, minlength=10), color='#36A2EB', edgecolor='white')
    ax.set_title(f'Bootstrap {i+1}\nOOB: {oob}', fontsize=10)
    ax.set_xticks(range(10))
    ax.set_xlabel('Data Index')
    ax.set_ylabel('Count')

plt.suptitle('Bootstrap Samples (with replacement)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Show that ~63.2% of data appears in each bootstrap
fractions = []
for _ in range(10000):
    sample = np.random.choice(100, size=100, replace=True)
    fractions.append(len(np.unique(sample)) / 100)
print(f"Average fraction of unique samples: {np.mean(fractions):.4f} (expected: 0.632)")
```

---

[← Introduction To Ensemble Learning](01-Introduction-To-Ensemble-Learning.md) | [Back to Index](../README.md) | [Next: Random Forest →](03-Random-Forest.md)
