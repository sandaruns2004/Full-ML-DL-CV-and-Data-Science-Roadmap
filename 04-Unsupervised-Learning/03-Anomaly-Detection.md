# 🚨 Anomaly Detection

> **Prerequisites**: Clustering, Linear Algebra, Probability | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Anomaly Detection Landscape](#1-the-anomaly-detection-landscape)
2. [Statistical Methods: Z-Score & Mahalanobis Distance](#2-statistical-methods-z-score--mahalanobis-distance)
3. [Isolation Forest (Tree-Based)](#3-isolation-forest-tree-based)
4. [Local Outlier Factor (Density-Based)](#4-local-outlier-factor-density-based)
5. [One-Class SVM (Boundary-Based)](#5-one-class-svm-boundary-based)
6. [PCA Reconstruction Error](#6-pca-reconstruction-error)
7. [Comparing Methods & Practical Guide](#7-comparing-methods--practical-guide)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. The Anomaly Detection Landscape

Anomaly detection (Outlier detection) is the identification of rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.

**Key Challenges:**
- Highly imbalanced data (anomalies might be 0.01% of data).
- The boundary between "normal" and "anomalous" is often blurry.
- Anomalies might mimic normal behavior (malicious attacks).

**Types of Anomalies:**
1. **Point Anomalies**: A single data point is far off (e.g., Credit card transaction of $10,000).
2. **Contextual Anomalies**: Anomaly depends on context (e.g., $100 grocery bill is normal, $100 coffee bill is anomalous).
3. **Collective Anomalies**: A sequence of points is anomalous, even if individual points aren't (e.g., DoS network attack).

---

## 2. Statistical Methods: Z-Score & Mahalanobis Distance

### Z-Score (Univariate)
Assumes data follows a Gaussian (Normal) distribution. 
$$Z = \frac{x - \mu}{\sigma}$$
We typically flag an anomaly if $|Z| > 3$ (outside 99.7% of the distribution).

### Mahalanobis Distance (Multivariate)
Z-Score fails in multi-dimensions because it ignores correlation between features. If Height and Weight are highly positively correlated, someone who is very tall and very light is an anomaly, even if their height and weight are individually within normal ranges.

Mahalanobis distance accounts for the covariance $\mathbf{\Sigma}$ between features:
$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

If $\mathbf{\Sigma}$ is the identity matrix, this reduces to standard Euclidean distance!

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

np.random.seed(42)
# Generate correlated data
mean = [5, 5]
cov = [[3, 2.5], [2.5, 3]]
X = np.random.multivariate_normal(mean, cov, 200)
X_outliers = np.array([[2, 8], [8, 2]]) # Off-diagonal outliers
X_all = np.vstack([X, X_outliers])

# Calculate empirical mean and covariance
emp_mean = np.mean(X_all, axis=0)
emp_cov = np.cov(X_all, rowvar=False)
inv_cov = np.linalg.inv(emp_cov)

# Calculate Mahalanobis distance for all points
distances = [mahalanobis(x, emp_mean, inv_cov) for x in X_all]

# Threshold using Chi-Square distribution (p=0.01)
threshold = np.sqrt(chi2.ppf((1-0.01), df=2))
anomalies = np.array([x for x, d in zip(X_all, distances) if d > threshold])
normals = np.array([x for x, d in zip(X_all, distances) if d <= threshold])

plt.figure(figsize=(8, 6))
plt.scatter(normals[:, 0], normals[:, 1], c='blue', alpha=0.6, label='Normal')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', marker='x', s=100, linewidths=3, label='Anomaly')

# Draw confidence ellipse
theta = np.linspace(0, 2*np.pi, 100)
circle = np.vstack((np.sin(theta), np.cos(theta)))
ellipse = np.dot(np.linalg.cholesky(emp_cov), circle) * threshold + emp_mean[:, None]
plt.plot(ellipse[0,:], ellipse[1,:], color='red', linestyle='--', label=f'Threshold (d={threshold:.2f})')

plt.title('Multivariate Anomaly Detection (Mahalanobis Distance)', fontsize=14, fontweight='bold')
plt.legend()
plt.axis('equal')
plt.savefig('mahalanobis.png', dpi=150)
plt.show()
```

---

## 3. Isolation Forest (Tree-Based)

Isolation Forest doesn't profile normal data; it explicitly **isolates** anomalies. 

**The Algorithm:**
1. Select a random feature.
2. Select a random split value between the min and max of that feature.
3. Repeat recursively until the data point is isolated into a leaf node.

**The Math (Path Length):**
Since anomalies are "few and different", they require **fewer splits** to be isolated. The anomaly score is inversely proportional to the path length $h(x)$ in the isolation trees.

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
- $E(h(x))$: Average path length over all trees in the forest.
- $c(n)$: Average path length of unsuccessful search in a Binary Search Tree (used for normalization).
- If $s \approx 1$, the instance is highly anomalous.
- If $s < 0.5$, it is quite safe to be regarded as normal.

```python
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X_all = np.vstack([X, outliers])

# Fit Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
preds = iso.fit_predict(X_all) # Returns 1 for normal, -1 for anomaly

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
Z = iso.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap='Blues', alpha=0.3)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red') # Decision Boundary
plt.scatter(X_all[preds==1, 0], X_all[preds==1, 1], c='blue', s=20, label='Normal')
plt.scatter(X_all[preds==-1, 0], X_all[preds==-1, 1], c='red', s=50, marker='x', label='Anomaly')

plt.title('Isolation Forest Decision Boundary', fontsize=14, fontweight='bold')
plt.legend()
plt.savefig('isolation_forest.png', dpi=150)
plt.show()
```

---

## 4. Local Outlier Factor (Density-Based)

LOF computes the local density deviation of a given data point with respect to its neighbors. It considers the **relative density**, not absolute density.

This is crucial for datasets with clusters of varying densities. A point slightly away from a dense cluster might be an anomaly, while a point far away from a sparse cluster might just be part of the sparse cluster!

**The Math Formulation:**
1. **Reachability Distance (RD)**: The distance between point $a$ and $b$, bounded below by the $k$-distance of $b$.
   $$RD_k(a, b) = \max\{k\text{-distance}(b), d(a, b)\}$$
2. **Local Reachability Density (LRD)**: Inverse of the average reachability distance of point $a$ from its $k$-nearest neighbors.
3. **LOF Score**: The ratio of the average LRD of $a$'s neighbors to the LRD of $a$.
   $$\text{LOF}_k(a) = \frac{\sum_{b \in N_k(a)} \frac{LRD_k(b)}{LRD_k(a)}}{|N_k(a)|}$$
   - LOF $\approx 1$: Similar density to neighbors (Normal).
   - LOF $\gg 1$: Lower density than neighbors (Anomaly).

```python
from sklearn.neighbors import LocalOutlierFactor

# Generate varying density clusters
X_dense, _ = make_blobs(n_samples=100, centers=[[0, 0]], cluster_std=0.5, random_state=42)
X_sparse, _ = make_blobs(n_samples=100, centers=[[4, 4]], cluster_std=2.0, random_state=42)
X_outliers = np.array([[2, 2], [-1, -1], [8, 8]])
X_all = np.vstack([X_dense, X_sparse, X_outliers])

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
preds = lof.fit_predict(X_all)
scores = -lof.negative_outlier_factor_ # Convert to positive LOF score

plt.figure(figsize=(8, 6))
plt.scatter(X_all[:, 0], X_all[:, 1], color='k', s=3., label='Data points')
radius = (scores - scores.min()) / (scores.max() - scores.min())
plt.scatter(X_all[:, 0], X_all[:, 1], s=1000 * radius, edgecolors='r', facecolors='none', label='LOF Score Radius')

plt.title('Local Outlier Factor (Varying Density Clusters)', fontsize=14, fontweight='bold')
plt.legend()
plt.savefig('lof_anomalies.png', dpi=150)
plt.show()
```

---

## 5. One-Class SVM (Boundary-Based)

SVM aims to find a hyperplane separating classes. But what if we only have one class (normal data)? 
**One-Class SVM** separates the data from the origin in the high-dimensional kernel space, pushing a hyperplane as close to the data as possible.

**Mathematical Formulation (Dual Form):**
$$\min_{\boldsymbol{\alpha}} \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j K(\mathbf{x}_i, \mathbf{x}_j)$$
Subject to: $0 \leq \alpha_i \leq \frac{1}{\nu N}$ and $\sum_i \alpha_i = 1$

Here, $\nu$ (nu) is an upper bound on the fraction of outliers and a lower bound on the fraction of support vectors. 

*Best used when the training data is mostly clean (Novelty Detection).*

```python
from sklearn.svm import OneClassSVM

# Fit One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
ocsvm.fit(X) # Note: Only fitting on the clean data (Novelty Detection)
preds = ocsvm.predict(X_all)
```

---

## 6. PCA Reconstruction Error

PCA acts as an autoencoder. It projects high-dimensional data down to a low-dimensional space, capturing the main variance, and then reconstructs it back.
Anomalies, by definition, have variance in unexpected directions. Thus, PCA will drop that information during projection, resulting in a **massive reconstruction error** when mapping back.

**Error Formulation**:
$$E(\mathbf{x}) = ||\mathbf{x} - \mathbf{x}_{reconstructed}||^2$$

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assume X_all contains 50 features. We scale it.
X_high_dim = np.random.randn(300, 50)
outliers_high_dim = np.random.uniform(-5, 5, (10, 50))
X_all_hd = np.vstack([X_high_dim, outliers_high_dim])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all_hd)

# Project to 5D and reconstruct back to 50D
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_pca)

# Calculate MSE for each point
reconstruction_errors = np.mean((X_scaled - X_reconstructed)**2, axis=1)

plt.figure(figsize=(10, 4))
plt.hist(reconstruction_errors, bins=50, color='blue', alpha=0.7)
plt.axvline(np.percentile(reconstruction_errors, 95), color='red', linestyle='--', label='95th Percentile Threshold')
plt.title('PCA Reconstruction Error Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Mean Squared Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

---

## 7. Comparing Methods & Practical Guide

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Mahalanobis** | Normally distributed, correlated data | Fast, rigorous stats | Fails on non-linear/multi-modal data |
| **Isolation Forest** | General purpose, tabular data | Scales well to high dims | Boundary can be too rigid |
| **LOF** | Clusters of varying densities | Handles complex local structures | Computationally heavy $O(N^2)$ |
| **One-Class SVM** | Novelty detection (clean training set) | Powerful non-linear boundaries via kernels | Very sensitive to hyperparameter tuning |
| **PCA Reconstruction** | High-dimensional correlated tabular data | Interpretable, fast | Only captures linear relationships |

**General Pipeline:**
1. Try **Isolation Forest** first as a robust baseline.
2. If data has complex non-linear manifolds, try **LOF**.
3. For deep learning, move towards Autoencoders (non-linear PCA equivalents).

---

## 8. Project Ideas & What's Next

### Project Ideas
- 🟢 **Credit Card Fraud Pipeline**: Use Kaggle's Credit Card Fraud dataset. Compare Isolation Forest, LOF, and PCA Reconstruction. Plot Precision-Recall curves.
- 🔴 **Custom Distance LOF**: Implement LOF using a custom distance metric (like Cosine Similarity) instead of Euclidean, useful for NLP document anomaly detection.

### What's Next
| Next | Why |
|------|-----|
| [Metrics & Evaluation](../05-Model-Evaluation/01-Metrics-And-Evaluation.md) | We just evaluated anomalies. How do we properly measure classification success in heavily imbalanced data? |

---

[← Dimensionality Reduction](./02-Dimensionality-Reduction.md) | [Back to Index](../README.md) | [Next: Gaussian Mixture Models →](./04-Gaussian-Mixture-Models.md)
