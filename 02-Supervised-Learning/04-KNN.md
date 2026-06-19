# 🎯 K-Nearest Neighbors (KNN)

> **Prerequisites**: Data Preprocessing, Feature Scaling | **Difficulty**: ⭐⭐☆☆☆ Elementary

---

## 📋 Table of Contents
1. [Intuition](#1-intuition)
2. [The Mathematics](#2-the-mathematics)
3. [Distance Metrics](#3-distance-metrics)
4. [Choosing K](#4-choosing-k)
5. [Curse of Dimensionality](#5-curse-of-dimensionality)
6. [Weighted KNN](#6-weighted-knn)
7. [Implementation from Scratch](#7-implementation-from-scratch)
8. [scikit-learn Implementation](#8-scikit-learn-implementation)
9. [KNN for Regression](#9-knn-for-regression)
10. [Project Ideas & What's Next](#10-project-ideas--whats-next)

---

## 1. Intuition

KNN is the simplest ML algorithm: **"You are the average of your neighbors."**

Given a new data point:
1. Find the $K$ closest data points in the training set
2. For **classification**: Take a majority vote
3. For **regression**: Take the average

**Key property**: KNN is a **lazy learner** — it doesn't build a model. It memorizes ALL the data and makes decisions at prediction time.

---

## 2. The Mathematics

**Classification**: Given query point $\mathbf{x}_q$, find the $K$ nearest neighbors $\mathcal{N}_K(\mathbf{x}_q)$:

$$\hat{y} = \arg\max_c \sum_{i \in \mathcal{N}_K(\mathbf{x}_q)} \mathbb{1}(y_i = c)$$

**Regression**: Take the average of neighbors' target values:

$$\hat{y} = \frac{1}{K}\sum_{i \in \mathcal{N}_K(\mathbf{x}_q)} y_i$$

---

## 3. Distance Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Euclidean (L2)** | $d = \sqrt{\sum(x_i - y_i)^2}$ | Default, continuous data |
| **Manhattan (L1)** | $d = \sum\|x_i - y_i\|$ | High dimensions, sparse data |
| **Minkowski** | $d = (\sum\|x_i - y_i\|^p)^{1/p}$ | General (p=1→L1, p=2→L2) |
| **Cosine** | $d = 1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$ | Text data, high dimensions |
| **Hamming** | Count of differing positions | Categorical/binary data |

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize different distance metrics in 2D
point = np.array([0, 0])
target = np.array([3, 4])

# L1 path (Manhattan)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Euclidean
axes[0].plot([0, 3], [0, 4], 'b-', linewidth=3)
axes[0].plot(0, 0, 'go', markersize=10)
axes[0].plot(3, 4, 'ro', markersize=10)
d = np.sqrt(3**2 + 4**2)
axes[0].set_title(f'Euclidean Distance = {d:.1f}', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# Manhattan
axes[1].plot([0, 3, 3], [0, 0, 4], 'r-', linewidth=3)
axes[1].plot(0, 0, 'go', markersize=10)
axes[1].plot(3, 4, 'ro', markersize=10)
d = abs(3) + abs(4)
axes[1].set_title(f'Manhattan Distance = {d:.1f}', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

# Equidistant circles vs diamonds
theta = np.linspace(0, 2*np.pi, 100)
axes[2].plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2, label='L2 unit circle')
diamond = np.array([[1,0],[0,1],[-1,0],[0,-1],[1,0]])
axes[2].plot(diamond[:,0], diamond[:,1], 'r-', linewidth=2, label='L1 unit diamond')
axes[2].set_title('Unit "Circles" for L1 vs L2', fontsize=13, fontweight='bold')
axes[2].set_aspect('equal')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distance_metrics.png', dpi=150)
plt.show()
```

---

## 4. Choosing K

- **K too small** (K=1): Overfitting, sensitive to noise
- **K too large** (K=n): Underfitting, always predicts majority class
- **Best practice**: Use odd K for binary classification (avoid ties)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=300, n_features=10, random_state=42)

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_values, cv_scores, 'bo-', linewidth=2, markersize=4)
ax.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
ax.set_xlabel('K (Number of Neighbors)', fontsize=12)
ax.set_ylabel('Cross-Validation Accuracy', fontsize=12)
ax.set_title('Choosing Optimal K', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('choosing_k.png', dpi=150)
plt.show()
```

---

## 5. Curse of Dimensionality

As dimensions increase, distances become **meaningless** — all points appear equidistant!

$$\text{As } d \to \infty: \quad \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate curse of dimensionality
dims = [2, 5, 10, 50, 100, 500, 1000]
n_points = 1000
ratios = []

for d in dims:
    points = np.random.uniform(0, 1, (n_points, d))
    query = np.random.uniform(0, 1, d)
    distances = np.linalg.norm(points - query, axis=1)
    ratio = (distances.max() - distances.min()) / distances.min()
    ratios.append(ratio)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dims, ratios, 'ro-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Dimensions', fontsize=12)
ax.set_ylabel('(max_dist - min_dist) / min_dist', fontsize=12)
ax.set_title('Curse of Dimensionality: Distances Become Meaningless', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curse_of_dimensionality.png', dpi=150)
plt.show()
```

**Solutions**: PCA, feature selection, use Manhattan distance in high dimensions.

---

## 6. Weighted KNN

Give closer neighbors **more weight**: $w_i = \frac{1}{d(\mathbf{x}_q, \mathbf{x}_i)^2}$

```python
from sklearn.neighbors import KNeighborsClassifier

# Uniform weights (default)
knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# Distance weights (closer neighbors have more influence)
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

---

## 7. Implementation from Scratch

```python
import numpy as np
from collections import Counter

class KNNFromScratch:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            # Compute distances to all training points
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            # Get K nearest neighbor indices
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            # Majority vote
            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# Test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

knn = KNNFromScratch(k=5)
knn.fit(X_train_s, y_train)
print(f"From-scratch accuracy: {knn.score(X_test_s, y_test):.2%}")
```

---

## 8. scikit-learn Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
knn.fit(X_train_s, y_train)

print(f"Accuracy: {knn.score(X_test_s, y_test):.4f}")
print(f"\n{classification_report(y_test, knn.predict(X_test_s), target_names=data.target_names)}")
```

---

## 9. KNN for Regression

```python
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.sort(np.random.uniform(0, 5, 100)).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(100) * 0.2

X_plot = np.linspace(0, 5, 200).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, k in zip(axes, [1, 5, 20]):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, y)
    y_pred = model.predict(X_plot)
    
    ax.scatter(X, y, alpha=0.4, s=20, color='#36A2EB')
    ax.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'KNN (K={k})')
    ax.plot(X_plot, np.sin(X_plot.ravel()), 'g--', alpha=0.5, label='True')
    ax.set_title(f'K = {k}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('KNN Regression with Different K Values', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('knn_regression.png', dpi=150)
plt.show()
```

---

## 10. Project Ideas & What's Next

### Project Ideas
- 🟢 **Handwritten Digit Recognition** with KNN on MNIST
- 🟡 **Movie Recommendation System** using KNN on user ratings
- 🔴 **Image Classification** with optimized KNN and dimensionality reduction

### What's Next
| Next | Why |
|------|-----|
| [Decision Trees](./05-Decision-Trees.md) | A different paradigm — rule-based classification |
| [Dimensionality Reduction](../04-Unsupervised-Learning/02-Dimensionality-Reduction.md) | Fight the curse of dimensionality |

---

[← Logistic Regression](./03-Logistic-Regression.md) | [Back to Index](../README.md) | [Next: Decision Trees →](./05-Decision-Trees.md)
