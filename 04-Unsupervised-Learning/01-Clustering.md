# 🔍 Clustering — K-Means, DBSCAN, Hierarchical, GMM

> **Prerequisites**: Linear Algebra, Feature Scaling | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [What Is Clustering?](#1-what-is-clustering)
2. [K-Means Clustering](#2-k-means-clustering)
3. [DBSCAN](#3-dbscan)
4. [Hierarchical Clustering](#4-hierarchical-clustering)
5. [Gaussian Mixture Models (GMM)](#5-gaussian-mixture-models-gmm)
6. [Choosing the Right Algorithm](#6-choosing-the-right-algorithm)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Implementation from Scratch](#8-implementation-from-scratch)
9. [Project Ideas & What's Next](#9-project-ideas--whats-next)

---

## 1. What Is Clustering?

Clustering is **unsupervised learning** — grouping similar data points without labels.

**Applications**: Customer segmentation, anomaly detection, image compression, document grouping, gene expression analysis.

---

## 2. K-Means Clustering

### Algorithm

1. Choose $K$ (number of clusters)
2. Initialize $K$ centroids randomly
3. **Assign**: Each point to the nearest centroid
4. **Update**: Move centroids to the mean of assigned points
5. Repeat steps 3-4 until convergence

### Mathematics

**Objective** (minimize within-cluster sum of squares):

$$J = \sum_{k=1}^{K}\sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

**Assignment step**: $c_i = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$

**Update step**: $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i \in C_k} \mathbf{x}_i$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# K-Means from scratch
class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iter=100):
        self.k = n_clusters
        self.max_iter = max_iter
    
    def fit(self, X):
        n = len(X)
        # Random initialization
        idx = np.random.choice(n, self.k, replace=False)
        self.centroids = X[idx].copy()
        
        for _ in range(self.max_iter):
            # Assign
            distances = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids])
            self.labels = distances.argmin(axis=0)
            
            # Update
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.k)])
            
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        distances = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids])
        return distances.argmin(axis=0)

# Run K-Means
km = KMeansFromScratch(n_clusters=4)
km.fit(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, s=20)
axes[0].set_title('Before Clustering', fontsize=13, fontweight='bold')

axes[1].scatter(X[:, 0], X[:, 1], c=km.labels, cmap='viridis', alpha=0.5, s=20)
axes[1].scatter(km.centroids[:, 0], km.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
axes[1].set_title('After K-Means (from scratch)', fontsize=13, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('kmeans.png', dpi=150)
plt.show()
```

### Choosing K — The Elbow Method

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
    
    from sklearn.metrics import silhouette_score
    silhouettes.append(silhouette_score(X, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2)
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_title('Elbow Method', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouettes, 'ro-', linewidth=2)
axes[1].set_xlabel('K')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Method', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
plt.show()
```

---

## 3. DBSCAN

**Density-Based Spatial Clustering of Applications with Noise**

Unlike K-Means, DBSCAN:
- Doesn't require specifying $K$
- Can find **arbitrarily shaped** clusters
- Identifies **outliers** (noise points)

### Parameters:
- $\epsilon$ (eps): Maximum distance between two points to be neighbors
- MinPts: Minimum points to form a dense region

### Point Types:
- **Core point**: Has ≥ MinPts within $\epsilon$ radius
- **Border point**: Within $\epsilon$ of a core point but not a core point itself
- **Noise point**: Neither core nor border

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

# Generate non-convex data (K-Means fails here!)
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# K-Means (fails on non-convex shapes)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, random_state=42, n_init=10)
axes[0].scatter(X[:, 0], X[:, 1], c=km.fit_predict(X), cmap='viridis', s=20)
axes[0].set_title('K-Means (fails!)', fontsize=13, fontweight='bold')

# DBSCAN (handles it!)
db = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X)
axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20)
noise = labels == -1
axes[1].scatter(X[noise, 0], X[noise, 1], c='red', marker='x', s=50, label='Noise')
axes[1].set_title(f'DBSCAN ({len(set(labels))-1} clusters)', fontsize=13, fontweight='bold')
axes[1].legend()

# Different eps values
for eps_val in [0.1, 0.3, 0.5]:
    db = DBSCAN(eps=eps_val, min_samples=5)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

axes[2].scatter(X[:, 0], X[:, 1], c=DBSCAN(eps=0.2, min_samples=5).fit_predict(X), cmap='viridis', s=20)
axes[2].set_title('DBSCAN with optimal eps', fontsize=13, fontweight='bold')

plt.suptitle('K-Means vs DBSCAN on Non-Convex Data', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('dbscan.png', dpi=150)
plt.show()
```

---

## 4. Hierarchical Clustering

Builds a tree (**dendrogram**) of clusters:

### Agglomerative (Bottom-Up):
1. Start with each point as its own cluster
2. Merge the two closest clusters
3. Repeat until one cluster remains

### Linkage Methods:
| Method | Distance Between Clusters |
|--------|--------------------------|
| **Single** | Min distance between any two points |
| **Complete** | Max distance between any two points |
| **Average** | Average distance between all pairs |
| **Ward** | Minimizes within-cluster variance |

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=50, centers=3, cluster_std=1.0, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Dendrogram
Z = linkage(X, method='ward')
dendrogram(Z, ax=axes[0], truncate_mode='lastp', p=20)
axes[0].set_title('Dendrogram (Ward Linkage)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Distance')

# Clustering result
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)
axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, edgecolors='black')
axes[1].set_title('Hierarchical Clustering (3 clusters)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('hierarchical.png', dpi=150)
plt.show()
```

---

## 5. Gaussian Mixture Models (GMM)

GMM assumes data is generated from a **mixture of Gaussian distributions**:

$$P(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

- $\pi_k$ = mixing coefficient (weight of cluster $k$)
- $\boldsymbol{\mu}_k$ = mean of cluster $k$
- $\boldsymbol{\Sigma}_k$ = covariance matrix of cluster $k$

**Trained with EM (Expectation-Maximization)** algorithm.

**Advantage over K-Means**: GMM gives **soft assignments** (probability of belonging to each cluster) and can model **elliptical** clusters (not just spherical).

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
# Generate elongated clusters (K-Means struggles)
X1 = np.random.multivariate_normal([0, 0], [[3, 1], [1, 1]], 150)
X2 = np.random.multivariate_normal([5, 5], [[1, -0.5], [-0.5, 2]], 150)
X = np.vstack([X1, X2])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, random_state=42, n_init=10)
axes[0].scatter(X[:, 0], X[:, 1], c=km.fit_predict(X), cmap='viridis', s=20)
axes[0].set_title('K-Means (spherical assumption)', fontsize=13, fontweight='bold')

# GMM
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
labels = gmm.fit_predict(X)
probs = gmm.predict_proba(X)

axes[1].scatter(X[:, 0], X[:, 1], c=probs[:, 0], cmap='viridis', s=20)
axes[1].set_title('GMM (soft assignments, elliptical)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('gmm.png', dpi=150)
plt.show()
```

---

## 6. Choosing the Right Algorithm

| Algorithm | Shape | K required? | Outliers | Scalability |
|-----------|-------|:-----------:|:--------:|:-----------:|
| **K-Means** | Spherical | Yes | Sensitive | ✅ Large |
| **DBSCAN** | Arbitrary | No | Handles | ⚠️ Medium |
| **Hierarchical** | Arbitrary | Cut level | No | ❌ Small |
| **GMM** | Elliptical | Yes | Moderate | ⚠️ Medium |

---

## 7. Evaluation Metrics

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import numpy as np

# Evaluate clustering quality (no ground truth needed)
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X)

print(f"Silhouette Score:       {silhouette_score(X, labels):.4f}")      # Higher = better
print(f"Calinski-Harabasz:      {calinski_harabasz_score(X, labels):.4f}") # Higher = better
print(f"Davies-Bouldin:         {davies_bouldin_score(X, labels):.4f}")   # Lower = better
```

---

## 8. Implementation from Scratch

See K-Means implementation in Section 2 above. Additional implementations:

```python
# Mini-Batch K-Means for large datasets
from sklearn.cluster import MiniBatchKMeans

mbkm = MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=42)
labels = mbkm.fit_predict(X)
print(f"Mini-Batch K-Means completed with {len(set(labels))} clusters")
```

---

## 9. Project Ideas & What's Next

### Project Ideas
- 🟢 **Customer Segmentation** — Cluster e-commerce customers by behavior
- 🟡 **Image Color Quantization** — Reduce image colors using K-Means
- 🔴 **Anomaly Detection System** — Use DBSCAN/GMM to detect outliers in network traffic

### What's Next
| Next | Why |
|------|-----|
| [Dimensionality Reduction](./02-Dimensionality-Reduction.md) | PCA, t-SNE for visualization |
| [Anomaly Detection](./03-Anomaly-Detection.md) | Detect unusual patterns |

---

[← Stacking And Voting](../03-Ensemble-Methods/03-Stacking-And-Voting.md) | [Back to Index](../README.md) | [Next: Dimensionality Reduction →](./02-Dimensionality-Reduction.md)
