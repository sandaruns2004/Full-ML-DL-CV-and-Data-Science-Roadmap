# 📉 Dimensionality Reduction — PCA, t-SNE, UMAP, SVD

> **Prerequisites**: Linear Algebra, Eigenvectors, Feature Scaling | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Curse of Dimensionality](#1-the-curse-of-dimensionality)
2. [Principal Component Analysis (PCA) — The Math](#2-principal-component-analysis-pca--the-math)
3. [Singular Value Decomposition (SVD) connection to PCA](#3-singular-value-decomposition-svd-connection-to-pca)
4. [PCA Implementation and Image Compression](#4-pca-implementation-and-image-compression)
5. [t-SNE (t-Distributed Stochastic Neighbor Embedding)](#5-t-sne-t-distributed-stochastic-neighbor-embedding)
6. [UMAP (Uniform Manifold Approximation and Projection)](#6-umap-uniform-manifold-approximation-and-projection)
7. [Comparing Techniques & Practical Guide](#7-comparing-techniques--practical-guide)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. The Curse of Dimensionality

As the number of features (dimensions) increases, the volume of the feature space increases exponentially. This leads to several problems:
1. **Distance becomes meaningless**: In high dimensions, the distance between any two points tends to converge, making distance-based algorithms (like KNN or K-Means) fail.
2. **Overfitting**: More features mean more parameters to learn, requiring exponentially more data to prevent overfitting.
3. **Computational Cost**: Matrix operations scale roughly as $O(n^3)$ for feature count $n$.

**Goal of Dimensionality Reduction**: Map data $\mathbf{X} \in \mathbb{R}^{N \times D}$ to a lower-dimensional space $\mathbf{Z} \in \mathbb{R}^{N \times d}$ (where $d \ll D$) while preserving the most important structural properties of the data.

---

## 2. Principal Component Analysis (PCA) — The Math

PCA is a linear dimensionality reduction technique. It finds a set of orthogonal axes (Principal Components) that capture the maximum variance in the data.

### Mathematical Derivation

Let $\mathbf{X}$ be an $N \times D$ data matrix. First, we **center the data**:
$$\mathbf{X}_{c} = \mathbf{X} - \boldsymbol{\mu}$$

We want to find a unit projection vector $\mathbf{w}$ ($||\mathbf{w}|| = 1$) such that the variance of the projected data $\mathbf{X}_c \mathbf{w}$ is maximized.

The variance of the projected data is:
$$\text{Var} = \frac{1}{N-1} (\mathbf{X}_c \mathbf{w})^T (\mathbf{X}_c \mathbf{w}) = \frac{1}{N-1} \mathbf{w}^T \mathbf{X}_c^T \mathbf{X}_c \mathbf{w}$$

Notice that $\mathbf{\Sigma} = \frac{1}{N-1} \mathbf{X}_c^T \mathbf{X}_c$ is the covariance matrix of $\mathbf{X}$. Thus, we want to maximize:
$$\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$
subject to the constraint $\mathbf{w}^T \mathbf{w} = 1$.

We use a **Lagrange multiplier** $\lambda$:
$$L(\mathbf{w}, \lambda) = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} - \lambda(\mathbf{w}^T \mathbf{w} - 1)$$

Taking the derivative with respect to $\mathbf{w}$ and setting it to zero:
$$\frac{\partial L}{\partial \mathbf{w}} = 2\mathbf{\Sigma}\mathbf{w} - 2\lambda\mathbf{w} = 0$$
$$\mathbf{\Sigma}\mathbf{w} = \lambda\mathbf{w}$$

This is the classic **Eigenvalue equation**! 
- The projection direction $\mathbf{w}$ that maximizes variance is the **eigenvector** of the covariance matrix.
- The amount of variance captured in that direction is exactly the corresponding **eigenvalue** $\lambda$.

### Multi-Dimensional Projection
To project into $d$ dimensions, we select the $d$ eigenvectors with the largest eigenvalues, forming a projection matrix $\mathbf{W} \in \mathbb{R}^{D \times d}$.
$$\mathbf{Z} = \mathbf{X}_c \mathbf{W}$$

---

## 3. Singular Value Decomposition (SVD) connection to PCA

Computing the covariance matrix $\mathbf{X}_c^T \mathbf{X}_c$ and its eigendecomposition is computationally expensive ($O(D^3)$) and numerically unstable.

Instead, modern libraries (like scikit-learn) use **Singular Value Decomposition (SVD)** directly on $\mathbf{X}_c$.

SVD factorizes the centered data matrix:
$$\mathbf{X}_c = \mathbf{U} \mathbf{S} \mathbf{V}^T$$
where:
- $\mathbf{U}$ ($N \times N$) contains the left singular vectors.
- $\mathbf{S}$ ($N \times D$) is a diagonal matrix of singular values $s_i$.
- $\mathbf{V}^T$ ($D \times D$) contains the right singular vectors.

The right singular vectors $\mathbf{V}$ are exactly the **Principal Components** (eigenvectors of $\mathbf{\Sigma}$).
The eigenvalues are related to singular values by: $\lambda_i = \frac{s_i^2}{N-1}$.

Therefore, the PCA projection is simply:
$$\mathbf{Z} = \mathbf{X}_c \mathbf{V}_d = (\mathbf{U} \mathbf{S} \mathbf{V}^T) \mathbf{V}_d = \mathbf{U}_d \mathbf{S}_d$$

---

## 4. PCA Implementation and Image Compression

Let's implement PCA to compress images. We will use the famous "Eigenfaces" concept.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Load dataset (400 images of faces, 64x64 pixels = 4096 dimensions)
faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=42)
n_samples, n_features = faces.shape

print(f"Original shape: {faces.shape}") # (400, 4096)

# We want to compress 4096 dimensions down to just enough to keep 95% variance
pca = PCA(n_components=0.95, svd_solver='full')
faces_pca = pca.fit_transform(faces)

print(f"Reduced shape: {faces_pca.shape}") # Will be around (400, ~120)
print(f"Number of components to retain 95% variance: {pca.n_components_}")

# Reconstruct images from the reduced dimensions
faces_reconstructed = pca.inverse_transform(faces_pca)

# Plot Original vs Reconstructed
fig, axes = plt.subplots(2, 5, figsize=(15, 6),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i in range(5):
    # Original
    axes[0, i].imshow(faces[i].reshape(64, 64), cmap='gray')
    axes[0, i].set_title(f"Original")
    
    # Reconstructed
    axes[1, i].imshow(faces_reconstructed[i].reshape(64, 64), cmap='gray')
    axes[1, i].set_title(f"Reconstructed\n({pca.n_components_} PCs)")

plt.suptitle("PCA Image Compression (4096D → ~120D)", fontsize=16, fontweight='bold')
plt.savefig('pca_faces.png', dpi=150)
plt.show()

# Visualize the first few Principal Components (Eigenfaces)
fig, axes = plt.subplots(1, 5, figsize=(15, 3), subplot_kw={'xticks':[], 'yticks':[]})
for i in range(5):
    eigenface = pca.components_[i].reshape(64, 64)
    axes[i].imshow(eigenface, cmap='viridis')
    axes[i].set_title(f"Eigenface {i+1}")
plt.suptitle("First 5 Principal Components", fontsize=16, fontweight='bold')
plt.show()
```

---

## 5. t-SNE (t-Distributed Stochastic Neighbor Embedding)

While PCA is linear and preserves global variance, **t-SNE** is non-linear and preserves **local neighborhood structure**. It is the gold standard for visualizing high-dimensional data in 2D or 3D.

### The Math of t-SNE

**Step 1: Compute Probabilities in High-Dimensional Space**
For every pair of points $\mathbf{x}_i$ and $\mathbf{x}_j$, compute the conditional probability that $\mathbf{x}_i$ would pick $\mathbf{x}_j$ as its neighbor, assuming a Gaussian distribution centered at $\mathbf{x}_i$:
$$p_{j|i} = \frac{\exp(-||\mathbf{x}_i - \mathbf{x}_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||\mathbf{x}_i - \mathbf{x}_k||^2 / 2\sigma_i^2)}$$
Make it symmetric: $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$

**Step 2: Compute Probabilities in Low-Dimensional Space**
Map the points to low-dimensional counterparts $\mathbf{y}_i, \mathbf{y}_j$. Here, use a **Student's t-distribution** (with 1 degree of freedom) instead of Gaussian. This "heavy tail" solves the *crowding problem*:
$$q_{ij} = \frac{(1 + ||\mathbf{y}_i - \mathbf{y}_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||\mathbf{y}_k - \mathbf{y}_l||^2)^{-1}}$$

**Step 3: Minimize KL Divergence**
Use Gradient Descent to adjust $\mathbf{y}_i$ to minimize the Kullback-Leibler (KL) divergence between $P$ and $Q$:
$$KL(P || Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# t-SNE is computationally heavy. Best practice: reduce to ~50D with PCA first if dimensions are huge.
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15, alpha=0.8)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title("t-SNE Projection of Digits", fontsize=16, fontweight='bold')
plt.axis('off')
plt.savefig('tsne_digits.png', dpi=150)
plt.show()
```

---

## 6. UMAP (Uniform Manifold Approximation and Projection)

UMAP is a modern, mathematically rigorous algorithm based on Riemannian geometry and algebraic topology. It often yields better results than t-SNE and is significantly faster.

**Key Mathematical Ideas of UMAP:**
1. **Simplicial Complex**: Models the high-dimensional data as a fuzzy topological structure.
2. **Local Manifold Approximations**: Assumes data is uniformly distributed on a local Riemannian manifold, allowing distance metrics to adapt to local density.
3. **Cross-Entropy Optimization**: Uses cross-entropy to align the fuzzy topological structure of the low-dimensional projection with the high-dimensional one.

Unlike t-SNE, UMAP preserves **both local and global structure**, and allows for out-of-sample mapping (transforming new data).

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
X_umap = reducer.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=15, alpha=0.8)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title("UMAP Projection of Digits", fontsize=16, fontweight='bold')
plt.axis('off')
plt.savefig('umap_digits.png', dpi=150)
plt.show()
```

---

## 7. Comparing Techniques & Practical Guide

| Feature | PCA | t-SNE | UMAP |
|---------|-----|-------|------|
| **Mathematical Basis** | Linear Algebra, SVD | Probability, KL Divergence | Topology, Riemannian Geometry |
| **Type** | Linear | Non-linear | Non-linear |
| **Preserves** | Global Variance / Distance | Local structure (neighborhoods) | Local + Global structure |
| **Speed** | Extremely Fast ($O(D^3)$ or less) | Very Slow ($O(N \log N)$) | Fast ($O(N)$) |
| **Transform New Data?**| Yes (`.transform()`) | No | Yes (`.transform()`) |
| **Use Case** | Preprocessing, Compression, Noise Reduction | Visualization | Visualization, Preprocessing for ML |

---

## 8. Project Ideas & What's Next

### Project Ideas
- 🟢 **Genomics Clustering**: Download a single-cell RNA sequencing dataset. Run PCA to reduce from 20,000 genes to 50 dimensions, then use UMAP to visualize the 50D data in 2D to identify distinct cell types.
- 🔴 **t-SNE from Scratch**: Implement the heavy-tail t-distribution math and gradient descent to build a simple t-SNE implementation.

### What's Next
| Next | Why |
|------|-----|
| [Anomaly Detection](./03-Anomaly-Detection.md) | Moving to unsupervised techniques for finding outliers, leveraging concepts like density and PCA reconstruction error. |

---

[← Clustering](./01-Clustering.md) | [Back to Index](../README.md) | [Next: Anomaly Detection →](./03-Anomaly-Detection.md)
