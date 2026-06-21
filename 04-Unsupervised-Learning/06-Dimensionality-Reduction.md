# 📉 Dimensionality Reduction & Visualization

> **Prerequisites**: Linear Algebra, Eigenvectors, Feature Scaling | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Curse of Dimensionality](#1-the-curse-of-dimensionality)
2. [Principal Component Analysis (PCA)](#2-principal-component-analysis-pca)
    - [Learning Progression](#learning-progression-pca)
    - [The Mathematics (Eigen-decomposition)](#the-mathematics-eigen-decomposition)
    - [Python Implementation (From Scratch)](#python-implementation-from-scratch)
3. [t-Distributed Stochastic Neighbor Embedding (t-SNE)](#3-t-distributed-stochastic-neighbor-embedding-t-sne)
    - [Learning Progression](#learning-progression-t-sne)
    - [The Mathematics (KL Divergence)](#the-mathematics-kl-divergence)
4. [Comparing Techniques & Practical Guide](#4-comparing-techniques--practical-guide)
5. [Mini-Project: Image Compression (PCA)](#5-mini-project-image-compression-pca)
6. [Mini-Project: Dimensionality Reduction Pipeline](#6-mini-project-dimensionality-reduction-pipeline)

---

## 1. The Curse of Dimensionality

As the number of features (dimensions) in a dataset increases, the volume of the feature space increases exponentially. This leads to the **Curse of Dimensionality**:
1. **Distance becomes meaningless**: In high dimensions, the distance between the closest point and the farthest point to a given reference point becomes almost identical. Algorithms relying on Euclidean distance (KNN, K-Means) fail completely.
2. **Overfitting**: More features mean more parameters for a model to learn. You need exponentially more data to prevent overfitting in high dimensions.
3. **Computational Cost**: Matrix operations scale at $O(n^3)$ relative to the number of features.

**Goal**: Map data $\mathbf{X} \in \mathbb{R}^{N \times D}$ to a lower-dimensional space $\mathbf{Z} \in \mathbb{R}^{N \times d}$ (where $d \ll D$) while losing as little structural information as possible.

---

## 2. Principal Component Analysis (PCA)

### Learning Progression: PCA

**🟢 Beginner (Intuition & Analogy)**
Imagine you have a 3D physical model of a teapot, and you want to take a 2D photograph of it. If you take a photo from the front, it looks like a teapot. If you take a photo from exactly the top down, it might just look like a circle (losing all the information about the spout and handle). 
PCA is an algorithm that automatically rotates the teapot to find the **best possible angle** to take the photo so that the resulting 2D image captures the *maximum amount of information* (variance) about the 3D shape. 

**🟡 Intermediate (Concept & Reasoning)**
PCA is a **linear** dimensionality reduction technique. It finds a new set of axes (Principal Components) that are orthogonal (perpendicular) to each other.
- **PC1 (First Principal Component)**: The line passing through the data that captures the absolute maximum variance.
- **PC2**: A line perpendicular to PC1 that captures the second most variance.
By keeping only the first few PCs, you compress the data while retaining the majority of the variation.

**🔴 Advanced (Mathematical Depth)**
We want to find a unit projection vector $\mathbf{w}$ such that the variance of the projected data $\mathbf{X}_c \mathbf{w}$ is maximized, where $\mathbf{X}_c$ is the mean-centered data.

The variance of the projected data is $\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$, where $\mathbf{\Sigma} = \frac{1}{N-1} \mathbf{X}_c^T \mathbf{X}_c$ is the covariance matrix.
Using a Lagrange multiplier $\lambda$ to enforce $||\mathbf{w}|| = 1$, we take the derivative and set it to zero, resulting in:
$$ \mathbf{\Sigma}\mathbf{w} = \lambda\mathbf{w} $$
This is the classic **Eigenvalue equation**! 
- The optimal projection direction $\mathbf{w}$ is the **eigenvector** of the covariance matrix.
- The variance captured is exactly the **eigenvalue** $\lambda$.

### Python Implementation (From Scratch)

```python
import numpy as np
import matplotlib.pyplot as plt

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # 1. Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # 2. Covariance matrix (features x features)
        # np.cov expects features as rows, so we transpose X
        cov_matrix = np.cov(X.T)

        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 4. Sort eigenvectors by eigenvalues in descending order
        # eigenvectors are returned as columns
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # 5. Store first n_components
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        # Project data
        X = X - self.mean
        return np.dot(X, self.components.T)

# Test on 2D -> 1D reduction
np.random.seed(42)
X_test = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

pca = PCAFromScratch(n_components=1)
pca.fit(X_test)
X_projected = pca.transform(X_test)

print(f"Original shape: {X_test.shape}")
print(f"Projected shape: {X_projected.shape}")
```

---

## 3. t-Distributed Stochastic Neighbor Embedding (t-SNE)

### Learning Progression: t-SNE

**🟢 Beginner (Intuition & Analogy)**
If PCA is like taking a photo of a solid object from the best angle, t-SNE is like unraveling a crumpled up ball of string. 
Imagine a map of cities in the USA. You know the exact driving distance between every pair of cities. Now, you must draw these cities on a blank piece of paper so that cities that are close in real life are close on the paper, and cities far in real life are far on the paper. t-SNE acts like an artist drawing this map, constantly shifting the cities around until the map accurately reflects the real-world neighborhoods.

**🟡 Intermediate (Concept & Reasoning)**
PCA is linear, meaning it fails on complex, curved manifolds (like a Swiss Roll dataset).
**t-SNE** is **non-linear**. It prioritizes preserving **local structure** (keeping close points close) rather than global variance.
It is primarily used for **visualization** (reducing 1000D data down to 2D or 3D) to see if distinct clusters exist.

**🔴 Advanced (Mathematical Depth)**
1. **High-Dim Probabilities**: t-SNE converts high-dimensional Euclidean distances between points into conditional probabilities $p_{j|i}$ that represent similarities. A Gaussian distribution is used here.
2. **Low-Dim Probabilities**: It creates a counterpart low-dimensional space and computes similarities $q_{ij}$. It uses a **Student's t-distribution** (which has heavier tails) to compute these. The heavy tails solve the "crowding problem" by pushing moderately far points much further away in the low-dimensional map.
3. **KL Divergence**: It uses Gradient Descent to minimize the Kullback-Leibler (KL) divergence between the high-dimensional probability distribution $P$ and the low-dimensional distribution $Q$.

---

## 4. Comparing Techniques & Practical Guide

| Feature | PCA | t-SNE |
|---------|-----|-------|
| **Type** | Linear | Non-linear |
| **Preserves** | Global Variance / Distance | Local structure (neighborhoods) |
| **Speed** | Extremely Fast ($O(D^3)$ or less) | Very Slow ($O(N \log N)$) |
| **Use Case** | Feature Extraction, Preprocessing, Image Compression | 2D/3D Data Visualization |
| **Transform New Data?**| Yes (`pca.transform(X_new)`) | No. Cannot map new data points |

---

## 5. Mini-Project: Image Compression (PCA)

**Goal**: Use PCA to compress high-dimensional image data and reconstruct it, visually demonstrating information retention.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# 1. Load Data (400 images, 64x64 pixels = 4096 dimensions)
faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=42)
print(f"Original Data Shape: {faces.shape}") # (400, 4096)

# 2. Fit PCA to retain 95% of the variance
# Notice we pass a float between 0 and 1! Scikit-learn automatically calculates how many PCs are needed.
pca = PCA(n_components=0.95, random_state=42)
faces_pca = pca.fit_transform(faces)

print(f"Compressed Data Shape: {faces_pca.shape}")
print(f"Dimensions reduced from 4096 -> {pca.n_components_} while keeping 95% of variance!")

# 3. Reconstruct Images (from 123D back to 4096D)
faces_reconstructed = pca.inverse_transform(faces_pca)

# 4. Visualize
fig, axes = plt.subplots(2, 5, figsize=(15, 6), subplot_kw={'xticks':[], 'yticks':[]})
for i in range(5):
    # Original
    axes[0, i].imshow(faces[i].reshape(64, 64), cmap='gray')
    axes[0, i].set_title(f"Original")
    # Reconstructed
    axes[1, i].imshow(faces_reconstructed[i].reshape(64, 64), cmap='gray')
    axes[1, i].set_title(f"Reconstructed ({pca.n_components_} PCs)")

plt.suptitle("PCA Image Compression", fontsize=16, fontweight='bold')
plt.savefig('pca_image_compression.png', dpi=150)
# plt.show()
```

---

## 6. Mini-Project: Dimensionality Reduction Pipeline

**Goal**: High-dimensional datasets are computationally heavy for t-SNE. The industry standard pipeline is to **first use PCA to reduce dimensions to ~50, and then apply t-SNE to reduce it to 2 for visualization.**

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

# 1. Load Digits Dataset (8x8 images = 64 dimensions)
digits = load_digits()
X, y = digits.data, digits.target

# 2. Step 1: PCA down to 30 dimensions
pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X)
print(f"PCA preserved {sum(pca.explained_variance_ratio_):.2f}% of variance in 30 dims.")

# 3. Step 2: t-SNE down to 2 dimensions for visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
t0 = time.time()
X_tsne = tsne.fit_transform(X_pca) # We feed it the PCA output!
print(f"t-SNE completed in {time.time() - t0:.2f} seconds.")

# 4. Visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15, alpha=0.8)
plt.legend(*scatter.legend_elements(), title="Digits", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("PCA + t-SNE Pipeline on Digits Dataset", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('tsne_pipeline.png', dpi=150)
# plt.show()
```

### Output Interpretation
The resulting t-SNE plot will show 10 highly distinct, separated clusters corresponding to the digits 0-9. If we had only used PCA to reduce to 2 dimensions, the numbers would have been heavily overlapping. t-SNE successfully unravels the complex 64-dimensional manifold into a beautiful 2D map.

---

[← Gaussian Mixture Models](./05-Gaussian-Mixture-Models.md) | [Back to Index](../README.md) | [Next: Anomaly Detection →](./07-Anomaly-Detection.md)
