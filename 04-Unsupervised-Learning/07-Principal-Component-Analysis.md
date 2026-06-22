# 📉 Principal Component Analysis (PCA)

> **Prerequisites**: Linear Algebra, Eigenvectors, Feature Scaling | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 📋 Table of Contents
1. [The Curse of Dimensionality](#1-the-curse-of-dimensionality)
2. [PCA Intuition](#2-pca-intuition)
3. [The Mathematics of PCA](#3-the-mathematics-of-pca)
4. [Python Implementation (From Scratch)](#4-python-implementation-from-scratch)
5. [Mini-Project: Image Compression (PCA)](#5-mini-project-image-compression-pca)

---

## 1. The Curse of Dimensionality

As the number of features (dimensions) in a dataset increases, the volume of the feature space increases exponentially. This leads to the **Curse of Dimensionality**:
1. **Distance becomes meaningless**: In high dimensions, the distance between the closest point and the farthest point to a given reference point becomes almost identical. Algorithms relying on Euclidean distance (KNN, K-Means) struggle.
2. **Overfitting**: More features mean more parameters to learn, requiring exponentially more data to prevent overfitting.
3. **Computational Cost**: Matrix operations scale at $O(D^3)$ relative to the number of features $D$.

**Goal**: Map data $\mathbf{X} \in \mathbb{R}^{N \times D}$ to a lower-dimensional space $\mathbf{Z} \in \mathbb{R}^{N \times d}$ (where $d \ll D$) while losing as little variance (structural information) as possible.

---

## 2. PCA Intuition

### 🟢 Beginner
Imagine you have a 3D physical model of a teapot, and you want to take a 2D photograph of it. If you take a photo from exactly the top down, it might just look like a circle (losing all the information about the spout and handle). 
PCA is an algorithm that automatically rotates the teapot to find the **best possible angle** to take the photo so that the resulting 2D image captures the *maximum amount of information* (variance) about the 3D shape. 

### 🟡 Intermediate
PCA is a **linear** dimensionality reduction technique. It finds a new set of orthogonal (perpendicular) axes:
- **PC1 (First Principal Component)**: The line passing through the data that captures the absolute maximum variance.
- **PC2 (Second Principal Component)**: A line perpendicular to PC1 that captures the second most variance.

---

## 3. The Mathematics of PCA

### 🔴 Advanced
We want to find a unit projection vector $\mathbf{w}$ ($||\mathbf{w}|| = 1$) such that the variance of the projected data $\mathbf{X}_c \mathbf{w}$ is maximized, where $\mathbf{X}_c$ is the mean-centered data matrix.

The variance of the projected data is:

$$\text{Var}(\mathbf{X}_c \mathbf{w}) = \frac{1}{N-1} (\mathbf{X}_c \mathbf{w})^T (\mathbf{X}_c \mathbf{w}) = \mathbf{w}^T \left( \frac{\mathbf{X}_c^T \mathbf{X}_c}{N-1} \right) \mathbf{w} = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$

where $\mathbf{\Sigma}$ is the covariance matrix of the data.
Using a Lagrange multiplier $\lambda$ to enforce $||\mathbf{w}||_2^2 = \mathbf{w}^T\mathbf{w} = 1$, we define the Lagrangian:

$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} - \lambda (\mathbf{w}^T \mathbf{w} - 1)$$

Taking the derivative with respect to $\mathbf{w}$ and setting it to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\mathbf{\Sigma}\mathbf{w} - 2\lambda\mathbf{w} = 0 \implies \mathbf{\Sigma}\mathbf{w} = \lambda\mathbf{w}$$

This is the classic **Eigenvalue equation**!
- The optimal projection directions $\mathbf{w}$ are the **eigenvectors** of the covariance matrix $\mathbf{\Sigma}$.
- The variance captured by each principal component is its corresponding **eigenvalue** $\lambda$.

---

## 4. Python Implementation (From Scratch)

Here is how to calculate PCA from scratch using NumPy:

```python
import numpy as np

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # 1. Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 4. Sort eigenvectors by eigenvalues descending
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[idxs][0:self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)
```

---

## 5. Mini-Project: Image Compression (PCA)

```python
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# 1. Load Olivetti Faces (4096 dimensions)
faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=42)

# 2. Fit PCA keeping 95% of variance
pca = PCA(n_components=0.95, random_state=42)
faces_pca = pca.fit_transform(faces)

# 3. Reconstruct
faces_reconstructed = pca.inverse_transform(faces_pca)

# 4. Plot original vs reconstructed
fig, axes = plt.subplots(2, 5, figsize=(12, 5), subplot_kw={'xticks':[], 'yticks':[]})
for i in range(5):
    axes[0, i].imshow(faces[i].reshape(64, 64), cmap='gray')
    axes[0, i].set_title("Original")
    axes[1, i].imshow(faces_reconstructed[i].reshape(64, 64), cmap='gray')
    axes[1, i].set_title(f"Reconstructed ({pca.n_components_} PCs)")
plt.suptitle("PCA Image Compression", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

[← Gaussian Mixture Models (GMM)](06-Gaussian-Mixture-Models.md) | [Back to Index](../README.md) | [Next: t-SNE (t-Distributed Stochastic Neighbor Embedding) →](08-tSNE.md)
