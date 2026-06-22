# 🗺️ t-SNE (t-Distributed Stochastic Neighbor Embedding)

> **Prerequisites**: PCA | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [t-SNE Intuition](#1-t-sne-intuition)
2. [How t-SNE Works (The Mathematics)](#2-how-t-sne-works-the-mathematics)
3. [Comparing PCA vs t-SNE](#3-comparing-pca-vs-t-sne)
4. [Mini-Project: PCA + t-SNE Pipeline](#4-mini-project-pca--t-sne-pipeline)

---

## 1. t-SNE Intuition

### 🟢 Beginner
PCA is a linear technique, meaning it fails on complex, curved structures (like a Swiss Roll). 
**t-SNE** is non-linear and acts like untangling a crumpled up ball of string. Imagine you have a map of USA cities. You know the driving distance between every pair of cities. t-SNE acts like an artist drawing this map on a flat piece of paper, trying to make sure that cities close in high-dimensional space are close in 2D, and cities far away remain far away, focusing heavily on preserving local neighborhoods.

---

## 2. How t-SNE Works (The Mathematics)

### 🔴 Advanced

#### 2.1 High-Dimensional Commensurate Affinities
t-SNE converts Euclidean distances between data points $x_i$ and $x_j$ into conditional probabilities $p_{j|i}$ that represent similarity:

$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$

where $\sigma_i$ is the variance of the Gaussian centered on $x_i$, determined by a user-defined **perplexity** parameter. To handle outliers, we define symmetric joint probabilities:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$

#### 2.2 Low-Dimensional Student's t-Distribution
In the low-dimensional map $y_i, y_j$, we compute similarities $q_{ij}$. Instead of a Gaussian, t-SNE uses a **Student's t-distribution** with 1 degree of freedom (equivalent to a Cauchy distribution):

$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}$$

*Why Cauchy?* In high dimensions, space volume scales exponentially, causing points to get crowded when projected to 2D (the "crowding problem"). The heavy tails of the Student's t-distribution allow moderately distant points to be mapped much further apart in the 2D space without blowing up the gradients.

#### 2.3 Optimization via KL Divergence
We position the low-dimensional points $y$ by minimizing the Kullback-Leibler (KL) divergence between $P$ and $Q$ using gradient descent:

$$KL(P || Q) = \sum_{i \neq j} p_{ij} \log \left(\frac{p_{ij}}{q_{ij}}\right)$$

---

## 3. Comparing PCA vs t-SNE

| Feature | PCA | t-SNE |
| :--- | :--- | :--- |
| **Type** | Linear | Non-linear |
| **Preserves** | Global Variance / Distance | Local structure (neighborhoods) |
| **Speed** | Extremely Fast ($O(D^3)$) | Very Slow ($O(N \log N)$ or $O(N^2)$) |
| **Transform New Data?**| Yes (`pca.transform(X_new)`) | No (must retrain entirely) |

---

## 4. Mini-Project: PCA + t-SNE Pipeline

t-SNE is computationally expensive for high dimensions. The industry standard pipeline is to **first use PCA to reduce dimensions to ~30-50, and then apply t-SNE to reduce to 2 for visualization**.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

# 1. Load Digits Dataset (64 dimensions)
digits = load_digits()
X, y = digits.data, digits.target

# 2. Step 1: PCA to 30 dimensions
pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X)

# 3. Step 2: t-SNE down to 2 dimensions
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
t0 = time.time()
X_tsne = tsne.fit_transform(X_pca) # We feed it the PCA output
print(f"t-SNE finished in {time.time() - t0:.2f} seconds.")

# 4. Visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15, alpha=0.8)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title("PCA + t-SNE Pipeline on Digits", fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()
```

---

[← Principal Component Analysis (PCA)](07-Principal-Component-Analysis.md) | [Back to Index](../README.md) | [Next: UMAP (Uniform Manifold Approximation and Projection) →](09-UMAP-Introduction.md)
