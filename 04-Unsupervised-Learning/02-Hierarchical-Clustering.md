# 🌳 Hierarchical Clustering

> **Prerequisites**: Distance Metrics, K-Means | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Learning Progression: Hierarchical Clustering](#1-learning-progression-hierarchical-clustering)
2. [Step-by-Step Algorithm Working](#2-step-by-step-algorithm-working)
3. [Linkage Methods (The Math)](#3-linkage-methods-the-math)
4. [Python Implementation (SciPy & Dendrograms)](#4-python-implementation-scipy--dendrograms)
5. [Scikit-Learn Implementation](#5-scikit-learn-implementation)
6. [Real-World Applications](#6-real-world-applications)
7. [Advantages & Limitations](#7-advantages--limitations)

---

## 1. Learning Progression: Hierarchical Clustering

### 🟢 Beginner (Intuition & Analogy)
Imagine you have 8 individual puzzle pieces scattered on a table. 
- You look for the 2 pieces that fit together the best (closest to each other) and join them. Now you have 6 individual pieces and 1 small cluster of two pieces.
- You again look at everything on the table and join the 2 closest items (maybe two individual pieces, or maybe you add a piece to your existing cluster).
- You keep doing this, building bigger and bigger chunks of the puzzle, until all the pieces are connected into one single, giant puzzle.
- If you wanted exactly 3 clusters, you could just stop joining pieces right before the final 2 steps!

### 🟡 Intermediate (Concept & Reasoning)
Hierarchical clustering builds a hierarchy of clusters. Unlike K-Means, **you do not need to specify $K$ in advance**.
There are two main types:
1. **Agglomerative (Bottom-Up)**: Start with each point as its own cluster. At each step, merge the two most similar clusters until only one giant cluster remains. (Most common).
2. **Divisive (Top-Down)**: Start with all points in one giant cluster. At each step, split the most heterogeneous cluster until all points are isolated.

The result is represented as a tree-like diagram called a **Dendrogram**. By drawing a horizontal line across the dendrogram at a specific height, you can choose exactly how many clusters you want.

### 🔴 Advanced (Algorithmic Complexity)
Agglomerative clustering requires calculating the pairwise distance between all clusters at every step.
- The time complexity is typically $O(N^3)$, although it can be optimized to $O(N^2 \log N)$ with priority queues.
- Space complexity is $O(N^2)$ because we must store the distance matrix.
This makes hierarchical clustering entirely unsuitable for massive datasets (e.g., millions of rows), unlike K-Means which is $O(N)$.

---

## 2. Step-by-Step Algorithm Working (Agglomerative)

1. **Initialization**: Treat each of the $N$ data points as an individual cluster. Compute an $N \times N$ distance matrix.
2. **Find Closest**: Find the two clusters with the shortest distance between them.
3. **Merge**: Merge these two clusters into a new, larger cluster.
4. **Update Distance Matrix**: Calculate the distance from the *new* cluster to all other existing clusters (this depends on the "Linkage Method").
5. **Repeat**: Repeat steps 2-4 until all data points are merged into a single cluster containing $N$ points.

---

## 3. Linkage Methods (The Math)

How do you calculate the distance between two *clusters* (which contain multiple points)? This is defined by the **Linkage Criterion**. Let cluster $A$ have points $a$ and cluster $B$ have points $b$.

| Linkage Method | Mathematical Definition | Characteristics |
| :--- | :--- | :--- |
| **Single Linkage** | $D(A, B) = \min \{ d(a, b) : a \in A, b \in B \}$ | Distance between the two *closest* points. Can cause "chaining" (long, elongated clusters). |
| **Complete Linkage** | $D(A, B) = \max \{ d(a, b) : a \in A, b \in B \}$ | Distance between the two *farthest* points. Creates compact, spherical clusters. |
| **Average Linkage** | $D(A, B) = \frac{1}{\vert A \vert \vert B \vert} \sum_{a \in A} \sum_{b \in B} d(a, b)$ | Average distance between all pairs. Good balance between single and complete. |
| **Ward's Method** | Minimizes the total within-cluster variance. | At each step, merge clusters that result in the smallest increase in WCSS. Very effective and widely used. |

---

## 4. Python Implementation (SciPy & Dendrograms)

The `scipy` library is excellent for Hierarchical clustering because of its powerful `dendrogram` plotting functions.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# 1. Generate Data
X, _ = make_blobs(n_samples=50, centers=3, cluster_std=1.0, random_state=42)

# 2. Perform Hierarchical/Agglomerative Clustering
# The linkage() function returns an (N-1) x 4 matrix Z.
# Z[i] tells us which clusters were merged at step i, the distance between them, and number of original observations.
Z = linkage(X, method='ward', metric='euclidean')

# 3. Plot the Dendrogram
plt.figure(figsize=(10, 6))
plt.title('Hierarchical Clustering Dendrogram (Ward)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# Draw the dendrogram
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)

# Draw a cut-off line
plt.axhline(y=15, color='r', linestyle='--', label='Cut-off line (3 clusters)')
plt.legend()
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=150)
# plt.show()
```

### Reading the Dendrogram
- The **x-axis** represents the individual data points.
- The **y-axis** represents the distance (or dissimilarity) between clusters.
- The height at which two branches merge indicates the distance between the two clusters being merged. 
- A **large vertical gap** between merges suggests an optimal place to "cut" the tree to form clusters.

---

## 5. Scikit-Learn Implementation

For actually assigning cluster labels to your dataset (rather than just visualization), use `scikit-learn`.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons

# Let's try it on non-spherical data!
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Single Linkage is actually good at following chains (like moons)
agg_cluster = AgglomerativeClustering(n_clusters=2, linkage='single')
labels = agg_cluster.fit_predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, edgecolors='k')
plt.title("Agglomerative Clustering (Single Linkage) on Moons", fontsize=14, fontweight='bold')
plt.savefig('agg_moons.png', dpi=150)
# plt.show()
```

---

## 6. Real-World Applications

- 🧬 **Bioinformatics**: Building evolutionary trees (phylogenetics) to see which species are most closely related, or clustering genes with similar expression patterns.
- 📚 **Taxonomy Construction**: Automatically organizing articles or web pages into a hierarchy of topics and subtopics.
- 🏦 **Financial Portfolio Risk**: Grouping stocks by historical price correlation to build a diversified portfolio.

---

## 7. Advantages & Limitations

### ✅ Advantages
- **No Need to Guess K**: The dendrogram gives you a complete picture of the data's structure at all levels of granularity. You can choose $K$ *after* seeing the tree.
- **Deterministic**: Unlike K-Means, which relies on random initialization, hierarchical clustering always yields the exact same result for a given dataset and linkage method.
- **Handles Non-Spherical Shapes**: With `single` linkage, it can cluster arbitrarily shaped data (like the half-moons above).

### ❌ Limitations
- **Computationally Expensive**: $O(N^3)$ time and $O(N^2)$ space makes it impossible to use on very large datasets (e.g., $N > 10,000$).
- **Irreversible**: Once two clusters are merged, they cannot be undone in later steps. If a bad merge happens early on, it corrupts the whole tree.
- **Sensitive to Noise/Outliers**: Especially when using `single` linkage, outliers can cause severe chaining issues.

---

[← K-Means Clustering](./01-K-Means-Clustering.md) | [Back to Index](../README.md) | [Next: DBSCAN Clustering →](./03-DBSCAN-Clustering.md)
