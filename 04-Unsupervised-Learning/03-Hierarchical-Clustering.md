# 🌳 Hierarchical Clustering

> **Difficulty**: ⭐⭐☆☆☆ Intermediate | **Prerequisites**: K-Means, Distance Metrics | **Estimated Reading Time**: 25 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Core Mathematics](#3-core-mathematics)
4. [Visual Explanation](#4-visual-explanation)
5. [Algorithm Workflow](#5-algorithm-workflow)
6. [From Scratch Implementation](#6-from-scratch-implementation)
7. [NumPy Implementation](#7-numpy-implementation)
8. [Scikit-Learn Implementation](#8-scikit-learn-implementation)
9. [Hyperparameter Deep Dive](#9-hyperparameter-deep-dive)
10. [Visualization Lab](#10-visualization-lab)
11. [Failure Cases](#11-failure-cases)
12. [Industry Applications](#12-industry-applications)
13. [Hands-On Exercises](#13-hands-on-exercises)
14. [Further Reading](#14-further-reading)
15. [What's Next?](#15-whats-next)

---

## 1. What Problem Does This Solve?

K-Means is powerful, but it forces you to guess the number of clusters $K$ upfront. Furthermore, it assumes that all clusters are strictly partitioned at a single flat level.

In many real-world scenarios, data is inherently hierarchical. For example, species in biology (Kingdom $\rightarrow$ Phylum $\rightarrow$ Class $\rightarrow$ Order $\rightarrow$ Family $\rightarrow$ Genus $\rightarrow$ Species). **Hierarchical Clustering** solves the problem of finding clusters without needing to specify $K$ beforehand, and it provides a beautiful tree-like visual representation of the data's structure called a **Dendrogram**.

---

## 2. Intuition

### 🟢 Beginner
Imagine you have 100 students in a room. 
*   **Agglomerative (Bottom-Up)**: Initially, every student is their own "cluster" of 1. The two closest friends hold hands to form a group of 2. Then, the next closest pair (or a person and a pair) join hands. This continues until everyone in the room is holding hands in one giant cluster.
*   **Divisive (Top-Down)**: The opposite. Everyone starts holding hands in one giant circle. You find the weakest link between people and break it, forming two groups. You keep breaking the weakest links until everyone is standing alone.

### 🟡 Intermediate
The most common approach is **Agglomerative**. The algorithm calculates the distance between every single data point. It merges the two closest points. But then, how do you calculate the distance between a single point and a *cluster of points*? Or between two clusters? This is defined by the **Linkage Criterion**. 

### 🔴 Advanced
Hierarchical clustering does not rely on an iterative loss minimization loop like K-Means. Instead, it computes a full $N \times N$ pairwise distance matrix. The merging process builds a strict mathematical hierarchy. The Dendrogram is a plot where the y-axis represents the *distance* at which two clusters merged. Cutting the dendrogram horizontally at a specific height $\tau$ yields a flat partition of $K$ clusters.

---

## 3. Core Mathematics

### Distance Matrix
Let $X = \{x_1, \dots, x_N\}$. We first compute the pairwise distance matrix $D$ where $D_{ij} = d(x_i, x_j)$.

### Linkage Criteria
When we have two clusters $A$ and $B$, how do we define the distance $D(A, B)$ between them?

1.  **Single Linkage (MIN)**: Distance between the two *closest* points in $A$ and $B$.
    $$ D(A,B) = \min_{a \in A, b \in B} d(a, b) $$
    *Prone to "chaining" (long, thin clusters).*

2.  **Complete Linkage (MAX)**: Distance between the two *farthest* points in $A$ and $B$.
    $$ D(A,B) = \max_{a \in A, b \in B} d(a, b) $$
    *Tends to produce compact, spherical clusters.*

3.  **Average Linkage**: The average distance between all pairs in $A$ and $B$.
    $$ D(A,B) = \frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} d(a, b) $$

4.  **Ward's Method**: Merges clusters that yield the *minimum increase* in total within-cluster variance (Inertia) after merging.
    $$ \Delta(A, B) = \sum_{x \in A \cup B} ||x - \mu_{A \cup B}||^2 - \left( \sum_{a \in A} ||a - \mu_A||^2 + \sum_{b \in B} ||b - \mu_B||^2 \right) $$
    *This is the most common and robust method, highly related to K-Means' objective.*

---

## 4. Visual Explanation

```mermaid
graph TD
    subgraph Iteration 1
    A1[A] --- B1[B]
    C1[C]
    D1[D]
    end
    
    subgraph Iteration 2
    AB[Cluster AB]
    C2[C] --- D2[D]
    end
    
    subgraph Iteration 3
    AB3[Cluster AB] --- CD[Cluster CD]
    end
    
    subgraph Final
    ABCD[Cluster ABCD]
    end
    
    Iteration 1 --> Iteration 2 --> Iteration 3 --> Final
```

*Agglomerative merging process (Bottom-Up).*

---

## 5. Algorithm Workflow (Agglomerative)

1.  **Initialize**: Treat each of the $N$ data points as an individual cluster.
2.  **Compute Distances**: Calculate the distance matrix between all clusters.
3.  **Merge**: Find the two clusters with the smallest distance based on the chosen Linkage criterion and merge them into a single cluster.
4.  **Update**: Update the distance matrix to reflect the distance between the new cluster and the remaining clusters.
5.  **Repeat**: Repeat steps 3 and 4 until all points are contained within a single large cluster.
6.  **Cut**: Slice the resulting Dendrogram at the desired depth to get $K$ clusters.

---

## 6. From Scratch Implementation

```python
import numpy as np

def single_linkage_scratch(X):
    n = len(X)
    clusters = [[i] for i in range(n)]
    
    # Precompute pairwise distances
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
            else:
                dist_matrix[i, j] = np.inf # Ignore self-distance
                
    # Iteratively merge
    history = []
    while len(clusters) > 1:
        # Find minimum distance between any two clusters
        min_dist = np.inf
        merge_pair = (-1, -1)
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Single linkage distance
                d = min([dist_matrix[c1, c2] for c1 in clusters[i] for c2 in clusters[j]])
                if d < min_dist:
                    min_dist = d
                    merge_pair = (i, j)
                    
        # Merge
        i, j = merge_pair
        history.append((clusters[i], clusters[j], min_dist))
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
        
    return history
```

---

## 7. NumPy Implementation

While writing the loops from scratch is highly inefficient $O(N^3)$, `SciPy` utilizes optimized C routines for hierarchical clustering. We typically skip direct NumPy logic here in favor of `scipy.cluster.hierarchy`.

```python
from scipy.spatial.distance import pdist

# pdist computes the condensed distance matrix (upper triangle)
# This is much faster and uses less memory
condensed_dist = pdist(X, metric='euclidean')
```

---

## 8. Scikit-Learn / SciPy Implementation

In practice, we use `SciPy` to generate the Dendrogram, and `Scikit-Learn` for pipeline integration.

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# --- 1. SciPy for Dendrogram ---
# Perform hierarchical/agglomerative clustering
Z = linkage(X, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance (Ward)")
plt.show()

# --- 2. Scikit-Learn for Predictions ---
# If we look at the dendrogram and decide we want 3 clusters:
hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
```

---

## 9. Hyperparameter Deep Dive

*   **`n_clusters` vs `distance_threshold`**: In Scikit-Learn, you can either explicitly ask for $K$ clusters (`n_clusters=3`), OR you can ask the algorithm to cut the dendrogram at a specific distance (`distance_threshold=1.5`). You cannot use both simultaneously.
*   **`linkage`**: 
    *   `ward`: Minimizes variance (similar to K-Means). Usually the best default.
    *   `single`: Good for non-spherical data but susceptible to chaining.
    *   `complete` / `average`: Good compromises that yield compact clusters.
*   **`metric`**: Usually `euclidean`, but `cosine` or `manhattan` can be used (note: `ward` strictly requires `euclidean`).

---

## 10. Visualization Lab

> **Note**: For interactive Dendrograms and Linkage comparisons, see `notebooks/02-Hierarchical-Clustering-Lab.ipynb`.

### The Dendrogram
The Dendrogram is the defining feature of Hierarchical Clustering. 
*   **X-axis**: Individual data points (or clusters of points).
*   **Y-axis**: The distance metric. The height of the vertical line connecting two clusters represents the mathematical distance between them when they merged.
*   **How to read it**: The tallest vertical line that doesn't intersect any horizontal lines indicates the largest "gap" or distance between clusters. Cutting a horizontal line through that gap gives the optimal $K$.

---

## 11. Failure Cases

1.  **Big Data (Computational Complexity)**: Agglomerative clustering requires computing an $N \times N$ distance matrix. Time complexity is $O(N^3)$ (or $O(N^2 \log N)$ optimally), and memory complexity is $O(N^2)$. It will crash your RAM on datasets larger than 30,000-50,000 rows.
2.  **No Backtracking**: Once two clusters are merged, they can never be split again later in the algorithm, even if it turns out to have been a poor choice globally.

---

## 12. Industry Applications

*   **Bioinformatics**: Building phylogenetic trees to trace evolutionary ancestry based on genetic distances.
*   **Taxonomy & NLP**: Creating hierarchies of documents or words based on semantic similarity.
*   **Finance**: Hierarchical Risk Parity (HRP) for portfolio allocation, which uses hierarchical clustering of stock correlations to distribute risk evenly.

---

## 13. Hands-On Exercises

**Easy**: Generate a small 10-point dataset. Use SciPy to plot the Dendrogram using `single` linkage. Then, change it to `ward` linkage and observe how the tree changes.
**Medium**: Train an `AgglomerativeClustering` model and a `KMeans` model on the `make_moons` dataset from Scikit-Learn. Use `single` linkage for the hierarchical model. Plot both. Which one captures the moons better?
**Hard**: Implement a function that takes the `Z` linkage matrix from SciPy and automatically finds the "longest vertical jump" to return the optimal suggested $K$.

---

## 14. Further Reading

*   *Elements of Statistical Learning* - Chapter 14.3 (Cluster Analysis)
*   SciPy Documentation: `scipy.cluster.hierarchy.linkage`
*   Scikit-Learn Documentation: `sklearn.cluster.AgglomerativeClustering`

---

## 15. What's Next?

### Summary
Hierarchical clustering builds a beautiful, interpretable tree of relationships without forcing us to guess $K$ upfront. We learned that the Linkage criterion entirely dictates how the space is merged, with Ward's method acting similarly to K-Means variance reduction. 

### Why it matters
For small to medium datasets where relationships are nested, or where visual interpretability of the cluster relationships is demanded by stakeholders, the Dendrogram is unmatched.

### Next Topic
Both K-Means and Hierarchical Clustering struggle with complex, dense geometric shapes (like nested rings) and they both force outliers into clusters. We will now learn **DBSCAN**, a density-based algorithm that beautifully handles weird shapes and completely ignores outliers.

[← K-Means Clustering](02-K-Means-Clustering.md) | [Return to Unsupervised Index](../README.md) | [Next: DBSCAN →](04-DBSCAN.md)
