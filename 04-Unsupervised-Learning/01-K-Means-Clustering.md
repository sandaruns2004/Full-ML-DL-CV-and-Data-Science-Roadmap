# 🎯 K-Means Clustering

> **Prerequisites**: Linear Algebra, Distance Metrics | **Difficulty**: ⭐⭐☆☆☆ Beginner-Intermediate

---

## 📋 Table of Contents
1. [Learning Progression: K-Means Clustering](#1-learning-progression-k-means-clustering)
2. [Step-by-Step Algorithm Working](#2-step-by-step-algorithm-working)
3. [Choosing the Right K: The Elbow Method](#3-choosing-the-right-k-the-elbow-method)
4. [Python Implementation (From Scratch)](#4-python-implementation-from-scratch)
5. [Scikit-Learn Implementation](#5-scikit-learn-implementation)
6. [Real-World Applications](#6-real-world-applications)
7. [Advantages & Limitations](#7-advantages--limitations)
8. [Mini-Project: Customer Segmentation](#8-mini-project-customer-segmentation)

---

## 1. Learning Progression: K-Means Clustering

### 🟢 Beginner (Intuition & Analogy)
Imagine you are a teacher in a school yard with 100 kids running around. You want to group them into $K=3$ teams for a game.
You toss 3 different colored flags (Red, Blue, Green) randomly onto the ground. 
1. You tell every kid: **"Run to the flag closest to you!"** (Assignment)
2. Once the kids are gathered around the flags, you move each flag to the **exact center** of the group of kids around it. (Update)
3. You tell the kids again: **"Run to the flag closest to you!"**
You keep repeating this until the flags stop moving and the teams don't change anymore. That is K-Means clustering!

### 🟡 Intermediate (Concept & Reasoning)
K-Means is a **centroid-based, unsupervised learning** algorithm. Its goal is to partition a dataset into $K$ distinct, non-overlapping subgroups (clusters).
- **Centroid**: The geometric center (mean) of all points in a cluster.
- **Objective**: To minimize the variance (or squared distance) within each cluster. Points in the same cluster should be as similar as possible, while points in different clusters should be as dissimilar as possible.
- K-Means is an **iterative** algorithm that uses **Expectation-Maximization (EM)** loosely. The "Expectation" step assigns points to the nearest centroid, and the "Maximization" step calculates the new mean to update the centroid.

### 🔴 Advanced (Mathematical Depth)
Mathematically, K-Means aims to minimize the **Within-Cluster Sum of Squares (WCSS)**, also known as **Inertia**.
Let the set of clusters be $C = \{C_1, C_2, ..., C_k\}$, and the centroid of cluster $C_k$ be $\mu_k$. The objective function $J$ is:

$$ J = \sum_{k=1}^{K} \sum_{x_i \in C_k} || x_i - \mu_k ||^2 $$

Where:
- $|| x_i - \mu_k ||^2$ is the squared Euclidean distance between a point $x_i$ and its assigned centroid $\mu_k$.
- We want to find the cluster assignments $C$ and centroids $\mu$ that minimize $J$. 

This problem is NP-hard, which is why we use the heuristic iterative algorithm (Lloyd's algorithm), which guarantees convergence to a *local* minimum, though not necessarily the *global* minimum.

---

## 2. Step-by-Step Algorithm Working

1. **Initialization**: Randomly select $K$ points from the data as the initial centroids. (Modern implementations use **K-Means++** initialization to pick initial centroids that are far away from each other to speed up convergence).
2. **Assignment Step**: Calculate the Euclidean distance from every data point to all $K$ centroids. Assign each data point to the cluster of the closest centroid.
3. **Update Step**: For each cluster, calculate the mean of all data points assigned to it. This new mean becomes the updated centroid.
4. **Convergence Check**: Repeat steps 2 and 3 until the centroids no longer move significantly or a maximum number of iterations is reached.

---

## 3. Choosing the Right K: The Elbow Method

Since K is a hyperparameter, we must choose it before running the algorithm. The **Elbow Method** is the most common technique:
1. Run K-Means for a range of $K$ values (e.g., $K=1$ to $10$).
2. For each $K$, compute the **WCSS (Inertia)**.
3. Plot the WCSS against $K$. The plot looks like an arm. 
4. Select the $K$ at the "elbow" point, where the rate of decrease in WCSS sharply slows down.

---

## 4. Python Implementation (From Scratch)

Writing K-Means from scratch builds a deep understanding of its inner workings.

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        # 1. Initialize centroids randomly from data points
        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]
        
        for _ in range(self.max_iters):
            # 2. Assignment step: find closest centroid for each point
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)
            
            # 3. Update step: calculate new centroids (means)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            # 4. Convergence check
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
                
            self.centroids = new_centroids
            
        self.labels_ = labels
        return self
        
    def _compute_distances(self, X):
        # Calculate Euclidean distances using broadcasting
        # Shape: (n_samples, k)
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

# Quick Test
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
    
    kmeans = KMeansFromScratch(k=4)
    kmeans.fit(X)
    
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], color='red', marker='X', s=200, label='Centroids')
    plt.title("K-Means from Scratch")
    plt.legend()
    # plt.show()
```

---

## 5. Scikit-Learn Implementation

In industry, we use `scikit-learn`'s optimized implementation which includes K-Means++ initialization under the hood.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate Synthetic Data
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=1.0, random_state=42)

# 2. Fit the Model
# n_init='auto' suppresses a warning and runs the optimal initialization scheme
kmeans_sklearn = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_sklearn.fit(X)

# 3. Extract parameters
labels = kmeans_sklearn.labels_
centroids = kmeans_sklearn.cluster_centers_

# 4. Visualization
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=30, alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Centroids')
plt.title("Scikit-Learn K-Means Clustering", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('kmeans_sklearn.png', dpi=150)
# plt.show()
```

---

## 6. Real-World Applications

- 🛒 **Customer Segmentation**: Grouping retail customers based on purchase history, demographics, or browsing behavior to target marketing campaigns.
- 🎨 **Image Color Quantization**: Reducing the number of distinct colors in an image by clustering similar pixel colors together.
- 📑 **Document Clustering**: Organizing a large corpus of text documents into distinct topic categories without reading them.
- 🌍 **Geospatial Analysis**: Finding optimal locations for new stores, warehouses, or cell towers based on population density clusters.

---

## 7. Advantages & Limitations

### ✅ Advantages
- **Fast and Scalable**: Scales extremely well to massive datasets $O(n \times K \times I \times d)$.
- **Simple to Understand**: The intuition and geometry are highly interpretable.
- **Guaranteed Convergence**: The algorithm will always converge (though potentially to a local minimum).

### ❌ Limitations
- **Must Specify K**: You have to know or guess the number of clusters beforehand.
- **Sensitive to Outliers**: Since it uses the mean, a single extreme outlier can drastically pull a centroid out of place.
- **Assumes Spherical Clusters**: K-Means expects clusters to be convex (spherical) and similarly sized. It completely fails on elongated or interlocking shapes like half-moons. (See DBSCAN or GMM for solutions).
- **Scale Dependent**: Highly sensitive to feature scaling. Always standardize/normalize your data before applying K-Means.

---

## 8. Mini-Project: Customer Segmentation

**Problem Statement:** You are an analyst for a mall. You have synthetic data of your customers containing `Annual Income (k$)` and `Spending Score (1-100)`. You want to segment these customers into groups to understand their behavior for targeted marketing.

### Full Python Solution

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Synthesize Customer Data
np.random.seed(42)
# Creating 5 distinct customer profiles
n = 50
c1 = np.random.normal(loc=[25, 20], scale=[5, 5], size=(n, 2))  # Low Income, Low Spend
c2 = np.random.normal(loc=[25, 80], scale=[5, 5], size=(n, 2))  # Low Income, High Spend
c3 = np.random.normal(loc=[60, 50], scale=[8, 8], size=(n*2, 2))# Mid Income, Mid Spend (Core customers)
c4 = np.random.normal(loc=[90, 20], scale=[6, 6], size=(n, 2))  # High Income, Low Spend
c5 = np.random.normal(loc=[90, 85], scale=[6, 6], size=(n, 2))  # High Income, High Spend

data = np.vstack((c1, c2, c3, c4, c5))
df = pd.DataFrame(data, columns=['Annual_Income', 'Spending_Score'])

# 2. Preprocessing (Standardization is crucial for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Determine Optimal K using the Elbow Method
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, marker='o', linestyle='--', color='blue')
plt.title('The Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True, alpha=0.3)

# 4. Train Final Model (K=5 looks like a good elbow)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Inverse transform centroids back to original scale for interpretation
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# 5. Visualize the Segments
plt.subplot(1, 2, 2)
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual_Income'], cluster_data['Spending_Score'], 
                s=40, c=colors[i], label=f'Cluster {i}', alpha=0.7)

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.tight_layout()
plt.savefig('customer_segmentation.png', dpi=150)
# plt.show()

print("\n--- Marketing Strategy based on Segments ---")
print("Target High Income / High Spend clusters with luxury items.")
print("Target Mid Income / Mid Spend with volume discounts.")
print("Investigate High Income / Low Spend to see why they aren't purchasing more.")
```

### Output Interpretation
The algorithm successfully identifies 5 distinct customer groups based solely on numerical similarity. The clusters visually separate those who earn a lot but spend little, those who earn little but spend a lot, and the average core customer base. The marketing team can immediately apply distinct strategies to each centroid group.

---

[Back to Index](../README.md) | [Next: Hierarchical Clustering →](./02-Hierarchical-Clustering.md)
