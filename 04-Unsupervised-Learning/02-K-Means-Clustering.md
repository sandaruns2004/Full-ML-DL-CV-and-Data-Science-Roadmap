# 🎯 K-Means Clustering

> **Prerequisites**: Introduction to Unsupervised | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 1. Centroids and Distance Metrics

### 🟢 Beginner
**Simple Explanation**: K-Means is like organizing a party into distinct friend groups. You pick $K$ random people to be the "center" of each group (centroids). Everyone else joins the group of the center person they are closest to. Then, the groups find their *actual* average center, and the process repeats until the groups stop shifting.

### 🟡 Intermediate
**Workflow**:
1. Initialize $K$ centroids randomly.
2. Assign each data point to the nearest centroid.
3. Update the centroid to be the mean of the assigned points.
4. Repeat 2 and 3 until convergence.

**Finding K**: The **Elbow Method** plots the Within-Cluster-Sum-of-Squares (WCSS) against $K$. The "elbow" point is usually the optimal $K$. The **Silhouette Score** measures how similar an object is to its own cluster compared to other clusters.

### 🔴 Advanced
**Mathematics & Optimization**:
Objective Function (Inertia): $\min_{\mu} \sum_{i=1}^K \sum_{x \in C_i} ||x - \mu_i||^2$
**Complexity**: $O(n \cdot K \cdot I \cdot d)$ where $n$ is data points, $I$ is iterations, $d$ is dimensions.
**Limitations**: Assumes spherical clusters of similar size. Fails on elongated or crescent-shaped data.

---

[← Introduction To Unsupervised Learning](01-Introduction-To-Unsupervised-Learning.md) | [Back to Index](../README.md) | [Next: Hierarchical Clustering →](03-Hierarchical-Clustering.md)
