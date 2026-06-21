# 🦠 DBSCAN (Density-Based Spatial Clustering)

> **Prerequisites**: Distance Metrics | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 1. Density and Noise

### 🟢 Beginner
**Simple Explanation**: DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. Imagine looking at a map of a city at night. The brightly lit, dense areas are the cities (clusters), and the dark areas are the wilderness (noise). DBSCAN groups together points that are packed closely and ignores the lonely outliers.

### 🟡 Intermediate
**Working Mechanism**: 
Requires two parameters:
1. `eps` (Epsilon): The maximum distance between two points to be considered neighbors.
2. `min_samples` (MinPts): The minimum number of points required in an `eps` radius to form a dense region.

**Advantages**: Discovers clusters of arbitrary shape (unlike K-Means) and automatically identifies outliers/noise.

### 🔴 Advanced
**Complexity & Limitations**: 
Complexity is $O(n \log n)$ with a spatial index (like a KD-Tree), but $O(n^2)$ without.
**Limitations**: Struggles heavily with datasets that have *varying densities* because `eps` is global. Extensions like OPTICS or HDBSCAN solve this by using hierarchical density estimates.
