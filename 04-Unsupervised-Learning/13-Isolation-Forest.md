# 🌲 Isolation Forest

> **Prerequisites**: Random Forest, Anomaly Detection | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 1. Tree-Based Isolation

### 🟢 Beginner
**Simple Explanation**: Normal points are clustered tightly together. Anomalies are far away. If you randomly draw lines to separate data points, it will take many lines to separate a normal point packed in a crowd. But a lonely anomaly will be isolated very quickly with just one or two lines!

### 🟡 Intermediate
**Working Mechanism**: 
Builds an ensemble of Random Trees. In each tree, it recursively generates random splits. The "anomaly score" is inversely proportional to the path length from the root of the tree to the terminating leaf. Shorter path length = high probability of being an anomaly.

### 🔴 Advanced
**Complexity & Scalability**: 
Unlike distance-based or density-based methods which are $O(n^2)$, Isolation Forest operates in $O(n \log n)$ time and requires very little memory. It scales beautifully to massive, high-dimensional datasets.
