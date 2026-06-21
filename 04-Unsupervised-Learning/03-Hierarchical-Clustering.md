# 🌳 Hierarchical Clustering

> **Prerequisites**: K-Means | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 1. Agglomerative vs Divisive

### 🟢 Beginner
**Simple Explanation**: Imagine building a family tree from the bottom up. At first, everyone is their own cluster. Then, the two most similar people are grouped into a pair. Then, the most similar pairs are grouped, and so on, until everyone is in one giant family. This is called **Agglomerative** (bottom-up) clustering.

### 🟡 Intermediate
**Implementation & Workflow**:
- Does not require pre-specifying $K$!
- **Dendrograms**: A tree-like diagram that records the sequences of merges or splits. You can "cut" the tree at any height to get the desired number of clusters.
- **Linkage Methods**: 
  - *Single Linkage*: Distance between closest points.
  - *Complete Linkage*: Distance between furthest points.
  - *Ward's Method*: Minimizes the variance within clusters.

### 🔴 Advanced
**Complexity Analysis**: 
Time complexity is extremely high: $O(n^3)$ for standard agglomerative algorithms, making it unsuitable for massive datasets (millions of rows) without optimizations like BIRCH or approximate nearest neighbors.
