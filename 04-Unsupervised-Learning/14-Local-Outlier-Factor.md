# 🔭 Local Outlier Factor (LOF)

> **Prerequisites**: KNN, Density | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 1. Local Density Analysis

### 🟢 Beginner
**Simple Explanation**: Sometimes an anomaly isn't just a point far away from everything. Sometimes it's a point that is just *slightly* further away from its local neighborhood than the neighbors are from each other. LOF compares the density of a point to the density of its friends.

### 🟡 Intermediate
**Working Mechanism**: 
LOF calculates the local density of a point based on its $k$ nearest neighbors. 
- $LOF \approx 1$: Point has similar density to neighbors (Normal).
- $LOF > 1$: Point has lower density than neighbors (Anomaly).

### 🔴 Advanced
**Mathematics**: 
Relies on reachability distance:
$reach\_dist_k(A, B) = \max(k\_distance(B), d(A, B))$
The Local Reachability Density (LRD) is the inverse of the average reachability distance of $A$ from its neighbors. The final LOF is the ratio of the average LRD of the neighbors to the LRD of point $A$.

---

[← Isolation Forest](13-Isolation-Forest.md) | [Back to Index](../README.md) | [Next: Introduction to Model Evaluation →](../05-Model-Evaluation/01-Introduction-To-Model-Evaluation.md)
