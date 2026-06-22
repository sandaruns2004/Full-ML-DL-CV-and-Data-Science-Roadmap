# 🏘️ K-Nearest Neighbors (KNN)

> **Prerequisites**: Intro to DS | **Difficulty**: ⭐☆☆☆☆ Beginner

---

## 1. Distance Metrics & K Selection

### 🟢 Beginner
**Simple Explanation**: "Tell me who your friends are, and I will tell you who you are." If a new data point is surrounded by mostly "Cats", it's probably a "Cat".

### 🟡 Intermediate
**Working Mechanism**: KNN stores all training data. When a new point arrives, it calculates the distance to all points, finds the K nearest ones, and takes a majority vote.

### 🔴 Advanced
**Mathematics**:
Euclidean Distance: $d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$
Curse of Dimensionality: In high dimensions, all points become almost equidistant, rendering KNN useless unless dimensionality reduction is applied first.

---

[← Logistic Regression](04-Logistic-Regression.md) | [Back to Index](../README.md) | [Next: Decision Trees →](06-Decision-Trees.md)
