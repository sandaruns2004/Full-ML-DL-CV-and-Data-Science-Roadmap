# 🔍 Introduction To Unsupervised Learning

> **Prerequisites**: Basic ML Concepts | **Difficulty**: ⭐☆☆☆☆ Beginner

---

## 📋 Table of Contents
1. [What is Unsupervised Learning?](#1-what-is-unsupervised-learning)
2. [Types of Unsupervised Learning](#2-types-of-unsupervised-learning)

---

## 1. What is Unsupervised Learning?

### 🟢 Beginner
**Simple Explanation**: Unlike Supervised Learning where you have a teacher giving you the answers (labels), Unsupervised Learning is like exploring a new city without a map. The algorithm looks at the raw data and tries to find hidden patterns, groupings, or rules all on its own.
**Real-World Analogy**: Imagine a librarian given a pile of thousands of unlabelled books and asked to organize them. They might group them by cover color, thickness, or subject, without knowing the official categories beforehand.

### 🟡 Intermediate
**Workflow**: Feed raw data $X$ (without $y$ labels) into the algorithm. Extract structural properties like clusters, principal components, or density representations. 
**Applications**: Customer segmentation, anomaly detection (fraud), genetic clustering, recommendation systems.

### 🔴 Advanced
**Mathematics**: Let the dataset be $X = \{x_1, x_2, \dots, x_N\}$. Unsupervised learning seeks to learn the underlying probability distribution $p(x)$ of the data, or to find a mapping $f(x)$ that transforms the data into a more compact or interpretable representation (manifold learning).

---

[← Ensemble Selection Strategies](../03-Ensemble-Methods/14-Model-Selection-For-Ensembles.md) | [Back to Index](../README.md) | [Next: K-Means Clustering →](02-K-Means-Clustering.md)
