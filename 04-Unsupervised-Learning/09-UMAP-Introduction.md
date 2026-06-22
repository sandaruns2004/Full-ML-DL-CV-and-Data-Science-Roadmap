# 🌌 UMAP (Uniform Manifold Approximation and Projection)

> **Prerequisites**: t-SNE | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 1. Modern Manifold Learning

### 🟢 Beginner
**Simple Explanation**: UMAP is the newer, faster, and arguably better sibling of t-SNE. While t-SNE only cares about keeping close neighbors close, UMAP also tries to preserve the *global* structure (the big picture) of the data, and it runs in a fraction of the time.

### 🟡 Intermediate
**Comparison with t-SNE**:
- **Speed**: UMAP is significantly faster.
- **Global Structure**: UMAP preserves distances between clusters better than t-SNE.
- **New Data**: UMAP can transform new, unseen data into the existing projection space (t-SNE cannot do this natively).

### 🔴 Advanced
**Mathematics**: 
UMAP builds on Riemannian geometry and algebraic topology. It constructs a fuzzy simplicial complex representation of the data and optimizes the low-dimensional embedding to have as similar a fuzzy topological structure as possible using cross-entropy loss instead of KL-divergence.

---

[← t-SNE (t-Distributed Stochastic Neighbor Embedding)](08-tSNE.md) | [Back to Index](../README.md) | [Next: Association Rule Mining →](10-Association-Rule-Mining.md)
