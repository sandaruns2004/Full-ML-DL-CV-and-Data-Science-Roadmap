# 🛒 Association Rule Mining

> **Prerequisites**: Basic Probability | **Difficulty**: ⭐☆☆☆☆ Beginner

---

## 1. Market Basket Analysis

### 🟢 Beginner
**Simple Explanation**: Also known as "If you bought this, you might also buy that". It's the algorithm behind Amazon recommendations and supermarket layouts. If people frequently buy Diapers and Beer together, the algorithm flags this as an association rule.

### 🟡 Intermediate
**Metrics Overview**:
- **Support**: How popular is an itemset? (e.g., Apple is in 20% of all transactions).
- **Confidence**: How likely is item $Y$ purchased when item $X$ is purchased? 
- **Lift**: How much more likely is $Y$ bought with $X$ compared to $Y$ being bought independently? Lift > 1 means a strong positive association.

### 🔴 Advanced
**Industry Considerations**: 
While simple, Association Rules explode computationally. If a supermarket has 10,000 unique items, there are $2^{10000}$ possible combinations. We need efficient algorithms (like Apriori or FP-Growth) to prune the search space.

---

[← UMAP (Uniform Manifold Approximation and Projection)](09-UMAP-Introduction.md) | [Back to Index](../README.md) | [Next: Apriori Algorithm →](11-Apriori-Algorithm.md)
