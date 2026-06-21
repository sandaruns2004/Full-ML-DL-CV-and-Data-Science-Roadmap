# 🌳 Decision Trees

> **Prerequisites**: None | **Difficulty**: ⭐☆☆☆☆ Beginner

---

## 1. Information Gain & Gini

### 🟢 Beginner
**Simple Explanation**: A flowchart that asks a series of yes/no questions to arrive at a prediction. "Is the person older than 30?" -> "Do they eat healthy?" -> "Likely to have low cholesterol."

### 🟡 Intermediate
**Working Mechanism**: The tree recursively splits the data into subsets that are increasingly homogeneous (pure).

### 🔴 Advanced
**Mathematics**:
Entropy: $H(S) = -\sum_{i=1}^c p_i \log_2(p_i)$
Gini Impurity: $G(S) = 1 - \sum_{i=1}^c p_i^2$
The split that maximizes Information Gain (reduction in Entropy) is chosen. Pruning is required to prevent overfitting.
