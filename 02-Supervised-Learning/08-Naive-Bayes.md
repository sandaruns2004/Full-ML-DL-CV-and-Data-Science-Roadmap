# 🎲 Naive Bayes

> **Prerequisites**: Probability Theory | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 1. Gaussian, Multinomial, Bernoulli

### 🟢 Beginner
**Simple Explanation**: Uses probability to predict categories. "Naive" because it assumes all features are independent, like assuming rain and wearing a blue shirt have no relationship.

### 🟡 Intermediate
**Working Mechanism**: 
- Gaussian: Continuous data (assumes normal distribution)
- Multinomial: Word counts (Text Classification)
- Bernoulli: Binary features

### 🔴 Advanced
**Mathematics**:
Bayes Theorem: $P(y|X) = \frac{P(X|y)P(y)}{P(X)}$
Assumption of conditional independence: $P(x_1, \dots, x_n|y) = \prod_{i=1}^n P(x_i|y)$

---

[← Support Vector Machines (SVM)](07-Support-Vector-Machines.md) | [Back to Index](../README.md) | [Next: Feature Engineering for Supervised Learning →](09-Feature-Engineering-For-Supervised-Learning.md)
