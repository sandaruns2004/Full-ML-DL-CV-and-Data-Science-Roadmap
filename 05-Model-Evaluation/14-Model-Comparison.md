# ⚔️ Model Comparison and Statistical Significance

> **Prerequisites**: Cross Validation | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [Which model is truly better?](#1-which-model-is-truly-better)
2. [Statistical Significance](#2-statistical-significance)

---

## 1. Which model is truly better?

### 🟢 Beginner
**Simple Explanation**: 
You build a Decision Tree that gets 85% accuracy. You build a Random Forest that gets 86% accuracy. Is the Random Forest actually smarter? Or did it just get lucky on the test set? To know for sure, we need to run statistical tests.

### 🟡 Intermediate
**Workflow**: 
When comparing two models, we don't just compare a single test score. We use K-Fold cross validation to get an *array* of scores for both models. Then, we use a paired Student's t-test to see if the difference between the arrays is statistically significant.

```python
from scipy.stats import ttest_rel
import numpy as np

# Cross-validation scores from 10 folds
scores_model_A = np.array([0.85, 0.86, 0.84, 0.85, 0.87, 0.84, 0.85, 0.86, 0.85, 0.84])
scores_model_B = np.array([0.86, 0.87, 0.86, 0.87, 0.88, 0.85, 0.86, 0.88, 0.87, 0.85])

# Perform paired t-test
t_stat, p_value = ttest_rel(scores_model_A, scores_model_B)

print(f"P-Value: {p_value:.4f}")
if p_value < 0.05:
    print("Model B is statistically significantly better than Model A.")
```

### 🔴 Advanced
**The McNemar's Test**:
For classification problems, especially deep neural networks where 10-fold cross validation is too computationally expensive, you can use McNemar's Test on the single test-set predictions. It focuses specifically on the cases where the models disagree. If Model A got it right and Model B got it wrong 50 times, but Model B got it right and Model A got it wrong only 5 times, McNemar's test will mathematically prove that the difference is not due to random chance.

---

[← Imbalanced Classification Evaluation](13-Imbalanced-Classification.md) | [Back to Index](../README.md) | [Next: Production Monitoring and Drift →](15-Production-Monitoring.md)
