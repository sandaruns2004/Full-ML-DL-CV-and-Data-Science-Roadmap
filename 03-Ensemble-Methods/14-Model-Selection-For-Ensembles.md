# ⚖️ Ensemble Selection Strategies

> **Prerequisites**: All Ensembles | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 1. Diversity vs Performance

### 🟢 Beginner
**Simple Explanation**: If you combine 5 identical Random Forests, you gain nothing. You need models that make *different* mistakes. Combine a Tree with a Neural Net and an SVM!

### 🟡 Intermediate
**Tradeoffs**: 
- **High Diversity**: Stacking/Voting with different algorithms.
- **High Performance, Single Alg**: XGBoost/LightGBM.
- **Interpretability**: A single Decision Tree is better than any ensemble if you need to explain the model to a human.

### 🔴 Advanced
**Industry Examples**: 
In production, ensembles add latency and memory overhead. A massive stacking architecture might win a Kaggle competition but is totally useless for a high-frequency trading bot requiring sub-millisecond inference. Model distillation (training a single model to mimic the ensemble) is often used to bridge this gap.
