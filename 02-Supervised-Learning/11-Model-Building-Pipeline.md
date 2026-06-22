# ⚙️ Model Building Pipeline

> **Prerequisites**: Feature Engineering | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 1. Scikit-Learn Pipelines

### 🟢 Beginner
**Simple Explanation**: A pipeline is an assembly line. Raw data goes in, gets cleaned, scaled, transformed, and then predicted by a model—all in one seamless motion.

### 🟡 Intermediate
**Workflow**: `sklearn.pipeline.Pipeline` chains multiple transformers and a final estimator. It prevents data leakage during cross-validation.

### 🔴 Advanced
**Industry Examples**:
In production, you never deploy just a model. You deploy the entire pipeline as a serialized artifact (`.pkl` or `.joblib` or ONNX format). This ensures that production data undergoes the exact same transformations (using the same learned scaling means/variances) as the training data.

---

[← Regularization (Ridge, Lasso, ElasticNet)](10-Regularization.md) | [Back to Index](../README.md) | [Next: Model Selection Guide →](12-Model-Selection-Guide.md)
