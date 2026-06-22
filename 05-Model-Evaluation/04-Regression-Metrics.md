# 📈 Regression Metrics

> **Prerequisites**: Model Evaluation Concepts | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 📋 Table of Contents
1. [Absolute vs Squared Errors](#1-absolute-vs-squared-errors)
2. [R-Squared Metrics](#2-r-squared-metrics)
3. [When to use which metric?](#3-when-to-use-which-metric)

---

## 1. Absolute vs Squared Errors

### 🟢 Beginner
**Simple Explanation**: 
When predicting a continuous number (like House Prices), we need to know how far off our guesses are. 
- **MAE (Mean Absolute Error)**: "On average, my predictions are off by $5,000."
- **RMSE (Root Mean Squared Error)**: Similar to MAE, but it heavily penalizes *huge* mistakes. If you are off by $100,000 on one house, RMSE will skyrocket, while MAE will be less affected.

### 🟡 Intermediate
**Practical Implementation**:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = [100, 150, 200, 250]
y_pred = [110, 140, 220, 210]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae} | RMSE: {rmse}")
```

**Which to choose?**:
Use **MAE** if you have lots of outliers in your dataset that you don't care about. Use **RMSE** if large errors are absolutely unacceptable (e.g., predicting drug dosages).

### 🔴 Advanced
**Mathematics and Statistical Properties**:

**MAE**: $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
Minimizing MAE leads to predicting the **median** of the target distribution.

**MSE**: $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
Minimizing MSE leads to predicting the **mean** of the target distribution. Because it is differentiable everywhere, it is the standard loss function for gradient descent algorithms.

**RMSLE (Root Mean Squared Logarithmic Error)**: 
$\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(y_i + 1) - \log(\hat{y}_i + 1))^2}$
RMSLE cares about *percentage* errors rather than absolute errors. Predicting 10 for a true value of 20 is penalized the same as predicting 100 for a true value of 200.

**Adjusted R²**: 
$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
Standard $R^2$ will artificially inflate if you just keep adding useless features to the model. Adjusted $R^2$ penalizes the addition of features that do not improve the model:
$R^2_{adj} = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - k - 1} \right]$
Where $n$ is sample size and $k$ is number of predictors.

---

[← The Bias-Variance Tradeoff](03-Bias-Variance-Tradeoff.md) | [Back to Index](../README.md) | [Next: Classification Metrics →](05-Classification-Metrics.md)
