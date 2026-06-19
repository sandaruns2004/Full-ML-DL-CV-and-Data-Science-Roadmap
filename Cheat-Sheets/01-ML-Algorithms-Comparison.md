# 📄 Cheat Sheet: ML Algorithms Comparison

> A quick-reference guide for deciding when to use which Machine Learning algorithm.

---

## 1. Regression Algorithms (Predicting Continuous Values)

| Algorithm | Best For | Pros | Cons | Key Assumptions |
|-----------|----------|------|------|-----------------|
| **Linear Regression** | Baselines, determining feature importance, inferring simple relationships. | Very fast, highly interpretable, won't overfit easily. | Cannot capture non-linear relationships. Sensitive to outliers. | Linear relationship, No Multicollinearity, Homoscedasticity. |
| **Ridge/Lasso Regression** | Datasets with many features where you suspect multicollinearity. | Prevents overfitting. Lasso performs automatic feature selection. | Requires hyperparameter tuning for alpha. | Same as Linear Regression. |
| **Decision Tree Regressor** | Datasets with non-linear relationships and interactions between features. | No feature scaling needed. Highly interpretable (can draw the tree). | Prone to massive overfitting if tree depth is not restricted. | None. |
| **Random Forest Regressor** | Almost all tabular data. The ultimate robust baseline model. | Extremely robust to outliers and noise. Captures complex non-linearities. | Slow to predict. Uses a lot of memory. Black box (hard to interpret). | None. |
| **XGBoost / LightGBM** | Winning Kaggle competitions. High-performance production tabular data. | Highest accuracy for tabular data. Handles missing values natively. | Requires careful hyperparameter tuning. Can overfit small datasets. | None. |
| **Neural Networks (MLP)** | Massive datasets, unstructured data (images/text), complex functional approximations. | Infinite flexibility. | Requires huge amounts of data. Needs extensive tuning and feature scaling. | Data must be scaled. |

---

## 2. Classification Algorithms (Predicting Categories)

| Algorithm | Best For | Pros | Cons | Key Assumptions |
|-----------|----------|------|------|-----------------|
| **Logistic Regression** | Binary classification baselines, providing probabilities. | Fast, interpretable, gives well-calibrated probabilities. | Assumes linear decision boundary. | Linear separability in feature space. |
| **K-Nearest Neighbors (KNN)** | Very small datasets with complex decision boundaries. Recommendation engines. | No training phase (lazy learning). Can model highly irregular boundaries. | Very slow at inference time (must calculate distance to all training points). | Features must be strictly scaled. |
| **Naive Bayes** | Text classification (spam detection, sentiment), high-dimensional sparse data. | Extremely fast. Works incredibly well on small text datasets. | "Naive" assumption often harms accuracy on correlated features. | Features are perfectly independent. |
| **Support Vector Machines (SVM)** | High-dimensional data with a clear margin of separation. | Effective in high dimensions. Kernel trick allows non-linear modeling. | Does not scale well to large datasets ($O(n^3)$ training time). | None. |
| **Random Forest Classifier** | General tabular data classification. Imbalanced classes (using class weights). | Handles categorical data well. Robust to outliers. Outputs feature importance. | Hard to interpret compared to a single decision tree. | None. |
| **XGBoost / LightGBM** | High-performance production tabular data classification. | State-of-the-art accuracy. Faster training than Random Forest (LightGBM). | Prone to overfitting on noisy data if not regularized properly. | None. |

---

## 3. Unsupervised Algorithms (Finding Patterns)

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **K-Means Clustering** | Segmenting customers into distinct groups. | Very fast, scales well to large datasets. | Must specify $K$ manually. Assumes clusters are spherical. Sensitive to outliers. |
| **DBSCAN** | Finding clusters of weird, non-spherical shapes, and identifying outliers. | Does not require specifying $K$. Native outlier detection. | Struggles with varying cluster densities. High dimensional distance metric issues. |
| **Gaussian Mixture Models** | Soft clustering (probabilities of belonging to a cluster). Elliptical clusters. | Very flexible cluster shapes (covariance matrices). Provides uncertainty measures. | Slower than K-Means. Can fail to converge. |
| **PCA (Principal Component Analysis)** | Dimensionality reduction, visualization, noise filtering. | Fast, deterministic. Orthogonal components. | Only captures linear correlations. Reduced features lose human interpretability. |
| **t-SNE / UMAP** | Visualizing high-dimensional data in 2D or 3D. | Unparalleled local structure preservation (looks great). | t-SNE is computationally heavy. Cannot be reliably used for clustering downstream. |

---

## 4. The Golden Rules of Model Selection

1. **Always start simple:** Always build a Logistic/Linear regression or a single Decision Tree first. This gives you a baseline to prove if complex models are actually adding value.
2. **If it's Tabular Data (CSV, SQL Database):** XGBoost, LightGBM, or CatBoost will almost certainly be the most accurate models.
3. **If it's Unstructured Data (Images, Audio, Text):** Deep Learning is the only viable choice.
4. **If Interpretability is required (Medical, Finance, Law):** Stick to Logistic Regression, Decision Trees, or explicitly use SHAP/LIME on Random Forests.
5. **If Data is small ($n < 1000$):** High bias / low variance models like Naive Bayes or heavily regularized Linear Models prevent overfitting. Avoid Neural Networks.

---

[← Future Versions](../16-Projects/07-Future-Versions.md) | [Back to Index](../README.md) | [Next: Deep Learning Reference →](./02-Deep-Learning-Reference.md)
