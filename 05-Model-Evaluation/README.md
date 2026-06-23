# 🎯 05: Model Evaluation

Welcome to the ultimate Model Evaluation module! This section is designed to take you from a basic understanding of model training to rigorous, industry-grade ML evaluation, optimization, and production monitoring.

## 📋 Learning Roadmap

1. [Introduction to Model Evaluation](./01-Introduction-To-Model-Evaluation.md)
2. [Train, Test, and Validation Split](./02-Train-Test-Validation-Split.md)
3. [Bias-Variance Tradeoff](./03-Bias-Variance-Tradeoff.md)
4. [Regression Metrics](./04-Regression-Metrics.md)
5. [Classification Metrics](./05-Classification-Metrics.md)
6. [Confusion Matrix](./06-Confusion-Matrix.md)
7. [ROC and AUC](./07-ROC-AUC.md)
8. [Precision-Recall Curves](./08-Precision-Recall-Curves.md)
9. [Cross Validation](./09-Cross-Validation.md)
10. [Learning Curves](./10-Learning-Curves.md)
11. [Validation Curves](./11-Validation-Curves.md)
12. [Hyperparameter Tuning](./12-Hyperparameter-Tuning.md)
13. [Imbalanced Classification](./13-Imbalanced-Classification.md)
14. [Model Comparison and Statistical Testing](./14-Model-Comparison-And-Statistical-Testing.md)
15. [Production Monitoring](./15-Production-Monitoring.md)
16. [Interpretability and Explainability](./16-Interpretability-And-Explainability.md)

## 📊 Visualizations & Workshops

We provide interactive Jupyter notebooks to visualize and code every core concept from the roadmap. See the [`notebooks/`](./notebooks/) directory for hands-on tutorials covering:
- Confusion Matrix Heatmaps
- ROC and Precision-Recall Curves
- Learning and Validation Curves
- Cross-Validation Stability
- SHAP and Permutation Importance
- Data Drift Detection 

## 💻 Projects

Apply your knowledge through comprehensive, standalone projects located in the [`projects/`](./projects/) directory:

1. **Model Evaluation Dashboard**: An interactive threshold simulator using `ipywidgets`.
2. **Classification Metrics Explorer**: Interactive exploration of ROC vs PR on highly imbalanced data.
3. **Cross-Validation Laboratory**: Variance tracking across standard vs stratified K-Fold.
4. **Hyperparameter Tuning Engine**: Execution time and score benchmarking for Grid Search, Random Search, and Optuna.
5. **Imbalanced Data Challenge**: Minimizing False Negatives using Class Weights and SMOTE in an imblearn Pipeline.
6. **End-To-End Model Deployment**: Generating Data Drift reports using Evidently AI.

## 🛠️ Setup & Structure

```bash
# Clone the repository
git clone https://github.com/sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap.git
cd 05-Model-Evaluation

# Install required packages
pip install -r projects/requirements.txt # (If present inside individual modules)
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn evidently shap ipywidgets

# Launch Jupyter
jupyter notebook
```

---
[← Return to Root Index](../README.md)
