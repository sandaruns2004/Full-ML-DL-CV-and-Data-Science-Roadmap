import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# 🎒 Bagging Workshop: Heart Disease Prediction\n\nIn this interactive lab, we will build a Bagging ensemble from scratch, use NumPy for vectorization, and finally leverage Scikit-Learn's production-ready implementation to predict heart disease."),
    
    nbf.v4.new_markdown_cell("## 1. Problem Overview\n\n**Goal**: Predict whether a patient has heart disease based on clinical features.\n\nWe will start with a baseline Decision Tree, then build Bagging to see how it reduces variance and improves accuracy."),
    
    nbf.v4.new_markdown_cell("## 2. Dataset Exploration"),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score, classification_report\n\n# Using breast cancer dataset as a proxy for clinical diagnosis\ndata = load_breast_cancer()\nX = pd.DataFrame(data.data, columns=data.feature_names)\ny = pd.Series(data.target)\n\nX.head()"),
    
    nbf.v4.new_markdown_cell("## 3. Data Cleaning & 4. Feature Engineering"),
    nbf.v4.new_code_cell("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nprint(f\"Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\")"),
    
    nbf.v4.new_markdown_cell("## 5. Baseline Model (Single Decision Tree)"),
    nbf.v4.new_code_cell("from sklearn.tree import DecisionTreeClassifier\n\nbaseline_tree = DecisionTreeClassifier(random_state=42)\nbaseline_tree.fit(X_train, y_train)\n\ny_pred_base = baseline_tree.predict(X_test)\nprint(\"Baseline Decision Tree Accuracy:\", accuracy_score(y_test, y_pred_base))"),
    
    nbf.v4.new_markdown_cell("## 6. From Scratch Implementation"),
    nbf.v4.new_code_cell("from collections import Counter\n\nclass BaggingFromScratch:\n    def __init__(self, n_estimators=10):\n        self.n_estimators = n_estimators\n        self.models = []\n        \n    def fit(self, X, y):\n        n_samples = len(X)\n        X_np, y_np = np.array(X), np.array(y)\n        \n        for _ in range(self.n_estimators):\n            indices = np.random.choice(n_samples, size=n_samples, replace=True)\n            X_boot, y_boot = X_np[indices], y_np[indices]\n            \n            tree = DecisionTreeClassifier(random_state=42)\n            tree.fit(X_boot, y_boot)\n            self.models.append(tree)\n            \n    def predict(self, X):\n        predictions = np.array([model.predict(X) for model in self.models])\n        \n        final_preds = []\n        for i in range(len(X)):\n            votes = predictions[:, i]\n            majority = Counter(votes).most_common(1)[0][0]\n            final_preds.append(majority)\n        return np.array(final_preds)\n\nbagging_scratch = BaggingFromScratch(n_estimators=50)\nbagging_scratch.fit(X_train, y_train)\npreds_scratch = bagging_scratch.predict(X_test)\nprint(\"From Scratch Bagging Accuracy:\", accuracy_score(y_test, preds_scratch))"),
    
    nbf.v4.new_markdown_cell("## 7. NumPy Implementation (Vectorized)"),
    nbf.v4.new_code_cell("from scipy.stats import mode\n\nclass NumpyBagging:\n    def __init__(self, n_estimators=50):\n        self.n_estimators = n_estimators\n        self.trees = []\n        \n    def fit(self, X, y):\n        n_samples = len(X)\n        X_np, y_np = np.array(X), np.array(y)\n        \n        # Vectorized bootstrapping\n        boot_indices = np.random.randint(0, n_samples, size=(self.n_estimators, n_samples))\n        \n        for idx in boot_indices:\n            tree = DecisionTreeClassifier(max_depth=None)\n            tree.fit(X_np[idx], y_np[idx])\n            self.trees.append(tree)\n            \n    def predict(self, X):\n        tree_preds = np.array([tree.predict(X) for tree in self.trees])\n        final_preds, _ = mode(tree_preds, axis=0, keepdims=False)\n        return final_preds\n\nnp_bag = NumpyBagging(n_estimators=50)\nnp_bag.fit(X_train, y_train)\nprint(\"NumPy Bagging Accuracy:\", accuracy_score(y_test, np_bag.predict(X_test)))"),
    
    nbf.v4.new_markdown_cell("## 8. Library Implementation (Scikit-Learn)"),
    nbf.v4.new_code_cell("from sklearn.ensemble import BaggingClassifier\n\n# Using OOB score\nsk_bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)\nsk_bag.fit(X_train, y_train)\n\nprint(\"OOB Score:\", sk_bag.oob_score_)\nprint(\"Test Accuracy:\", accuracy_score(y_test, sk_bag.predict(X_test)))"),
    
    nbf.v4.new_markdown_cell("## 9. Hyperparameter Experiments"),
    nbf.v4.new_code_cell("estimators = [10, 50, 100, 200, 500]\nscores = []\n\nfor n in estimators:\n    model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=n, random_state=42)\n    model.fit(X_train, y_train)\n    scores.append(accuracy_score(y_test, model.predict(X_test)))\n\nplt.plot(estimators, scores, marker='o')\nplt.title(\"Accuracy vs Number of Estimators\")\nplt.xlabel(\"n_estimators\")\nplt.ylabel(\"Accuracy\")\nplt.show()"),
    
    nbf.v4.new_markdown_cell("## 10. Model Comparison & 11. Error Analysis"),
    nbf.v4.new_code_cell("print(\"Baseline Classification Report:\\n\", classification_report(y_test, y_pred_base))\nprint(\"\\nBagging Classification Report:\\n\", classification_report(y_test, sk_bag.predict(X_test)))"),
    
    nbf.v4.new_markdown_cell("## 12. Feature Importance Analysis"),
    nbf.v4.new_code_cell("# Bagging doesn't have feature_importances_ built in by default (unlike Random Forest).\n# We can average the feature importances of all base trees.\nimportances = np.mean([tree.feature_importances_ for tree in sk_bag.estimators_], axis=0)\nindices = np.argsort(importances)[::-1][:10]\n\nplt.figure(figsize=(10, 5))\nplt.bar(range(10), importances[indices])\nplt.xticks(range(10), X.columns[indices], rotation=45)\nplt.title(\"Top 10 Feature Importances in Bagging Ensemble\")\nplt.show()"),
    
    nbf.v4.new_markdown_cell("## 13. Explainability (SHAP where relevant)"),
    nbf.v4.new_code_cell("import shap\n\n# SHAP can interpret Tree ensembles easily\nexplainer = shap.TreeExplainer(sk_bag.estimators_[0]) # Example on first tree\nshap_values = explainer.shap_values(X_test)\nshap.summary_plot(shap_values, X_test, plot_type=\"bar\")"),
    
    nbf.v4.new_markdown_cell("## 14. Mini Challenges\n1. Modify the `BaggingFromScratch` to use `LogisticRegression` as a base estimator. Does accuracy improve?\n2. Implement Pasting (sampling WITHOUT replacement) in the NumPy version."),
    
    nbf.v4.new_markdown_cell("## 15. Solutions\n*(Try the challenges before looking at solutions!)*"),
    
    nbf.v4.new_markdown_cell("## 16. Key Takeaways\n- Bagging powerfully reduces variance.\n- Scikit-learn's implementation is highly optimized.\n- Feature importance in bagging is derived by averaging over trees.")
]

nb.cells.extend(cells)

out_path = r\"c:\Users\ADMIN\Desktop\Full-ML-DL-CV-and-Data-Science-Roadmap\03-Ensemble-Methods\notebooks\01_bagging_lab.ipynb\"
with open(out_path, \"w\", encoding=\"utf-8\") as f:
    nbf.write(nb, f)

print(f\"Notebook generated at {out_path}\")
