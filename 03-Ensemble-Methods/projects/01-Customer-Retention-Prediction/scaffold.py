import os
import nbformat as nbf

base_dir = r"c:\Users\ADMIN\Desktop\Full-ML-DL-CV-and-Data-Science-Roadmap\03-Ensemble-Methods\projects\01-Customer-Retention-Prediction"
dirs = ["data", "notebooks", "src", "models", "reports", "images"]

for d in dirs:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    with open(os.path.join(base_dir, d, ".gitkeep"), "w") as f:
        pass

# Create a starter notebook
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("# Customer Retention Prediction using Bagging\n\n## Business Problem\nIdentify customers at risk of churn and apply bagging classifiers to improve prediction stability."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.ensemble import BaggingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.datasets import make_classification\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report, accuracy_score"),
    nbf.v4.new_markdown_cell("## Dataset Generation (Mock Data)\nSince we don't have a real DB connection, we generate a synthetic mock dataset simulating e-commerce behavior."),
    nbf.v4.new_code_cell("X, y = make_classification(n_samples=5000, n_features=15, n_informative=8, \n                           n_redundant=4, weights=[0.7, 0.3], random_state=42)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"),
    nbf.v4.new_markdown_cell("## Model Training"),
    nbf.v4.new_code_cell("base_clf = DecisionTreeClassifier(max_depth=None)\nbagging_clf = BaggingClassifier(estimator=base_clf, n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)\n\nbagging_clf.fit(X_train, y_train)\nprint(f\"OOB Score: {bagging_clf.oob_score_:.4f}\")"),
    nbf.v4.new_markdown_cell("## Evaluation & Error Analysis"),
    nbf.v4.new_code_cell("y_pred = bagging_clf.predict(X_test)\nprint(classification_report(y_test, y_pred))\nprint(f\"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\")")
]
nb.cells.extend(cells)

out_nb = os.path.join(base_dir, "notebooks", "01_Model_Training.ipynb")
with open(out_nb, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Customer Retention project scaffolded.")
