import json
import os

def create_notebook(title, filename, algorithm_specific_content):
    cells = []
    
    def add_markdown(content):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in content.split("\n")]
        })
        
    def add_code(content):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in content.split("\n")]
        })

    # 1. Problem Overview
    add_markdown(f"# {title}\n\n## 1. Problem Overview\nIn this notebook, we will explore {title} through hands-on implementation. We'll start from scratch, then move to NumPy vectorization, and finally use Scikit-Learn for production-grade modeling.")
    
    # 2. Dataset Exploration
    add_markdown("## 2. Dataset Exploration\nLet's load a standard dataset and explore its properties, distributions, and correlations.")
    add_code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load dataset here (e.g., from sklearn.datasets)\n# sns.pairplot(df)")
    
    # 3. Data Preprocessing
    add_markdown("## 3. Data Preprocessing\nHandling missing values, encoding categoricals, and feature scaling.")
    add_code("from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\n\n# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n# scaler = StandardScaler()\n# X_train_scaled = scaler.fit_transform(X_train)")
    
    # 4. From Scratch Implementation
    add_markdown("## 4. From Scratch Implementation\nBuilding the algorithm using pure Python to understand the underlying mechanics.")
    add_code(algorithm_specific_content.get("scratch", "# Pure Python implementation goes here"))
    
    # 5. NumPy Implementation
    add_markdown("## 5. NumPy Implementation\nVectorizing the math for massive speed improvements.")
    add_code(algorithm_specific_content.get("numpy", "# NumPy implementation goes here"))
    
    # 6. Scikit-Learn Workflow
    add_markdown("## 6. Scikit-Learn Workflow\nThe industry standard way to build the model.")
    add_code(algorithm_specific_content.get("sklearn", "from sklearn.base import BaseEstimator\n# model.fit(X_train, y_train)"))
    
    # 7. Visualization Lab
    add_markdown("## 7. Visualization Lab\nVisualizing decision boundaries, learning curves, and feature importance.")
    add_code(algorithm_specific_content.get("viz", "# Plotting code goes here\nplt.show()"))
    
    # 8. Hyperparameter Experiments
    add_markdown("## 8. Hyperparameter Experiments\nTesting different parameters to see their effect on bias and variance.")
    add_code("# Code testing different hyperparameters (e.g. max_depth, C, k)")
    
    # 9. Failure Cases
    add_markdown("## 9. Failure Cases\nWhen does this algorithm fail? Let's intentionally break it.")
    add_code("# Code generating adverse datasets (outliers, non-linear data)")
    
    # 10. Real Dataset Case Study
    add_markdown("## 10. Real Dataset Case Study\nApplying everything we've learned to a messy, real-world dataset.")
    add_code("# End-to-end pipeline on real data")
    
    # 11. Mini Challenges
    add_markdown("## 11. Mini Challenges\nTry completing these tasks on your own:\n1. Challenge 1\n2. Challenge 2")
    add_code("# Your code here")
    
    # 12. Solutions
    add_markdown("## 12. Solutions\nDouble click to reveal solutions.")
    add_code("# Solution code here")
    
    # 13. Key Takeaways
    add_markdown("## 13. Key Takeaways\n- Takeaway 1\n- Takeaway 2")
    
    # 14. Further Practice
    add_markdown("## 14. Further Practice\n- Kaggle dataset links\n- Project ideas")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

notebooks_to_create = [
    ("Linear Regression", "01_Linear_Regression.ipynb", {}),
    ("Polynomial Regression", "02_Polynomial_Regression.ipynb", {}),
    ("Logistic Regression", "03_Logistic_Regression.ipynb", {}),
    ("Decision Trees", "04_Decision_Trees.ipynb", {}),
    ("K-Nearest Neighbors", "05_KNN.ipynb", {}),
    ("Support Vector Machines", "06_Support_Vector_Machines.ipynb", {}),
    ("Naive Bayes", "07_Naive_Bayes.ipynb", {}),
    ("Feature Engineering", "08_Feature_Engineering.ipynb", {}),
    ("Regularization", "09_Regularization.ipynb", {})
]

import os
base_dir = r"c:\Users\ADMIN\Desktop\Full-ML-DL-CV-and-Data-Science-Roadmap\02-Supervised-Learning\notebooks"
os.makedirs(base_dir, exist_ok=True)

for title, filename, content in notebooks_to_create:
    filepath = os.path.join(base_dir, filename)
    create_notebook(title, filepath, content)
    print(f"Created {filepath}")
