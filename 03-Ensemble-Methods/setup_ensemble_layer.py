import os
import json

base_dir = r"c:\Users\ADMIN\Desktop\Full-ML-DL-CV-and-Data-Science-Roadmap\03-Ensemble-Methods"

# Scaffold Notebooks
notebooks = [
    "01_Random_Forest_Lab.ipynb",
    "02_Extra_Trees_Lab.ipynb",
    "03_AdaBoost_Lab.ipynb",
    "04_Gradient_Boosting_Lab.ipynb",
    "05_XGBoost_Lab.ipynb",
    "06_LightGBM_Lab.ipynb",
    "07_CatBoost_Lab.ipynb"
]

notebooks_dir = os.path.join(base_dir, "notebooks")
os.makedirs(notebooks_dir, exist_ok=True)

empty_nb = {
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

for nb in notebooks:
    with open(os.path.join(notebooks_dir, nb), "w", encoding="utf-8") as f:
        json.dump(empty_nb, f)

# Scaffold Projects
projects = [
    "01-Fraud-Detection",
    "02-Insurance-Claim-Prediction",
    "03-Marketing-Response-Prediction",
    "04-Customer-Lifetime-Value-Prediction",
    "05-Kaggle-Style-Classification",
    "06-Large-Scale-Customer-Prediction",
    "07-E-Commerce-Purchase-Prediction"
]

subdirs = ["data", "notebooks", "src", "models", "reports", "images"]
projects_dir = os.path.join(base_dir, "projects")
os.makedirs(projects_dir, exist_ok=True)

for p in projects:
    p_path = os.path.join(projects_dir, p)
    os.makedirs(p_path, exist_ok=True)
    
    # Create subdirs and .gitkeep
    for sub in subdirs:
        sub_path = os.path.join(p_path, sub)
        os.makedirs(sub_path, exist_ok=True)
        with open(os.path.join(sub_path, ".gitkeep"), "w") as f:
            f.write("")
            
    # Create requirements.txt
    with open(os.path.join(p_path, "requirements.txt"), "w") as f:
        f.write("pandas\nnumpy\nscikit-learn\nmatplotlib\nseaborn\njupyter\n")

    # Create README.md
    with open(os.path.join(p_path, "README.md"), "w") as f:
        f.write(f"# {p.replace('-', ' ')}\n\nPortfolio Project Architecture.\n")

print("Ensemble Projects and Notebooks scaffolded successfully.")
