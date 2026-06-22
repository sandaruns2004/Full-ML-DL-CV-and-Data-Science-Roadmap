import os
import json

base_dir = r"c:\Users\ADMIN\Desktop\Full-ML-DL-CV-and-Data-Science-Roadmap\03-Ensemble-Methods\notebooks"

notebooks = [
    "01_Random_Forest_Lab.ipynb",
    "02_Extra_Trees_Lab.ipynb",
    "03_AdaBoost_Lab.ipynb",
    "04_Gradient_Boosting_Lab.ipynb",
    "05_XGBoost_Lab.ipynb",
    "06_LightGBM_Lab.ipynb",
    "07_CatBoost_Lab.ipynb"
]

headers = [
    "## Problem Overview",
    "## Dataset Exploration",
    "## Data Cleaning",
    "## Feature Engineering",
    "## From Scratch Model",
    "## NumPy Model",
    "## Scikit-Learn / Native API Model",
    "## Visualization Lab",
    "## Hyperparameter Experiments",
    "## Model Comparison",
    "## Failure Analysis",
    "## Mini Challenges",
    "## Solutions",
    "## Key Takeaways"
]

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    }

def create_code_cell():
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": []
    }

for nb in notebooks:
    algo_name = nb.split('_', 1)[1].replace('_Lab.ipynb', '').replace('_', ' ')
    
    cells = [create_markdown_cell(f"# 🧪 {algo_name} Practical Workshop\n\nWelcome to the hands-on lab for {algo_name}. Please complete the sections below.")]
    
    for header in headers:
        cells.append(create_markdown_cell(header))
        cells.append(create_code_cell())
        
    notebook_content = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(os.path.join(base_dir, nb), "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)

print("Notebooks populated with workshop structures successfully.")
