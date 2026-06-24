# 06 - Credit Risk Assessment

## 🎯 Objective
Use a Multi-Layer Perceptron (MLP) to assess the probability of a customer defaulting on a loan, focusing heavily on evaluating models built on imbalanced data.

## 🧠 Concepts Covered
- **Imbalanced Datasets**: Handling situations where the target class (Default) is much rarer than the majority class (Paid).
- **Class Weights**: Modifying the PyTorch Loss function (`BCEWithLogitsLoss(pos_weight=...)`) to penalize minority class errors more heavily.
- **Precision/Recall Trade-offs**: Why "Accuracy" is a dangerous metric for credit risk, and how to use Precision-Recall curves instead.

## 🚀 Getting Started

This project is structured as a purely analytical Jupyter Notebook.
1. Ensure you have Jupyter installed (`pip install jupyter`).
2. Navigate to the `notebooks/` directory.
3. Open `Credit_Risk_Modeling.ipynb` to step through the data preparation, weighted neural network training, and Precision-Recall analysis process.

## 📂 Project Structure
```
06-Credit-Risk-Assessment/
│
├── notebooks/
│   └── Credit_Risk_Modeling.ipynb   # Analytical code for imbalanced learning
│
└── README.md
```
