# 07 - CIFAR10 Classifier (Building ResNet)

## 🎯 Objective
Understand how to solve the Vanishing Gradient problem in Deep Neural Networks by building Residual Blocks (Skip Connections) from scratch.

## 🧠 Concepts Covered
- **Skip Connections**: Adding the input tensor directly to the output of a convolution block (`out += x`).
- **Residual Blocks**: Grouping convolutions, batch normalizations, and ReLU activations into modular classes.
- **Deep Networks**: Using residual blocks to build networks much deeper than a standard sequential CNN.

## 🚀 Getting Started
This project focuses on the core PyTorch engineering required to build modern architectures.

To explore the code:
1. Ensure you have Jupyter installed (`pip install jupyter`).
2. Navigate to the `notebooks/` directory.
3. Open `ResNet_Blocks.ipynb` and step through the implementation of `ResidualBlock`.

## 📂 Project Structure
```
07-CIFAR10-Classifier/
│
├── notebooks/
│   └── ResNet_Blocks.ipynb    # Scratch implementation of ResNet skip connections
│
└── README.md
```
