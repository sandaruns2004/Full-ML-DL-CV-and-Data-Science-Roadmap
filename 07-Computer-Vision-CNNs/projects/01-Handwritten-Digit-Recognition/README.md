# 01 - Handwritten Digit Recognition (CNN vs MLP)

## 🎯 Objective
Understand the practical benefits of replacing a standard Multi-Layer Perceptron (MLP) with a Convolutional Neural Network (CNN) when dealing with image data like the MNIST dataset.

## 🧠 Concepts Covered
- **Spatial Hierarchies**: Why flattening an image (MLP) loses spatial context.
- **Convolutional Layers**: Using `nn.Conv2d` to extract local features.
- **Pooling Layers**: Using `nn.MaxPool2d` for downsampling and translation invariance.
- **Feature Maps**: Visualizing what the CNN actually "sees".

## 🚀 Getting Started
This project focuses on code and architecture rather than a web interface. 

To explore the comparison, open the Jupyter Notebook:
1. Ensure you have Jupyter installed (`pip install jupyter`).
2. Navigate to the `notebooks/` directory.
3. Open `CNN_vs_MLP.ipynb` and step through the cells to train and compare both architectures.

## 📂 Project Structure
```
01-Handwritten-Digit-Recognition/
│
├── notebooks/
│   └── CNN_vs_MLP.ipynb    # Training code & Architecture comparison
│
└── README.md
```
