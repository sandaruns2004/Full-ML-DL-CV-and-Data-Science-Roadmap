# 01 - Bank Check Digit Reader

## 🎯 Objective
Learn how to apply a basic Artificial Neural Network (MLP) to recognize handwritten digits, demonstrating how machines can process visual data like bank checks.

## 🧠 Concepts Covered
- PyTorch Datasets & DataLoaders (`torchvision.datasets.MNIST`)
- Flattening multidimensional arrays (28x28 image to 784 1D array)
- Multi-Layer Perceptrons (MLPs) and Activation Functions (ReLU)
- Softmax output for multiclass classification

## 🚀 Getting Started

### 1. Training the Model
We provide a comprehensive Jupyter Notebook that will walk you through downloading the MNIST dataset, defining the MLP architecture, and training the model.
1. Navigate to the `notebooks/` directory.
2. Open and run `02-Model-Training.ipynb`. 
3. This will generate a `model.pth` file in the `src/` folder.

### 2. Testing the Web App
Once trained, you can interact with your model using the provided Streamlit dashboard. You can literally draw a digit on a canvas and watch your neural network predict it in real-time!
```bash
streamlit run app.py
```

## 📂 Project Structure
```
01-Bank-Check-Digit-Reader/
│
├── notebooks/
│   └── 02-Model-Training.ipynb   # Architecture & Training loop
├── src/
│   └── model.pth                 # Saved weights (generated after training)
├── app.py                        # Streamlit drawing dashboard
└── README.md
```
