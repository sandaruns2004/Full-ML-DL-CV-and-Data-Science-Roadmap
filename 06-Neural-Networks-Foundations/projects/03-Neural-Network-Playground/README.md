# 🛝 Project 03: Neural Network Playground

## 🎯 Goal
Build an interactive web application (using Streamlit) that allows users to construct, train, and visualize a neural network in real-time right in their browser.

## 📝 Description
Inspired by the famous "TensorFlow Playground," this project focuses on the intersection of deep learning and frontend engineering. You will create a UI where users can change hyperparameters on the fly and watch a live graph of the decision boundary warping to fit the data.

## ✅ Requirements
- Build a UI using `Streamlit`.
- Provide controls for: Number of Hidden Layers, Neurons per Layer, Learning Rate, and Activation Function.
- Use `scikit-learn.datasets` to generate 2D toy datasets (moons, circles, blobs).
- Build the dynamic PyTorch model under the hood.
- Plot the live decision boundary using `matplotlib` or `plotly` updating continuously during the training loop.

## 📁 Files
- `app.py` (Run with `streamlit run app.py`)
- `model.py`
- `utils.py`
