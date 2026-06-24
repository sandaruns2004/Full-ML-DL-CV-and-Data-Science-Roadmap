# 03 - Plant Disease Detection System

## 🎯 Objective
Create a multi-class image classification system using Convolutional Neural Networks (CNNs) to identify healthy plants and various plant diseases from leaf images.

## 🧠 Concepts Covered
- **Multi-Class Classification**: Using `CrossEntropyLoss` for more than 2 categories.
- **Deeper CNN Architectures**: Stacking more convolution and pooling layers.
- **Handling Multi-Class Outputs**: Using `softmax` and `torch.max` to extract the winning class prediction and confidence score.

## 🚀 Getting Started

### 1. Training the Model
If you have the PlantVillage dataset (or a subset), place it in `../data/plant_village/` with subfolders for each class (e.g., `Apple_Healthy/`, `Apple_Scab/`). 
If you do not have the dataset, running the script will automatically generate a mock "FakeData" dataset.
```bash
cd src/
python train.py
```
This outputs `model.pth` (the weights) and `classes.json` (the list of class names).

### 2. Running the Web App
We have provided an interactive Streamlit dashboard where you can upload a leaf image to test the model's diagnostic capabilities.
```bash
streamlit run app.py
```

## 📂 Project Structure
```
03-Plant-Disease-Detection-System/
│
├── src/
│   ├── train.py          # PyTorch training loop for multi-class CNN
│   ├── model.pth         # Saved weights (generated after training)
│   └── classes.json      # Saved class names mapping (generated after training)
├── app.py                # Streamlit Diagnostic Web App
└── README.md
```
