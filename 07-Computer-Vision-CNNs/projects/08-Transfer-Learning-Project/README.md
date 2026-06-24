# 08 - Transfer Learning Project

## 🎯 Objective
Learn how to leverage a massive pre-trained model (ResNet18 trained on ImageNet) and adapt it to a highly specific, small custom dataset without training from scratch.

## 🧠 Concepts Covered
- **Freezing Weights**: Using `param.requires_grad = False` to freeze the convolutional feature extractors.
- **Replacing the Classification Head**: Stripping off the final 1000-class fully connected layer and replacing it with a new `nn.Linear(num_features, 2)` layer.
- **Fine-Tuning**: Running gradient descent only on the newly added layer to learn the new task rapidly.

## 🚀 Getting Started

This project is documented as an interactive Jupyter Notebook.
1. Ensure you have Jupyter installed (`pip install jupyter`).
2. Navigate to the `notebooks/` directory.
3. Open `Transfer_Learning.ipynb` and execute the cells.

## 📂 Project Structure
```
08-Transfer-Learning-Project/
│
├── notebooks/
│   └── Transfer_Learning.ipynb    # Code demonstrating weight freezing and fine-tuning
│
└── README.md
```
