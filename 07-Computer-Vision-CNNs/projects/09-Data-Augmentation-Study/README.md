# 09 - Data Augmentation Study

## 🎯 Objective
Visualize and understand how artificially augmenting a training dataset can prevent Convolutional Neural Networks from memorizing the data (overfitting).

## 🧠 Concepts Covered
- **Transforms**: Utilizing `torchvision.transforms`.
- **Spatial Augmentation**: Random cropping, horizontal/vertical flipping, and rotation.
- **Color Augmentation**: Adjusting brightness, contrast, saturation, and hue via `ColorJitter`.

## 🚀 Getting Started

This project is a visual study provided as an interactive Jupyter Notebook.
1. Ensure you have Jupyter installed (`pip install jupyter`).
2. Navigate to the `notebooks/` directory.
3. Open `Augmentation_Visualizer.ipynb` and execute the cells to see how a single image is transformed dynamically in memory.

## 📂 Project Structure
```
09-Data-Augmentation-Study/
│
├── notebooks/
│   └── Augmentation_Visualizer.ipynb    # Visualizes the effects of PyTorch transforms
│
└── README.md
```
