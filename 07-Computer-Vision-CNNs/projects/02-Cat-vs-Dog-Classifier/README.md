# 02 - Cat vs Dog Classifier

## 🎯 Objective
Build a binary image classification system using a custom Convolutional Neural Network (CNN) to distinguish between images of cats and dogs.

## 🧠 Concepts Covered
- **Custom CNN Architecture**: Building deep networks with multiple `Conv2d` and `MaxPool2d` layers.
- **Image Augmentation**: Using `torchvision.transforms` to artificially expand the dataset.
- **Binary Cross Entropy**: Using `BCELoss` with a Sigmoid output for binary classification.

## 🚀 Getting Started

### 1. Training the Model
If you have a dataset of Cat and Dog images, place them in `../data/cats_dogs/` (with subfolders `cats/` and `dogs/`).
If you do not have the dataset, running the script will automatically generate a mock "FakeData" dataset to demonstrate the training loop without crashing.
```bash
cd src/
python train.py
```
This will output a `model.pth` weight file.

### 2. Running the Web App
We have provided an interactive Streamlit dashboard where you can upload an image from your computer to test the model's prediction.
```bash
streamlit run app.py
```

## 📂 Project Structure
```
02-Cat-vs-Dog-Classifier/
│
├── src/
│   └── train.py          # PyTorch training loop and CNN architecture
├── app.py                # Streamlit Web App
└── README.md
```
