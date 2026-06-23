# Project 02: Cat vs Dog Classifier

> [!NOTE]  
> Now that you've built a CNN from scratch, it's time to tackle larger RGB images. This project introduces Transfer Learning and Data Augmentation.

## The Goal
Build a robust binary classifier that can tell the difference between a Cat and a Dog using high-resolution color images.

## Requirements

1. **Dataset**: Download the Kaggle "Dogs vs. Cats" dataset or use the built-in TensorFlow Datasets / PyTorch Torchvision version.
2. **Transfer Learning**: Do NOT train a CNN from scratch. Instead, load a pre-trained **ResNet-18** or **MobileNetV2**.
   - Freeze the convolutional base.
   - Replace the final classification head with a single dense layer (or two) for binary classification.
3. **Data Augmentation**: Apply the following during training:
   - Random Horizontal Flips
   - Random Rotations (up to 20 degrees)
   - Subtle Color Jitter
4. **Evaluation**: Achieve a validation accuracy of at least **95%**.

## Bonus Challenge
Deploy the model! Write a simple Python script `predict.py` that takes an image path from the command line and outputs `CAT` or `DOG`.
