# Project 01: Handwritten Digit Recognition

> [!NOTE]  
> Welcome to the "Hello World" of Computer Vision! In this project, you will build your very first Convolutional Neural Network to recognize handwritten digits.

## The Goal
The MNIST dataset contains 70,000 images of handwritten digits (0-9). Your goal is to build a CNN that can accurately predict the number written in a $28 \times 28$ grayscale pixel grid.

## Requirements

1. **Dataset**: Load the MNIST dataset (built-in to PyTorch/TensorFlow).
2. **Architecture**: Build a simple CNN with at least:
   - 2 Convolutional Layers
   - 2 MaxPooling Layers
   - ReLU Activations
   - 1 Fully Connected (Dense) Output Layer
3. **Training**: Train the model for 5-10 epochs. Track the training and validation loss.
4. **Evaluation**: Build a simple evaluation dashboard (using Matplotlib) that displays:
   - The original image.
   - The model's prediction.
   - The true label.
   - A bar chart of the Softmax probabilities for all 10 classes.

## Bonus Challenge
Find 5 images where the model makes a mistake. Plot them and try to figure out *why* the model was confused (e.g., a badly drawn `7` that looks like a `1`).
