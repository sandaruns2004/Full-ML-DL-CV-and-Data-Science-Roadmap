# Project 03: Plant Disease Detection System

> [!NOTE]  
> Let's solve a real-world problem. In agriculture, catching crop diseases early can save entire harvests. This project simulates an industry workflow.

## The Goal
Build a multi-class CNN capable of identifying various plant diseases from images of their leaves.

## Requirements

1. **Dataset**: Use the "PlantVillage" dataset (contains healthy and diseased leaves across multiple crop types like Apple, Corn, and Tomato).
2. **Architecture Strategy**: 
   - Choose a modern, efficient architecture (e.g., **EfficientNet-B0**).
   - Use fine-tuning: Train the classification head first, then unfreeze the top 30% of the network and train with a very low learning rate.
3. **Class Imbalance**: The dataset may not be perfectly balanced. Implement **Class Weights** in your loss function to penalize misclassifying rare diseases.
4. **Evaluation**: Because a False Negative (missing a disease) is costly, evaluate your model using a **Confusion Matrix** and calculate **Recall** for the disease classes.

## Bonus Challenge
Use **Grad-CAM** to visualize the predictions. Show a heatmap of a diseased leaf to prove the CNN is actually looking at the spots/lesions on the leaf, and not just the green background.
