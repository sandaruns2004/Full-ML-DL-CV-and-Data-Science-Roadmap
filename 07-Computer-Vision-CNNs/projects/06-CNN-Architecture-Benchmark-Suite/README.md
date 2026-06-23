# Project 06: CNN Architecture Benchmark Suite

> [!NOTE]  
> As a Computer Vision Engineer, you will constantly have to choose between different architectures. Is ResNet-50 better than EfficientNet-B0 for your specific problem? In this project, you will build a framework to empirically answer that question.

## The Goal
Code a rigorous benchmarking suite that trains, tests, and compares several famous CNN architectures on the same dataset.

## Requirements

1. **Dataset**: Use a medium-complexity dataset like CIFAR-100 or a subset of ImageNet (e.g., Tiny ImageNet).
2. **The Contenders**: Implement training pipelines for at least three of the following:
   - ResNet-18 or ResNet-50
   - VGG-16
   - MobileNetV2
   - EfficientNet-B0
3. **The Benchmark Script**: Write a unified script that loops through the models, training them with the exact same hyperparameters (Learning Rate, Batch Size, Optimizer, Epochs).
4. **Metrics to Capture**:
   - Final Validation Accuracy.
   - Total Training Time.
   - Inference Speed (How many milliseconds it takes to predict a single image).
   - Number of Parameters (Model Size).
5. **The Report**: Automatically generate a Markdown table or a Pandas DataFrame summarizing the results.

## Bonus Challenge
Create a scatter plot using `matplotlib` or `seaborn` where the X-axis is "Inference Time" and the Y-axis is "Validation Accuracy". This visual trade-off curve is highly valued in industry presentations.
