# 📈 Experiment Tracking

> **Prerequisites**: ML Pipeline | **Difficulty**: ⭐⭐☆☆☆ Elementary

---

## 📋 Table of Contents
1. [What Needs to be Tracked?](#1-what-needs-to-be-tracked)
2. [Weights & Biases (W&B)](#2-weights--biases-wb)
3. [MLflow](#3-mlflow)
4. [Summary](#summary)

---

In traditional software engineering, code is tracked using version control systems like Git. However, Machine Learning involves not just code, but also **Data**, **Hyperparameters**, and **Model Weights**. 

If a data scientist trains a model that achieves 95% accuracy on Tuesday, and then continues tweaking the code and data until Friday when the accuracy drops to 80%, they need to be able to answer: *"What exactly did I do on Tuesday?"*

**Experiment Tracking** tools solve this by automatically logging everything related to an ML training run.

---

## 1. What Needs to be Tracked?

A robust experiment tracking system should log:

1.  **Code Version**: The Git commit hash.
2.  **Hyperparameters**: Learning rate, batch size, number of layers, optimizer used, dropout rate.
3.  **Metrics**: Training loss, validation accuracy, F1-score, recorded at every epoch.
4.  **Artifacts**: The trained model weights (`.pt` or `.h5` files), the pre-processed dataset.
5.  **Environment**: Python version, library versions (e.g., `requirements.txt`), GPU type.
6.  **Visualizations**: Confusion matrices, PR curves, sample images from the model output.

---

## 2. Weights & Biases (W&B)

[Weights & Biases (wandb)](https://wandb.ai) is currently the most popular cloud-based experiment tracking tool, especially in deep learning. It integrates with almost every framework (PyTorch, TensorFlow, HuggingFace, Keras) with just a few lines of code.

### 2.1 Basic W&B Integration

Here is how you integrate W&B into a standard PyTorch training loop:

```python
# pip install wandb
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Initialize the W&B run
# Project: Groups related experiments together
# Config: A dictionary of all your hyperparameters
wandb.init(
    project="my-awesome-image-classifier",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "architecture": "ResNet18"
    }
)

# You can access the config anywhere using wandb.config
config = wandb.config

model = get_my_model(config.architecture)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(config.epochs):
    # ... training loop ...
    train_loss = train(model, optimizer, criterion)
    val_accuracy = evaluate(model)
    
    # 2. Log metrics to W&B
    # This automatically creates live updating charts on the W&B dashboard
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuracy": val_accuracy
    })

# 3. Log Artifacts (Save the trained model)
torch.save(model.state_dict(), "model.pth")
artifact = wandb.Artifact("resnet-model", type="model")
artifact.add_file("model.pth")
wandb.log_artifact(artifact)

# Finish the run
wandb.finish()
```

### 2.2 W&B Sweeps (Hyperparameter Optimization)

W&B has a built-in system called "Sweeps" to automatically search for the best hyperparameters using Random Search, Grid Search, or Bayesian Optimization.

1. You define a sweep configuration (usually a YAML file):
```yaml
method: bayes # Bayesian optimization
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
```
2. You run the sweep agent, which spins up multiple training runs, automatically testing different combinations from the YAML file and plotting the results on an interactive dashboard.

---

## 3. MLflow

[MLflow](https://mlflow.org/) is an open-source platform for the complete machine learning lifecycle. While W&B is highly focused on tracking and visualizing training (and is typically cloud-hosted), MLflow is broader and is often hosted internally on a company's own servers.

MLflow has four primary components:
1.  **MLflow Tracking**: Logging parameters, code versions, metrics, and output files.
2.  **MLflow Projects**: Packaging ML code in a reusable, reproducible format.
3.  **MLflow Models**: Managing and deploying models from a variety of ML libraries to a variety of model serving and inference platforms.
4.  **MLflow Model Registry**: A central repository to collaboratively manage the full lifecycle of an MLflow Model (Staging -> Production -> Archived).

### 3.1 Basic MLflow Integration

```python
# pip install mlflow
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Enable auto-logging (works for scikit-learn, xgboost, lightgbm, pytorch lightning, etc.)
# This automatically logs all hyperparameters and metrics without needing explicit log() calls!
mlflow.autolog()

with mlflow.start_run(run_name="Random_Forest_Run"):
    # Define parameters
    n_estimators = 100
    max_depth = 5
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # If not using autolog(), you would manually log:
    # mlflow.log_param("n_estimators", n_estimators)
    # mlflow.log_metric("accuracy", accuracy)
    # mlflow.sklearn.log_model(model, "model")

print("Run completed. View results by typing 'mlflow ui' in your terminal.")
```

### 3.2 MLflow UI

Once a run is complete, typing `mlflow ui` in the terminal spins up a local web server (usually at `http://localhost:5000`) where you can:
*   View all historical runs side-by-side in a table.
*   Filter runs (e.g., `metrics.accuracy > 0.90`).
*   Select two runs and generate a visual "diff" to see exactly what parameters changed between Tuesday's run and Friday's run.

---

## Summary

Never rely on printing metrics to the console or saving hyperparameters in a spreadsheet. 
*   Use **Weights & Biases** for deep learning projects where real-time visualization of loss curves, images, and gradients is crucial.
*   Use **MLflow** if you are building an internal company platform, need robust model registry capabilities (Staging vs. Production), or are working heavily with scikit-learn/XGBoost.

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Write a simple script to train a Random Forest and log it to a local MLflow instance.
- 🟡 **Intermediate**: Set up a W&B Sweep to optimize hyperparameters for a PyTorch neural network.

### What's Next
| Next | Why |
|------|-----|
| [Beginner Projects](../16-Projects/01-Beginner-Projects.md) | Start building end-to-end projects. |

---

[← Edge ML & Optimization](05-Edge-ML-And-Optimization.md) | [Back to Index](../README.md) | [Next: Distributed Training →](07-Distributed-Training.md)
