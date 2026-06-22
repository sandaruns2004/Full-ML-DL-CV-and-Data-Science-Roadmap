# 💻 CNN from Scratch (PyTorch)

---

## 📋 Table of Contents
1. [Introduction](#1-introduction)
2. [CNN Pipeline Architecture](#2-cnn-pipeline-architecture)
3. [PyTorch Implementation Code](#3-pytorch-implementation-code)

---

## 1. Introduction
This module provides a complete, runnable training pipeline for a custom Convolutional Neural Network (CNN) in PyTorch. The code utilizes standard practices such as:
- Custom subclassing of `nn.Module`
- `DataLoader` and custom dataset wrapping
- Correct training and validation evaluation modes (`model.train()`, `model.eval()`, `torch.no_grad()`)
- Dynamic optimization steps using AdamW

---

## 2. CNN Pipeline Architecture

```
[Input Tensor] ---> [Conv2d 3x3] ---> [BatchNorm] ---> [ReLU] ---> [MaxPool 2x2] ---> [Linear Head]
```

---

## 3. PyTorch Implementation Code

Save and run this code in a Python 3.10+ environment with `torch` and `torchvision` installed.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class CustomCNNClassifier(nn.Module):
    """
    Custom 2D Convolutional Neural Network.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # Conv block 1: out (Batch, 16, 16, 16) if input is (Batch, 3, 32, 32)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2: out (Batch, 32, 8, 8)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Linear Head classifier
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        return self.fc(x)

def run_pipeline():
    torch.manual_seed(42)
    
    # 1. Generate synthetic dataset (Batch, Channels, Height, Width)
    X = torch.randn(128, 3, 32, 32)
    # Simple binary classification target
    y = (X[:, 0, 0, 0] + X[:, 1, 0, 0] > 0.0).long()
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 2. Initialize Model, Loss, and Optimizer
    model = CustomCNNClassifier(in_channels=3, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # 3. Train
    model.train()
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
        print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_loss/total:.4f} | Accuracy: {(correct/total)*100:.2f}%")

if __name__ == "__main__":
    run_pipeline()
```

---

[← Image Segmentation Introduction](15-Image-Segmentation-Introduction.md) | [Back to Index](../README.md) | [Next: RNN Fundamentals →](../08-Sequence-Models/01-RNN-Fundamentals.md)
