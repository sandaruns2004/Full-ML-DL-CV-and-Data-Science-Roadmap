# 🛠️ CNN From Scratch (PyTorch)

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Prerequisites**: Entire CNN Module | **Estimated Reading Time**: 35 Minutes

---

## 📋 Table of Contents
1. [The Capstone Philosophy](#1-the-capstone-philosophy)
2. [Step 1: Imports and Hardware Setup](#2-step-1-imports-and-hardware-setup)
3. [Step 2: Data Loading & Augmentation](#3-step-2-data-loading--augmentation)
4. [Step 3: The Architecture Definition](#4-step-3-the-architecture-definition)
5. [Step 4: The Training Loop](#5-step-4-the-training-loop)
6. [Step 5: The Evaluation Loop](#6-step-5-the-evaluation-loop)
7. [Module Conclusion](#7-module-conclusion)

---

## 1. The Capstone Philosophy

Reading theory is essential, but Neural Networks are a highly applied science. You do not truly understand Computer Vision until you can write the full pipeline from memory without crashing the GPU.

This file provides the complete, unbroken, production-ready PyTorch template for training a CNN from scratch. It incorporates everything we have learned: Augmentation, The VGG Block, ReLU, Cross-Entropy, and the Epoch Loop.

---

## 2. Step 1: Imports and Hardware Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CRITICAL: Always explicitly define your device. 
# Training a CNN on a CPU will take days.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

---

## 3. Step 2: Data Loading & Augmentation

```python
# 1. Define Training Transforms (WITH Augmentation)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(), # Converts 0-255 to 0.0-1.0 and changes shape to [C,H,W]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Mean and Std
])

# 2. Define Test Transforms (NO Augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. Download CIFAR-10 Dataset (32x32 color images)
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# 4. Create DataLoaders (Handles batching and shuffling)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
```

---

## 4. Step 3: The Architecture Definition

```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # FEATURE EXTRACTOR (VGG-Style Blocks)
        self.features = nn.Sequential(
            # Block 1
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # Shape is now: 32 x 16 x 16
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # Shape is now: 64 x 8 x 8
        )
        
        # CLASSIFIER HEAD
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization to prevent overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Instantiate and move to GPU
model = CustomCNN().to(device)
```

---

## 5. Step 4: The Training Loop

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # 1. Move data to GPU
        images, labels = images.to(device), labels.to(device)
        
        # 2. Zero the parameter gradients
        optimizer.zero_grad()
        
        # 3. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 4. Backward pass (Calculate gradients)
        loss.backward()
        
        # 5. Optimize (Update weights)
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f}")
```

---

## 6. Step 5: The Evaluation Loop

```python
model.eval() # Set model to evaluation mode (turns off Dropout)
correct = 0
total = 0

# Turn off gradient calculation for speed and memory saving
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        
        # Get the class index with the highest probability
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')
```

---

## 7. Module Conclusion

### Congratulations!
You have completed the **Computer Vision & CNNs** module. You have journeyed from understanding what a pixel is, through the mathematics of Convolutions and Pooling, up to coding complete Neural Networks in PyTorch. 

You are now ready to tackle the real-world projects in the `/projects` directory to solidify your skills.

[← VGG Net](20-VGG-Net.md) | [Return to Module Index](./README.md)
