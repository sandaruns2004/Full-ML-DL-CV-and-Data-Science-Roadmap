# 🔥 Neural Network PyTorch Implementation

---

## 📋 Table of Contents
1. [Beginner: Introduction to Tensors and PyTorch](#1-beginner-introduction-to-tensors-and-pytorch)
2. [Intermediate: Training Pipeline with Datasets and Modules](#2-intermediate-training-pipeline-with-datasets-and-modules)
3. [Advanced: Custom Autograd Functions & Compilation Engine](#3-advanced-custom-autograd-functions--compilation-engine)

---

## 1. Beginner: Introduction to Tensors and PyTorch

### Simple Explanation
While implementing neural networks in NumPy is great for learning, it is too slow for training large networks on massive datasets. Industry-standard frameworks like **PyTorch** solve this. 
PyTorch replaces NumPy arrays with **Tensors**—n-dimensional arrays that look and behave like NumPy arrays but can run on graphics cards (GPUs) to run operations thousands of times faster. PyTorch also has built-in **Autograd**, which automatically tracks all calculations to compute gradients for backpropagation.

### Real-World Analogy: The Magic Accounting Book
Imagine you run a business:
- In NumPy, you have to write down all transactions in a ledger, manually calculate tax forms, and do the math at the end of the year by hand.
- In PyTorch, your accounting book is magic. As you write down expenses and profits (forward pass), the book automatically fills out your tax forms and tells you exactly how much money each product made or cost you (Autograd).

---

## 2. Intermediate: Training Pipeline with Datasets and Modules

Let us construct a clean, professional PyTorch training pipeline using PyTorch's native structural blocks:
- `nn.Module` for model construction.
- `Dataset` and `DataLoader` for batch feeding.
- `optim.AdamW` for optimization.
- `ReduceLROnPlateau` for scheduling learning rates.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. Custom Dataset Definition
class RandomRegressionDataset(Dataset):
    """Generates a random regression dataset for demonstration."""
    def __init__(self, num_samples: int = 1000, num_features: int = 10):
        self.X = torch.randn(num_samples, num_features)
        # Target: y = X * w + noise
        self.y = self.X[:, 0:1] * 2.5 - self.X[:, 1:2] * 1.5 + torch.randn(num_samples, 1) * 0.1

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

# 2. Model Structure Definition
class MLPRegressor(nn.Module):
    """Multi-Layer Perceptron (MLP) Regressor using PyTorch."""
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# 3. Training Pipeline
def train_pipeline():
    # Set seeds
    torch.manual_seed(42)
    
    # Initialize Data
    dataset = RandomRegressionDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # Initialize Model, Optimizer, Loss, Scheduler
    model = MLPRegressor(in_features=10, hidden_features=16)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training Loop
    epochs = 15
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Scheduler Step
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    train_pipeline()
```

---

## 3. Advanced: Custom Autograd Functions & Compilation Engine

### Custom Autograd Functions
Sometimes you want to create a non-standard mathematical operation that PyTorch does not support out of the box, or you want to write a custom backward pass to improve numerical stability or speed. 

We can achieve this by inheriting from `torch.autograd.Function`. Let's implement a custom activation function: **Exp-ReLU** ($f(x) = e^x - 1$ if $x \le 0$, else $x$), and write its derivative by hand.

```python
class ExpReLU(torch.autograd.Function):
    """
    Custom activation function:
    Forward: f(x) = x if x > 0 else exp(x) - 1
    Backward: f'(x) = 1 if x > 0 else exp(x)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # Save input tensor for use in backward pass
        ctx.save_for_backward(x)
        return torch.where(x > 0, x, torch.exp(x) - 1.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Retrieve saved input tensor
        x, = ctx.saved_tensors
        # Compute gradient
        grad_input = torch.where(x > 0, 1.0, torch.exp(x))
        # Return downstream gradient (upstream * local)
        return grad_output * grad_input

# Usage
x_tensor = torch.tensor([-1.0, 2.0], requires_grad=True)
activation = ExpReLU.apply
y_tensor = activation(x_tensor)
loss = y_tensor.sum()
loss.backward()

print("Inputs:", x_tensor.data)
print("Forward output:", y_tensor.data)
print("Gradients:", x_tensor.grad) # Expected: [exp(-1), 1.0] -> [0.3679, 1.0]
```

### PyTorch compilation (`torch.compile`)
In PyTorch 2.0+, you can optimize your network by compiling it. Simply call `torch.compile`:

```python
compiled_model = torch.compile(model)
```

**How it works under the hood**:
1. **TorchDynamo**: Intercepts Python execution before compilation and extracts the PyTorch operations into a clean graph.
2. **AOTAutograd**: Generates the backward graph automatically before compilation.
3. **TorchInductor**: Translates the operations into efficient C++/Triton kernels specifically optimized for your target GPU architecture, fusing operations to reduce memory reads and writes.

---

[← Neural Network from Scratch (NumPy)](13-Neural-Network-From-Scratch-Numpy.md) | [Back to Index](../README.md) | [Next: Introduction to Computer Vision →](../07-Computer-Vision-CNNs/01-Introduction-To-Computer-Vision.md)
