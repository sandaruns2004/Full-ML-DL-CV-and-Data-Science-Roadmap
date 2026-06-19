# 🛠️ Deep Learning Frameworks: Keras vs PyTorch (Deep Dive)

> **Prerequisites**: Regularization, Initialization | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Computational Graph (Autograd)](#1-the-computational-graph-autograd)
2. [TensorFlow / Keras (High-Level API)](#2-tensorflow--keras-high-level-api)
3. [PyTorch (Dynamic Computation Graphs)](#3-pytorch-dynamic-computation-graphs)
4. [A Professional PyTorch Training Pipeline](#4-a-professional-pytorch-training-pipeline)
5. [Side-by-Side Comparison](#5-side-by-side-comparison)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The Computational Graph (Autograd)

Frameworks save us from writing backpropagation by hand using **Automatic Differentiation (Autograd)**.

When you perform mathematical operations on Tensors (GPU-accelerated arrays), the framework implicitly builds a Directed Acyclic Graph (DAG).
- **Nodes** are the Tensors.
- **Edges** are the mathematical operations.

When you call `.backward()`, the framework traverses this graph in reverse, applying the chain rule to compute gradients perfectly, down to the floating-point limit.

---

## 2. TensorFlow / Keras (High-Level API)

Keras operates by defining the graph statically (mostly) and compiling it into an optimized execution engine (XLA). This makes it incredibly fast for production deployment.

### The Functional API

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add
from tensorflow.keras.models import Model

# Functional API (Better than Sequential as it allows branching/skip connections)
inputs = Input(shape=(100,))

x = Dense(64, activation='relu')(inputs)
x = BatchNormalization()(x)

# Skip Connection (ResNet style)
residual = x
x = Dense(64, activation='relu')(x)
x = Add()([x, residual])  # Merge branch

x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Callbacks and Training
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
# model.fit(X_train, y_train, validation_split=0.2, epochs=50, callbacks=[early_stop])
```

---

## 3. PyTorch (Dynamic Computation Graphs)

PyTorch uses **Define-by-Run**. The computation graph is built dynamically on the fly as your Python code executes. This allows standard Python debuggers (`pdb`) to step into the exact math happening on the GPU.

### Object-Oriented Network Definition

```python
import torch
import torch.nn as nn

class ResidualMLP(nn.Module):
    def __init__(self, input_size=100):
        super(ResidualMLP, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.fc2 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define the forward pass (The graph is built here dynamically!)
        x1 = self.relu(self.bn1(self.fc1(x)))
        
        # Skip connection
        x2 = self.relu(self.fc2(x1))
        x = x1 + x2
        
        x = self.dropout(x)
        return self.sigmoid(self.out(x))
```

---

## 4. A Professional PyTorch Training Pipeline

In real-world projects, you don't just write a single `for` loop. You need a robust pipeline that handles Data Loading, Training, Validation, Model Checkpointing, and Early Stopping.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import copy

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    """
    A complete PyTorch training loop with Validation and Early Stopping.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode (enables Dropout/BatchNorm)
            else:
                model.eval()   # Set model to evaluate mode (disables Dropout/BatchNorm)

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float() # Binary classification expects floats

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

            # Epoch loss
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            history[f'{phase}_loss'].append(epoch_loss)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')

            # Deep copy the model if it's the best validation loss
            if phase == 'val':
                scheduler.step(epoch_loss) # Reduce LR on plateau
                
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

    print(f'Training complete. Best Val Loss: {best_val_loss:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# ==========================================
# Example Usage:
# ==========================================
# 1. Create Dummy Data
# X = torch.randn(1000, 100)
# y = torch.randint(0, 2, (1000,))
# dataset = TensorDataset(X, y)
# train_size = int(0.8 * len(dataset))
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

# 2. Create DataLoaders
# dataloaders = {
#     'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
#     'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
# }

# 3. Initialize Model, Loss, Optimizer
# model = ResidualMLP(input_size=100)
# criterion = nn.BCELoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 4. Train!
# best_model, hist = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=50)
```

---

## 5. Side-by-Side Comparison

| Feature | TensorFlow/Keras | PyTorch |
|---------|------------------|---------|
| **Core Philosophy** | Static Graph / Production First | Dynamic Graph / Research First |
| **Learning Curve** | Gentle (`model.fit()`) | Steep (Explicit loops & data loaders) |
| **Debugging** | Harder (Errors map to C++ engine) | Easy (Standard Python tracebacks) |
| **Deployment (Mobile/Edge)** | Excellent (TFLite) | Improving rapidly (TorchScript) |
| **Industry Adoption** | High in Legacy Enterprise | Dominant in AI Research & Generative AI Startups |

**Recommendation**: 
Learn **PyTorch**. If you understand PyTorch, you understand exactly how Deep Learning works under the hood. Most modern architectures (Transformers, Diffusion models) are researched and released exclusively in PyTorch first.

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **PyTorch Subclassing**: Take a standard Scikit-Learn `MLPClassifier` and rewrite it entirely as a PyTorch `nn.Module`. Train it on the Iris dataset using the professional training pipeline provided above. Plot the loss curves using matplotlib.

### What's Next
| Next | Why |
|------|-----|
| [RNN Fundamentals](../08-Sequence-Models/01-RNN-Fundamentals.md) | We've mastered standard MLPs. Now we move into sequence data (Time Series, Text) by introducing Recurrent Neural Networks. |

---

[← Weight Initialization](./07-Weight-Initialization.md) | [Back to Index](../README.md) | [Next: DL Methods And Functions →](./09-DL-Methods-And-Functions.md)
