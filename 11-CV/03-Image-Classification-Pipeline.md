# 🏭 End-to-End Image Classification Pipeline

> **Prerequisites**: CNNs, Albumentations, PyTorch | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [The Anatomy of a Professional Pipeline](#1-the-anatomy-of-a-professional-pipeline)
2. [Step 1: The Dataset Class (Albumentations)](#2-step-1-the-dataset-class-albumentations)
3. [Step 2: Model Initialization (TIMM)](#3-step-2-model-initialization-timm)
4. [Step 3: The Engine (AMP + Accumulation)](#4-step-3-the-engine-amp--accumulation)
5. [Step 4: The Main Execution Loop](#5-step-4-the-main-execution-loop)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The Anatomy of a Professional Pipeline

A toy PyTorch script fits in one file. A professional Deep Learning vision pipeline is modular, hardware-accelerated, and heavily optimized. 

A standard pipeline requires:
1. **`dataset.py`**: A robust PyTorch Dataset utilizing OpenCV for fast disk reading and Albumentations for complex, randomized augmentations.
2. **`model.py`**: Utilizing `timm` (PyTorch Image Models) to instantly pull state-of-the-art architectures with pre-trained weights.
3. **`engine.py`**: The training and validation loops containing Mixed Precision (AMP), Gradient Accumulation, and progress tracking (tqdm).
4. **`train.py`**: The main script that configures hyperparameters, builds DataLoaders, sets up the Scheduler, and kicks off the engine.

Below, we condense these into a single masterclass template.

---

## 2. Step 1: The Dataset Class (Albumentations)

```python
import torch
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VisionDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        df: Pandas DataFrame containing 'filename' and 'label' columns
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = f"{self.image_dir}/{row['filename']}"
        
        # 1. Fast read via OpenCV & convert to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Apply Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image'] # Now it's a PyTorch Tensor
            
        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label

# Example Train Transform
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

---

## 3. Step 2: Model Initialization (TIMM)

Instead of using `torchvision`, the industry standard is `timm` (created by Ross Wightman). It contains over 700 pre-trained state-of-the-art architectures (ConvNeXt, EfficientNet, ViT).

```python
import timm
import torch.nn as nn

def build_model(model_name='convnext_tiny', num_classes=10, pretrained=True):
    # TIMM automatically handles stripping the old classifier head and 
    # attaching a new one with the correct `num_classes`!
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model
```

---

## 4. Step 3: The Engine (AMP + Accumulation)

```python
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    
    # Progress Bar
    pbar = tqdm(dataloader, desc="Training")
    
    for idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Automatic Mixed Precision Forward Pass
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
            
        # Scaled Backward Pass
        scaler.scale(loss).backward()
        
        # Step optimizer only after accumulating enough gradients
        if (idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': running_loss / (idx + 1)})
        
    return running_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        
        # AMP for faster inference too
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        running_loss += loss.item()
        
        # Calculate Accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    acc = correct / total
    return running_loss / len(dataloader), acc
```

---

## 5. Step 4: The Main Execution Loop

```python
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data (Assume train_loader and val_loader are created)
    # ...
    
    # 2. Build Model
    model = build_model('convnext_tiny', num_classes=5).to(device)
    
    # 3. Optimizer & Loss
    # AdamW is strictly superior to Adam for training deep CNNs/Transformers
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Scaler for AMP
    scaler = torch.cuda.amp.GradScaler()
    
    # 5. Advanced Scheduler (OneCycleLR: Warms up the LR, then cosine decays it)
    epochs = 10
    # Dummy len(train_loader), replace with actual
    total_steps = 1000 * epochs 
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps)
    
    # 6. Training Loop
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, accumulation_steps=4)
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step the scheduler AT THE END OF THE EPOCH (or batch, depending on setup)
        # Note: OneCycleLR usually steps per-batch inside train_one_epoch, 
        # but shown here for structural simplicity.
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("🌟 Saved new best model!")

# if __name__ == "__main__":
#     main()
```

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Kaggle Cassava Leaf Disease**: Go to Kaggle, download the Cassava Leaf Disease Classification dataset. Implement the exact pipeline above using `timm`'s `tf_efficientnet_b4_ns` model. Use Albumentations to aggressively augment the data (leaves need rotation, flipping, and color jitter). See if you can break 85% accuracy!

### What's Next
| Next | Why |
|------|-----|
| [Object Detection](./04-Object-Detection.md) | We know how to classify an image. But what if there are 5 different objects in the image, and we need to draw a box around every single one of them? |

---

[← OpenCV Masterclass](./02-OpenCV-Masterclass.md) | [Back to Index](../README.md) | [Next: Object Detection →](./04-Object-Detection.md)
