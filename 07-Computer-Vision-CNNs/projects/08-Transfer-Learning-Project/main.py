import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

def main():
    print("Running Project 02: Transfer Learning Project...")
    torch.manual_seed(42)
    
    # 1. Initialize synthetic data: 128 images, 3 channels, 224x224 input size
    X = torch.randn(64, 3, 224, 224)
    y = torch.randint(0, 2, (64,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. Setup pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze convolutional weights
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final head (fully connected layer)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Only train the fc parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=0.01)
    
    # 3. Train
    model.train()
    for epoch in range(3):
        total_loss = 0.0
        correct = 0
        total = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss / total:.4f} | Accuracy: {(correct / total)*100:.2f}%")
        
    print("Project 02 finished successfully.")

if __name__ == "__main__":
    main()
