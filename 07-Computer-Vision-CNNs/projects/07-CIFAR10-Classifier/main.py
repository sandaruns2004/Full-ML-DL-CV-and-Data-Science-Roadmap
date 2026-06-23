import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 8x8
        )
        self.classifier = nn.Linear(32 * 8 * 8, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class ImageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_model(model: nn.Module, loader: DataLoader, epochs: int = 5) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
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
            
    return correct / total

def main():
    print("Running Project 01: CIFAR-10 Image Classifier (Simulated Data)...")
    torch.manual_seed(42)
    
    # Generate mock CIFAR-like dataset (3 channels, 32x32 size)
    X = torch.randn(300, 3, 32, 32)
    y = (X[:, 0, 0, 0] + X[:, 1, 0, 0] > 0.0).long()
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    cnn = ImageCNN()
    mlp = ImageMLP()
    
    cnn_acc = train_model(cnn, loader, epochs=5)
    mlp_acc = train_model(mlp, loader, epochs=5)
    
    print(f"CNN Final Accuracy: {cnn_acc * 100:.2f}%")
    print(f"MLP Final Accuracy: {mlp_acc * 100:.2f}%")
    print("Project 01 finished successfully.")

if __name__ == "__main__":
    main()
