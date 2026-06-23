import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt

class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(8 * 16 * 16, 2)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_model(X, y, use_augmentation=False, epochs=10) -> list[float]:
    torch.manual_seed(42)
    model = MiniCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Simple data augmentation pipeline
    augment_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15)
    ])
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        inputs = X
        if use_augmentation:
            # Apply transformation to the batch
            inputs = augment_transform(X)
            
        logits = model(inputs)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return losses

def main():
    print("Running Project 03: Data Augmentation Study...")
    
    torch.manual_seed(42)
    # 100 images, 3 channels, 32x32 size
    X = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 2, (100,))
    
    losses_with = train_model(X, y, use_augmentation=True, epochs=25)
    losses_without = train_model(X, y, use_augmentation=False, epochs=25)
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(losses_with, label="With Augmentation", lw=2)
    plt.plot(losses_without, label="Without Augmentation", lw=2, linestyle='--')
    plt.title("Convergence Comparison: Augmentation vs. No Augmentation", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = "augmentation_study.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Visualization plot saved to '{plot_path}'")
    plt.close()
    
    print("Project 03 finished successfully.")

if __name__ == "__main__":
    main()
