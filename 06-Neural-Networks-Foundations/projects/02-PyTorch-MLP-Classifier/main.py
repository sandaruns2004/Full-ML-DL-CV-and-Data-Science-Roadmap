import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class PyTorchClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def main():
    print("Running Project 02: PyTorch MLP Classifier...")
    
    torch.manual_seed(42)
    
    # Create simple binary dataset
    X = torch.randn(200, 4)
    # Target: 1 if first two features sum to > 0, else 0
    y = (X[:, 0] + X[:, 1] > 0.0).long()
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PyTorchClassifier(in_dim=4, hidden_dim=8, out_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    for epoch in range(10):
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
            
        epoch_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy*100:.1f}%")
        
    print("Project 02 finished successfully.")

if __name__ == "__main__":
    main()
