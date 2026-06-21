import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class ConfigurableMLP(nn.Module):
    def __init__(self, activation_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 32),
            activation_cls(),
            nn.Linear(32, 32),
            activation_cls(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def run_experiment(activation_cls, X, y, epochs=100) -> list[float]:
    torch.manual_seed(42)
    model = ConfigurableMLP(activation_cls)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def main():
    print("Running Project 03: Activation Function Comparison Study...")
    
    # Generate random binary classification dataset
    torch.manual_seed(42)
    X = torch.randn(500, 20)
    # y target defined by linear combination of features + noise
    y = (torch.sum(X[:, 0:5], dim=-1, keepdim=True) > 0.0).float()
    
    activations = {
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'ReLU': nn.ReLU,
        'Leaky ReLU': lambda: nn.LeakyReLU(0.1)
    }
    
    plt.figure(figsize=(10, 6))
    
    for name, act_cls in activations.items():
        losses = run_experiment(act_cls, X, y, epochs=150)
        plt.plot(losses, label=name, lw=2)
        print(f" - Completed experiment for {name}. Final Loss: {losses[-1]:.4f}")
        
    plt.title("Activation Function Loss Convergence Comparison", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = "activation_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Visualization plot saved to '{plot_path}'")
    plt.close()
    print("Project 03 finished successfully.")

if __name__ == "__main__":
    main()
