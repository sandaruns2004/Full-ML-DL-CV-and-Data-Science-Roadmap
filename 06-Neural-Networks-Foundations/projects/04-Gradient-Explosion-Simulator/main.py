import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def analyze_gradients(num_layers: int, init_std: float, activation_fn) -> list[float]:
    torch.manual_seed(42)
    
    # Construct sequence of layers
    layers = []
    in_dim = 50
    for _ in range(num_layers):
        layer = nn.Linear(in_dim, in_dim, bias=False)
        # Manually scale initialization
        nn.init.normal_(layer.weight, mean=0.0, std=init_std)
        layers.append(layer)
        layers.append(activation_fn())
        
    model = nn.Sequential(*layers)
    
    # Input tensor
    x = torch.randn(10, in_dim)
    
    # Forward pass
    out = model(x)
    
    # Compute pseudo-loss
    loss = out.pow(2).sum()
    loss.backward()
    
    # Extract gradient norms per layer
    grad_norms = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            if layer.weight.grad is not None:
                grad_norms.append(layer.weight.grad.norm().item())
                
    return grad_norms

def main():
    print("Running Project 04: Gradient Explosion/Vanishing Simulator...")
    
    num_layers = 25
    
    # 1. Vanishing Gradients (Sigmoid activation and small weight initialization)
    vanishing_norms = analyze_gradients(num_layers, 0.05, nn.Sigmoid)
    
    # 2. Exploding Gradients (ReLU activation and large weight initialization)
    exploding_norms = analyze_gradients(num_layers, 1.8, nn.ReLU)
    
    # 3. Stable Gradients (ReLU activation and proper He initialization std = sqrt(2/fan_in))
    he_std = (2.0 / 50) ** 0.5
    stable_norms = analyze_gradients(num_layers, he_std, nn.ReLU)
    
    plt.figure(figsize=(10, 6))
    plt.plot(vanishing_norms, label="Vanishing (Sigmoid + small weights)", marker='o')
    plt.plot(exploding_norms, label="Exploding (ReLU + large weights)", marker='s')
    plt.plot(stable_norms, label="Stable (ReLU + He Init)", marker='^', lw=2)
    plt.yscale('log')
    plt.xlabel("Linear Layer index (Input -> Output)")
    plt.ylabel("Gradient Norm (log scale)")
    plt.title("Gradient Flow Across 25 Layers", fontweight='bold')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    plot_path = "gradient_simulation.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Visualization plot saved to '{plot_path}'")
    plt.close()
    print("Project 04 finished successfully.")

if __name__ == "__main__":
    main()
