import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# XOR input features and targets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Neural Network Architecture: 2 inputs -> 4 hidden -> 1 output
W1 = np.random.randn(2, 4) * 1.0
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 1.0
b2 = np.zeros((1, 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def main():
    global W1, b1, W2, b2
    print("Running Project 05: XOR Problem Solver...")
    
    lr = 0.5
    epochs = 5000
    
    for epoch in range(epochs):
        # Forward pass
        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        
        # Loss (MSE)
        loss = np.mean((A2 - y) ** 2)
        
        # Backward pass
        dZ2 = (A2 - y) * sigmoid_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1:04d} | Loss: {loss:.6f}")

    # Generate decision boundary grid
    x_grid = np.linspace(-0.5, 1.5, 100)
    y_grid = np.linspace(-0.5, 1.5, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Forward pass on grid points
    Z1_grid = np.dot(grid_points, W1) + b1
    A1_grid = sigmoid(Z1_grid)
    Z2_grid = np.dot(A1_grid, W2) + b2
    A2_grid = sigmoid(Z2_grid)
    Z_pred = A2_grid.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(7, 6))
    contour = plt.contourf(xx, yy, Z_pred, levels=50, cmap="RdBu", alpha=0.8)
    plt.colorbar(contour, label="Model Prediction Probability")
    
    # Plot dataset
    for xi, yi in zip(X, y):
        color = "blue" if yi[0] == 1 else "red"
        marker = "X" if yi[0] == 1 else "o"
        plt.scatter(xi[0], xi[1], color=color, marker=marker, s=150, edgecolors='black', zorder=5)
        
    plt.title("XOR Problem Decoded Decision Boundary", fontweight='bold')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True, alpha=0.3)
    
    plot_path = "xor_boundary.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Visualization plot saved to '{plot_path}'")
    plt.close()
    print("Project 05 finished successfully.")

if __name__ == "__main__":
    main()
