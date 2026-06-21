import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Set seed for reproducibility
np.random.seed(42)

class Layer:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.inputs = None
        self.dW = None
        self.db = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.dot(inputs, self.W) + self.b

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        m = self.inputs.shape[0]
        self.dW = np.dot(self.inputs.T, output_gradient) / m
        self.db = np.sum(output_gradient, axis=0, keepdims=True) / m
        return np.dot(output_gradient, self.W.T)

class ReLU(Layer):
    def __init__(self):
        self.inputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * (self.inputs > 0)

class Sigmoid(Layer):
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-np.clip(inputs, -500, 500)))
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self.output * (1.0 - self.output)

class BCECost:
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return - (y_true / y_pred - (1 - y_true) / (1 - y_pred))

def main():
    print("Running Project 01: NN From Scratch (NumPy)...")
    
    # Generate simple separable synthetic binary dataset
    X = np.random.randn(150, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    # Model configuration
    layers: List[Layer] = [
        Dense(2, 4),
        ReLU(),
        Dense(4, 1),
        Sigmoid()
    ]
    
    cost_fn = BCECost()
    lr = 0.1
    
    # Simple training loop
    for epoch in range(100):
        # Forward pass
        out = X
        for layer in layers:
            out = layer.forward(out)
            
        loss = cost_fn.compute(y, out)
        
        # Backward pass
        grad = cost_fn.gradient(y, out)
        for layer in reversed(layers):
            grad = layer.backward(grad)
            
        # Update weights (SGD style)
        for layer in layers:
            if isinstance(layer, Dense):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db
                
        if (epoch + 1) % 20 == 0:
            preds = (out > 0.5).astype(float)
            acc = np.mean(preds == y)
            print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
            
    print("Project 01 finished successfully.")

if __name__ == "__main__":
    main()
