# 🧠 16 - Building A Neural Network From Scratch

---

## 📋 Table of Contents
1. [The Capstone Project](#the-capstone-project)
2. [Step 1: Initialization & Architecture](#step-1-initialization--architecture)
3. [Step 2: Forward Propagation](#step-2-forward-propagation)
4. [Step 3: Loss Function](#step-3-loss-function)
5. [Step 4: Backpropagation](#step-4-backpropagation)
6. [Step 5: The Training Loop](#step-5-the-training-loop)
7. [What's Next](#whats-next)

---

## 🎓 The Capstone Project

To truly understand how neural networks learn, you must build one without using deep learning frameworks like PyTorch or TensorFlow. 

In this capstone, we will build a 2-layer Neural Network (1 hidden layer, 1 output layer) using **pure Python and NumPy**. We will train it to solve a binary classification problem.

*(Note: The full, runnable code and dataset for this tutorial can be found in the [Neural Network From Scratch Notebook](./notebooks/06-Neural-Network-From-Scratch.ipynb)).*

---

## 🏗️ Step 1: Initialization & Architecture

First, we define our architecture. 
- Input Layer: $n\_x$ features.
- Hidden Layer: $n\_h$ neurons, using **ReLU** activation.
- Output Layer: $n\_y$ neurons (1 for binary classification), using **Sigmoid** activation.

Because we are using ReLU, we must use **He Initialization** for the hidden layer to prevent vanishing gradients.

```python
import numpy as np

def initialize_parameters(n_x, n_h, n_y):
    # He Initialization for Hidden Layer
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((n_h, 1))
    
    # Xavier Initialization for Output Layer (Sigmoid)
    W2 = np.random.randn(n_y, n_h) * np.sqrt(1.0 / n_h)
    b2 = np.zeros((n_y, 1))
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
```

---

## 🌊 Step 2: Forward Propagation

We take our input $X$ and push it through the network using Matrix Multiplication (`np.dot`). 

```python
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def forward_propagation(X, params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    
    # Layer 1: Linear Math -> ReLU
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    
    # Layer 2: Linear Math -> Sigmoid
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    # Save values we will need later for Backpropagation
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    
    return A2, cache
```

---

## 📉 Step 3: Loss Function

We use Binary Cross-Entropy (BCE) since this is a binary classification problem.

```python
def compute_loss(A2, Y):
    m = Y.shape[1] # Number of examples in the batch
    
    # Binary Cross Entropy Formula
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    loss = - (1.0 / m) * np.sum(logprobs)
    
    # Ensure loss is a single number, not an array
    return np.squeeze(loss) 
```

---

## 🕵️ Step 4: Backpropagation

This is the hardest part. We must apply the Calculus Chain Rule backward to find the gradients for $W_1$, $b_1$, $W_2$, and $b_2$.

```python
def relu_derivative(Z):
    return Z > 0 # Returns 1 if Z > 0, else 0

def backward_propagation(params, cache, X, Y):
    m = X.shape[1]
    
    W2 = params["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    
    # 1. Output Layer Gradients (Math simplifies beautifully for BCE + Sigmoid)
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # 2. Hidden Layer Gradients (Propagating blame backward)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1) # Chain rule through ReLU
    
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
```

---

## 🔄 Step 5: The Training Loop (Gradient Descent)

Finally, we tie it all together in a loop. We define our hyperparameters (learning rate and epochs), update our weights, and watch the network learn!

```python
def update_parameters(params, grads, learning_rate):
    # Subtract the gradient * learning_rate from the current weights
    params["W1"] = params["W1"] - learning_rate * grads["dW1"]
    params["b1"] = params["b1"] - learning_rate * grads["db1"]
    params["W2"] = params["W2"] - learning_rate * grads["dW2"]
    params["b2"] = params["b2"] - learning_rate * grads["db2"]
    
    return params

def build_and_train_network(X, Y, n_h, learning_rate, epochs):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    # Initialize
    params = initialize_parameters(n_x, n_h, n_y)
    
    # Loop
    for i in range(epochs):
        # Forward Pass
        A2, cache = forward_propagation(X, params)
        
        # Calculate Error
        loss = compute_loss(A2, Y)
        
        # Backward Pass (Calculus)
        grads = backward_propagation(params, cache, X, Y)
        
        # Optimize
        params = update_parameters(params, grads, learning_rate)
        
        # Print progress
        if i % 100 == 0:
            print(f"Epoch {i} | Loss: {loss:.4f}")
            
    return params
```

When you run this code on a dataset, you will see the loss start high, and slowly decrease as the math twists and folds the network's internal representations to separate the classes perfectly. 

**Congratulations! You have built a functioning artificial brain from scratch.**

---

## 🚀 What's Next

### Key Takeaways
- Deep Learning is not magic. It is just matrix multiplication, calculus, and an optimization loop.
- Building from scratch proves you truly understand the flow of gradients and dimensions.

### Practical Recommendations
- Now that you know exactly how the engine works, **you never need to write this from scratch again.** 
- In the real world, you will use PyTorch or TensorFlow. Those frameworks handle the Backpropagation calculus automatically using Autograd, and they execute the matrix math on GPUs exponentially faster than NumPy.

### Proceed to the Next Module
You have mastered the foundational math and mechanics of Neural Networks. It is time to apply this knowledge to real-world unstructured data. 

Proceed to **Module 07**, where we will restructure these simple dense layers into Convolutional Neural Networks (CNNs) to give our AI the power of sight.

[← Previous Topic](./15-Hyperparameter-Tuning.md) | [Return to Root Index](../../README.md)
