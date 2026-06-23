# 🧠 06 - Forward Propagation

---

## 📋 Table of Contents
1. [The Big Picture](#the-big-picture)
2. [A Simple Network Topology](#a-simple-network-topology)
3. [Step-by-Step Calculation (The Math)](#step-by-step-calculation-the-math)
4. [The Matrix Advantage (Vectorization)](#the-matrix-advantage-vectorization)
5. [What's Next](#whats-next)

---

## 🌊 The Big Picture

**Forward Propagation** (or the Forward Pass) is simply the process of pushing data through the neural network to get a prediction.

Information flows in one direction:
`Input Data ➡️ Hidden Layers ➡️ Output Layer ➡️ Prediction`

During this pass, no learning happens. The network simply uses its current weights and biases to make the best guess it can. 

---

## 🕸️ A Simple Network Topology

To understand forward propagation, we will trace the math through a very small, concrete example.

Let's build a network that predicts whether a student will pass an exam (`1`) or fail (`0`) based on two inputs:
- $x_1$: Hours studied.
- $x_2$: Hours slept.

Our network has:
- **Input Layer:** 2 nodes ($x_1$, $x_2$)
- **Hidden Layer:** 2 neurons with ReLU activation.
- **Output Layer:** 1 neuron with Sigmoid activation (to output a probability between 0 and 1).

```mermaid
graph LR
    subgraph Input
        X1((x₁: 4 hrs))
        X2((x₂: 8 hrs))
    end

    subgraph Hidden Layer (ReLU)
        H1((h₁))
        H2((h₂))
    end

    subgraph Output Layer (Sigmoid)
        Y((ŷ))
    end

    X1 -- w₁₁=0.5 --> H1
    X1 -- w₁₂=-0.2 --> H2
    
    X2 -- w₂₁=0.1 --> H1
    X2 -- w₂₂=0.4 --> H2
    
    H1 -- v₁=0.8 --> Y
    H2 -- v₂=0.6 --> Y
    
    classDef input fill:#3b82f6,color:#fff,stroke:#1e40af;
    classDef hidden fill:#8b5cf6,color:#fff,stroke:#5b21b6;
    classDef output fill:#10b981,color:#fff,stroke:#047857;
    class X1,X2 input;
    class H1,H2 hidden;
    class Y output;
```

---

## 🧮 Step-by-Step Calculation (The Math)

Let's assume our current weights are set to the random numbers shown in the diagram above. Let's also assume biases are `0` to keep the math simple.

**The Student:** Studied 4 hours ($x_1=4$), Slept 8 hours ($x_2=8$).

### Step 1: Calculate Hidden Layer Sums ($z^{[1]}$)

We calculate the linear combination for each neuron in the hidden layer.

**For Neuron $h_1$:**
$z_1^{[1]} = (x_1 \times w_{11}) + (x_2 \times w_{21}) + b_1$
$z_1^{[1]} = (4 \times 0.5) + (8 \times 0.1) + 0$
$z_1^{[1]} = 2.0 + 0.8 = 2.8$

**For Neuron $h_2$:**
$z_2^{[1]} = (x_1 \times w_{12}) + (x_2 \times w_{22}) + b_2$
$z_2^{[1]} = (4 \times -0.2) + (8 \times 0.4) + 0$
$z_2^{[1]} = -0.8 + 3.2 = 2.4$

### Step 2: Apply Hidden Layer Activations ($a^{[1]}$)

We apply the ReLU activation function $f(z) = \max(0, z)$ to both sums.

**For Neuron $h_1$:**
$a_1^{[1]} = \text{ReLU}(2.8) = 2.8$

**For Neuron $h_2$:**
$a_2^{[1]} = \text{ReLU}(2.4) = 2.4$

The hidden layer's final output is `[2.8, 2.4]`.

### Step 3: Calculate Output Layer Sum ($z^{[2]}$)

Now, the output of the hidden layer becomes the *input* for the output layer. We multiply by the next set of weights ($v$).

$z^{[2]} = (a_1^{[1]} \times v_1) + (a_2^{[1]} \times v_2) + b_{out}$
$z^{[2]} = (2.8 \times 0.8) + (2.4 \times 0.6) + 0$
$z^{[2]} = 2.24 + 1.44 = 3.68$

### Step 4: Apply Output Layer Activation ($a^{[2]}$)

Since this is a binary classification problem, we use the Sigmoid activation function to squish `3.68` into a probability.

$\hat{y} = \sigma(z^{[2]}) = \frac{1}{1 + e^{-3.68}}$
$\hat{y} \approx 0.975$

### The Result
The network predicts `0.975` (a 97.5% probability). It confidently predicts the student will pass the exam!

---

## ⚡ The Matrix Advantage (Vectorization)

Calculating the sums neuron-by-neuron using standard arithmetic is fine for 3 neurons. But a modern network like ChatGPT has *hundreds of billions* of neurons. 

If we used `for-loops` to calculate this, forward propagation would take years for a single image or text snippet.

Instead, we use **Linear Algebra (Matrices)**. Modern GPUs are designed to multiply massive grids of numbers simultaneously.

The entire hidden layer calculation we did above can be written as a single matrix multiplication:

$$ Z^{[1]} = W^{[1]} \cdot X + B^{[1]} $$
$$ A^{[1]} = \text{ReLU}(Z^{[1]}) $$

Where:
- $X$ is a vector of inputs `[4, 8]`
- $W^{[1]}$ is a matrix containing all the weights: $\begin{bmatrix} 0.5 & 0.1 \\ -0.2 & 0.4 \end{bmatrix}$

By writing code in NumPy or PyTorch, you execute the math using matrices. A GPU can compute $W^{[1]} \cdot X$ for millions of neurons in milliseconds. This concept is called **Vectorization**.

---

## 🚀 What's Next

### Key Takeaways
- Forward propagation is simply calculating the network's prediction from input to output.
- The output of layer $L$ becomes the input for layer $L+1$.
- We use Matrix Multiplication (Vectorization) to perform these calculations on millions of neurons simultaneously using GPUs.

### Common Mistakes
- **Matrix Dimension Mismatches:** When building neural networks from scratch, 90% of your bugs will be matrices that don't align. If layer 1 outputs a `[1 x 64]` matrix, the weight matrix of layer 2 must have `64` rows.

### Practical Recommendations
- When debugging a neural network model in PyTorch or TensorFlow, always print the `shape` of your tensors at every layer during forward propagation. 

### Next Topic
Our network predicted `0.975`. But what if the student actually failed? The network made a mistake. How do we quantify how wrong the network was? We use a Loss Function.

[← Previous Topic](./05-Activation-Functions.md) | [Next Topic: Loss Functions →](./07-Loss-Functions.md)
