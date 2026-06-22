# 🔄 RNN Fundamentals & Backpropagation Through Time

> **Prerequisites**: Backpropagation, Matrix Math | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Problem with Feedforward Networks](#1-the-problem-with-feedforward-networks)
2. [Types of Sequence Models](#2-types-of-sequence-models)
3. [Recurrent Neural Networks (RNN) Math](#3-recurrent-neural-networks-rnn-math)
4. [Backpropagation Through Time (BPTT)](#4-backpropagation-through-time-bptt)
5. [The Vanishing Gradient Problem in RNNs](#5-the-vanishing-gradient-problem-in-rnns)
6. [PyTorch RNN Implementation](#6-pytorch-rnn-implementation)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Problem with Feedforward Networks

Standard MLPs and CNNs have two massive limitations:
1. They require a fixed input size (e.g., $256 \times 256$ images) and produce a fixed output size.
2. They process data **independently**. If you feed the word "Apple" into an MLP, it doesn't remember the previous word "The".

**Sequential Data** (Text, Audio, Time Series, Video) requires a network that can handle variable-length inputs and maintain **memory** (state) across time steps.

---

## 2. Types of Sequence Models

RNNs are highly flexible and can be mapped to different problem types:

1. **One-to-Many**: Image Captioning (Input: 1 Image, Output: Sequence of words).
2. **Many-to-One**: Sentiment Analysis (Input: Sequence of words, Output: 1 Sentiment Score).
3. **Many-to-Many (Aligned)**: Video Classification on frame-level (Input: Sequence of frames, Output: Sequence of labels).
4. **Many-to-Many (Unaligned / Seq2Seq)**: Machine Translation (Input: French sentence, Output: English sentence). The encoder processes the whole input before the decoder generates the output.

---

## 3. Recurrent Neural Networks (RNN) Math

An RNN solves the memory problem by adding a loop. It takes the input $x^{\langle t \rangle}$ at time step $t$, and the **hidden state** from the previous time step $h^{\langle t-1 \rangle}$, to compute the new hidden state and the output.

**Mathematical Formulation:**
For a given time step $t$:
$$h^{\langle t \rangle} = \tanh(\mathbf{W}_{hh} h^{\langle t-1 \rangle} + \mathbf{W}_{hx} x^{\langle t \rangle} + \mathbf{b}_h)$$
$$\hat{y}^{\langle t \rangle} = \text{Softmax}(\mathbf{W}_{yh} h^{\langle t \rangle} + \mathbf{b}_y)$$

*(Note: The initial hidden state $h^{\langle 0 \rangle}$ is typically initialized to a vector of zeros).*

Notice that the exact same weight matrices ($\mathbf{W}_{hh}, \mathbf{W}_{hx}, \mathbf{W}_{yh}$) are used across *all* time steps. This is **parameter sharing** across time, drastically reducing the number of parameters the network needs to learn, and allowing it to generalize to sequences of unseen lengths.

---

## 4. Backpropagation Through Time (BPTT)

To train an RNN, we conceptually "unroll" it through time. An RNN trained on a sequence of 10 words is mathematically equivalent to a 10-layer feedforward network, where all 10 layers share the exact same weights.

The total loss across the sequence is the sum of the losses at each time step:
$$\mathcal{L} = \sum_{t=1}^{T_y} \mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle})$$

To update the recurrent weights $\mathbf{W}_{hh}$, we must use the chain rule backward through time. Since $\mathbf{W}_{hh}$ is used at every step, its total gradient is the sum of gradients at each step:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}^{\langle t \rangle}}{\partial \mathbf{W}_{hh}}$$

This sum requires calculating the derivative of the hidden state at time $t$ with respect to the hidden state at a previous time $k$ ($k < t$). By the chain rule, this is a product of Jacobians:
$$\frac{\partial h^{\langle t \rangle}}{\partial h^{\langle k \rangle}} = \prod_{i=k+1}^{t} \frac{\partial h^{\langle i \rangle}}{\partial h^{\langle i-1 \rangle}}$$

Since $h^{\langle i \rangle} = \tanh(\mathbf{W}_{hh} h^{\langle i-1 \rangle} + \dots)$, the local derivative is:
$$\frac{\partial h^{\langle i \rangle}}{\partial h^{\langle i-1 \rangle}} = \mathbf{W}_{hh}^T \cdot \text{diag}(1 - \tanh^2(...))$$

---

## 5. The Vanishing Gradient Problem in RNNs

Look closely at the unrolled chain rule product:
$$\prod_{i=k+1}^{t} \mathbf{W}_{hh}^T$$

If we unroll the RNN for 100 time steps (a 100-word sentence), we are multiplying the matrix $\mathbf{W}_{hh}$ by itself 100 times.

- **Vanishing Gradients**: If the largest eigenvalue of $\mathbf{W}_{hh}$ is strictly $< 1$, multiplying it by itself 100 times causes the gradient to shrink exponentially to **0**. The network cannot update weights based on errors made 100 steps ago. It effectively suffers from "short-term memory loss" and cannot learn long-term dependencies (e.g., matching a pronoun on page 2 to a noun on page 1).
- **Exploding Gradients**: If the largest eigenvalue of $\mathbf{W}_{hh}$ is $> 1$, the gradient exponentially explodes to **Infinity (NaN)**. This causes the optimizer to take massive, destructive steps.
  - *Fix for Exploding Gradients*: **Gradient Clipping**. If the norm of the gradient exceeds a threshold, we simply scale it down before applying the optimizer step.

Because of the Vanishing Gradient problem, Vanilla RNNs are almost never used in practice for long sequences.

---

## 6. PyTorch RNN Implementation

To solidify the math, let's implement a standard Vanilla RNN Cell and the unrolling loop from scratch using PyTorch.

```python
import torch
import torch.nn as nn
import math

class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Learnable parameters
        self.W_hx = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)
        
    def forward(self, x_t, h_prev):
        # x_t shape: (batch_size, input_size)
        # h_prev shape: (batch_size, hidden_size)
        
        # Calculate new hidden state using the math formula:
        # h_t = tanh(W_hh * h_prev + W_hx * x_t + b)
        h_t = torch.tanh(self.W_hh(h_prev) + self.W_hx(x_t))
        return h_t

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = VanillaRNNCell(input_size, hidden_size)
        
        # Output layer mapping hidden state to prediction
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden state with zeros
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # We will collect all outputs
        outputs = []
        
        # Unroll the RNN through time
        for t in range(seq_length):
            x_t = x[:, t, :] # Extract the input for the current time step
            h_t = self.rnn_cell(x_t, h_t)
            
            # Compute output prediction for current time step
            y_hat_t = self.fc_out(h_t)
            outputs.append(y_hat_t.unsqueeze(1))
            
        # Concatenate outputs along the time dimension
        # Resulting shape: (batch_size, seq_length, output_size)
        return torch.cat(outputs, dim=1), h_t

# --- Example Usage ---
batch_size = 3
seq_length = 5
input_size = 10
hidden_size = 20
output_size = 2

# Create the model
model = VanillaRNN(input_size, hidden_size, output_size)

# Create some dummy sequential data
dummy_sequence = torch.randn(batch_size, seq_length, input_size)

# Forward pass
predictions, final_hidden_state = model(dummy_sequence)

print(f"Input Shape: {dummy_sequence.shape} -> (Batch, Seq_Len, Input_Size)")
print(f"Predictions Shape: {predictions.shape} -> (Batch, Seq_Len, Output_Size)")
print(f"Final Hidden State Shape: {final_hidden_state.shape} -> (Batch, Hidden_Size)")
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Manual BPTT**: Build a simple Vanilla RNN in pure NumPy for a 3-step sequence. Implement the forward pass and carefully write the Backpropagation Through Time algorithm to calculate the gradients manually. Compare your gradients against a PyTorch implementation.

### What's Next
| Next | Why |
|------|-----|
| [LSTMs & GRUs](./02-LSTM-And-GRU.md) | How do we solve the Vanishing Gradient problem? We add a "conveyor belt" of memory called the Cell State. |

---

[← CNN from Scratch (PyTorch)](../07-Computer-Vision-CNNs/16-CNN-From-Scratch-PyTorch.md) | [Back to Index](../README.md) | [Next: Long Short-Term Memory (LSTM) & GRUs →](02-LSTM-And-GRU.md)
