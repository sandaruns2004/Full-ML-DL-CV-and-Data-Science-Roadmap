# ⏱️ Temporal Convolutional Networks (TCNs)

> **Prerequisites**: Convolutional Neural Networks (CNNs), RNNs | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Why not use RNNs for everything?](#1-why-not-use-rnns-for-everything)
2. [1D Convolutions for Time Series](#2-1d-convolutions-for-time-series)
3. [Causal Convolutions (No Peeking!)](#3-causal-convolutions-no-peeking)
4. [Dilated Convolutions (Massive Receptive Fields)](#4-dilated-convolutions-massive-receptive-fields)
5. [The TCN Architecture](#5-the-tcn-architecture)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. Why not use RNNs for everything?

Recurrent Neural Networks (LSTMs, GRUs) are the theoretical gold standard for sequences because they have infinite memory (mathematically). 

However, in practice, they have huge engineering drawbacks:
1. **Inherently Sequential**: To compute the hidden state at $t=100$, you *must* wait for the computations at $t=1 \dots 99$ to finish. You cannot parallelize this across a GPU. Training is incredibly slow.
2. **Exploding/Vanishing Gradients**: Even with LSTM gates, extremely long sequences (e.g., audio waves with 16,000 samples per second) completely break backpropagation through time.

**The Question**: Can we process sequential data using CNNs? Yes, using **Temporal Convolutional Networks (TCNs)**.

---

## 2. 1D Convolutions for Time Series

In image processing, we use 2D Convolutions (sliding a square filter over an image's height and width).
For time series or text, we use **1D Convolutions**. We slide a 1D filter (e.g., size 3) purely across the *time* axis.

If we have an input sequence of length $L$ and $C_{in}$ channels (e.g., word embeddings of size 100, or 3 sensors measuring temperature/pressure/humidity), a 1D Convolution applies a filter of shape `(kernel_size, C_in, C_out)`.

Because convolutions can be applied to all time steps simultaneously, **1D CNNs can be massively parallelized on GPUs**, training up to 10x faster than LSTMs.

---

## 3. Causal Convolutions (No Peeking!)

Standard 1D CNNs have a "cheating" problem when applied to forecasting.

If we use a standard Convolution with `kernel_size=3` and padding to maintain sequence length, the output at time $t$ is calculated using inputs $x_{t-1}, x_t,$ and **$x_{t+1}$**.
If you are trying to predict the stock market at time $t$, your network is mathematically cheating by looking one step into the future!

**Causal Convolutions** fix this by shifting the padding. We only pad the *left* side of the input sequence. This ensures that the output at time $t$ is calculated using *only* inputs from time $t$ and earlier: $x_{t-2}, x_{t-1}, x_t$. 

Information can only flow forward in time.

---

## 4. Dilated Convolutions (Massive Receptive Fields)

If we use a causal convolution with `kernel_size=3`, the network can only look back 2 time steps into the past. 
To look back 100 time steps, we would need 50 layers of convolutions! This is computationally awful.

**Dilated Convolutions** solve this by injecting holes (zeros) into the convolutional filter.
- Dilation = 1: Standard convolution (looks at $t, t-1, t-2$)
- Dilation = 2: Skips 1 step (looks at $t, t-2, t-4$)
- Dilation = 4: Skips 3 steps (looks at $t, t-4, t-8$)

By exponentially increasing the dilation factor at each layer (1, 2, 4, 8, 16...), the **receptive field** of the network grows exponentially! With just 10 layers, the top node can look back at thousands of time steps, requiring very few parameters and remaining fully parallelizable.

This specific architecture was popularized by DeepMind's **WaveNet** (2016) for generating highly realistic human speech audio.

---

## 5. The TCN Architecture

A modern TCN consists of:
1. **1D Causal Convolutions**: No leaking of future data.
2. **Exponentially Dilated Receptive Fields**: To capture very long-term dependencies.
3. **Residual Blocks**: Like ResNet, adding the input of the block directly to the output. This ensures gradients don't vanish through the deep layers.
4. **Weight Normalization & Dropout**: For aggressive regularization.

### PyTorch Implementation of a Causal Dilated Conv1d

PyTorch's standard `nn.Conv1d` does not support causal padding natively. We have to pad it manually and then slice off the excess right-side outputs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        
        # Calculate how much we need to pad the left side
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=self.padding, 
            dilation=dilation
        )

    def forward(self, x):
        # x shape: [batch_size, in_channels, seq_len]
        out = self.conv(x)
        
        # The conv operation added padding to both sides (PyTorch default behavior for padding).
        # We slice off the right side to keep it strictly causal!
        # If input length was L, output length is L + padding. We only want the first L elements.
        return out[:, :, :-self.padding]

# Example Usage
batch_size = 32
channels = 10     # e.g., 10 different stock indicators
seq_length = 100  # 100 days of history

x = torch.randn(batch_size, channels, seq_length)

# Dilation 1
layer1 = CausalConv1d(in_channels=10, out_channels=32, kernel_size=3, dilation=1)
out1 = layer1(x)
print(f"Layer 1 output shape: {out1.shape}") # Should be [32, 32, 100]

# Dilation 2
layer2 = CausalConv1d(in_channels=32, out_channels=64, kernel_size=3, dilation=2)
out2 = layer2(out1)
print(f"Layer 2 output shape: {out2.shape}") # Should be [32, 64, 100]
```

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Audio Generation / WaveNet**: Download a dataset of simple piano notes. Train a TCN to predict the next audio sample given the previous samples. After training, feed it a random noise seed and have it autoregressively generate 5 seconds of piano music! Compare the training speed against an LSTM.

### What's Next
| Next | Why |
|------|-----|
| [Attention Mechanism](../09-Transformers/01-Attention-Mechanism.md) | We have mastered RNNs and TCNs. Now we enter the modern era: Attention mechanisms and Transformers! |

---

[← Seq2Seq And Attention](./03-Seq2Seq-And-Attention.md) | [Back to Index](../README.md) | [Next: Attention Mechanism →](../09-Transformers/01-Attention-Mechanism.md)
