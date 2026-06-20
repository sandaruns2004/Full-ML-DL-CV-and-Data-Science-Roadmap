# 🎯 The Attention Mechanism

> **Prerequisites**: RNNs, Matrix Algebra | **Difficulty**: ⭐⭐⭐⭐⭐ Expert

---

## 📋 Table of Contents
1. [The Information Bottleneck in RNNs](#1-the-information-bottleneck-in-rnns)
2. [The Database Analogy (Query, Key, Value)](#2-the-database-analogy-query-key-value)
3. [Scaled Dot-Product Attention (The Math)](#3-scaled-dot-product-attention-the-math)
4. [Multi-Head Attention](#4-multi-head-attention)
5. [Implementation in PyTorch](#5-implementation-in-pytorch)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The Information Bottleneck in RNNs

In traditional Sequence Models (RNNs/LSTMs/Seq2Seq), processing language involves two major flaws:

1. **Sequential Processing**: To process the 100th word in a paragraph, the network must sequentially process words 1 through 99. This is incredibly slow and prevents the massive parallelization GPUs are designed for.
2. **The Information Bottleneck**: In a standard Seq2Seq translation model, an entire English paragraph must be compressed into a single, fixed-size mathematical vector (the hidden state) before the Decoder can start translating to French. If the paragraph is long, the network simply forgets the beginning of the text.

In 2017, Google researchers solved both problems in a paper called *"Attention Is All You Need"*. They proposed discarding RNNs entirely and replacing them with a purely mathematical operation called **Attention**, which allows the network to process every word simultaneously while "looking back" at specific, relevant words.

---

## 2. The Database Analogy (Query, Key, Value)

Attention mimics how a database works. 

Imagine you are searching YouTube (the database):
1. You type in a **Query (Q)**: `"Machine Learning Tutorial"`.
2. YouTube checks your Query against the **Keys (K)** (the titles/tags) of every video in its database.
3. YouTube calculates a similarity score between your Query and every Key.
4. It returns the **Values (V)** (the actual video content) of the records with the highest scores.

In the Attention mechanism, every single word in a sentence generates its own Query, its own Key, and its own Value! 
- If the word "bank" is processing its context in the sentence *"The bank of the river"*, it acts as the **Query**.
- It compares itself to the **Keys** of *"The"*, *"bank"*, *"of"*, *"the"*, *"river"*.
- It finds a high mathematical correlation with *"river"*, so it absorbs the **Value** (meaning) of *"river"* to understand that it is a river bank, not a financial institution.

---

## 3. Scaled Dot-Product Attention (The Math)

For each word token vector $X_i$, we multiply it by three learned weight matrices to generate the $Q$, $K$, and $V$ vectors:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}^K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}^V$$

Then, we apply the core Attention formula to all tokens simultaneously using matrix multiplication:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

Let's break this down:
1. **$\mathbf{Q} \mathbf{K}^T$**: Computes the dot product between every Query and every Key. The dot product measures similarity. High dot product = the words are highly related.
2. **$\sqrt{d_k}$**: A scaling factor ($d_k$ is the dimensionality of the Key vectors). As dimensions grow, dot products become massive. Massive numbers push the Softmax function into flat regions with zero gradients, stopping neural network training. We divide by $\sqrt{d_k}$ to keep the variance at $1$.
3. **$\text{Softmax}$**: Normalizes the scores so they sum to $1$. This creates the **Attention Matrix** (e.g., word $A$ pays 90% attention to word $B$, and 10% to word $C$).
4. **$\times \mathbf{V}$**: We multiply the Attention Matrix by the Values. The final representation for word $A$ is now a weighted mixture of the values of the words it paid attention to!

```mermaid
graph TD
    %% Inputs
    X[Input Tokens X] --> WQ[Weights W_Q]
    X --> WK[Weights W_K]
    X --> WV[Weights W_V]
    
    WQ --> Q[Queries Q]
    WK --> K[Keys K]
    WV --> V[Values V]
    
    %% Math Operations
    Q --> Dot[MatMul: Q * K^T]
    K --> Dot
    
    Dot --> Scale[Scale: / sqrt d_k]
    Scale --> Mask[Optional Masking]
    Mask --> Softmax[Softmax]
    
    Softmax --> AttnMat[Attention Matrix]
    
    AttnMat --> FinalDot[MatMul: Attn * V]
    V --> FinalDot
    
    FinalDot --> Output[Attention Output Context Vectors]
    
    classDef tensor fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classdef op fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    class Q,K,V,AttnMat,Output tensor;
    class Dot,Scale,Mask,Softmax,FinalDot op;
```

---

## 4. Multi-Head Attention

Why look at a sentence from only one perspective? 

Instead of performing a single large attention operation, **Multi-Head Attention** splits the $Q$, $K$, and $V$ vectors into $h$ different, smaller subspaces (heads). It performs the Scaled Dot-Product Attention independently in each head, and concatenates the results back together.

This allows the model to attend to information from different representation subspaces simultaneously:
- **Head 1**: Might specialize in grammatical syntax (verbs attending to their subjects).
- **Head 2**: Might specialize in semantic meaning (adjectives attending to the nouns they modify).
- **Head 3**: Might track pronouns (resolving that "it" refers to "the dog").

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O$$
where $\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$.

---

## 5. Implementation in PyTorch

Let's implement the core math of Scaled Dot-Product Attention from scratch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import seaborn as sns
import matplotlib.pyplot as plt

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        # Q, K, V shape: [batch_size, num_heads, seq_len, depth]
        d_k = Q.size(-1)
        
        # 1. Q * K^T
        # K.transpose(-2, -1) swaps the last two dimensions (seq_len and depth)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores shape: [batch_size, num_heads, seq_len, seq_len]
        
        # 2. Masking (Optional: Used in Decoders to prevent looking at future words)
        if mask is not None:
            # Fill the masked positions (where mask == 0) with -infinity
            # so Softmax turns them into 0 probabilities.
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 3. Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4. Multiply by V
        output = torch.matmul(attention_weights, V)
        # output shape: [batch_size, num_heads, seq_len, depth]
        
        return output, attention_weights

# --- Demonstration ---
seq_len = 4
d_k = 8
batch_size = 1
num_heads = 1

# Mock Q, K, V tensors (e.g., 4 words, each represented by a vector of size 8)
Q = torch.randn(batch_size, num_heads, seq_len, d_k)
K = torch.randn(batch_size, num_heads, seq_len, d_k)
V = torch.randn(batch_size, num_heads, seq_len, d_k)

attention = ScaledDotProductAttention()

# 1. Standard Self-Attention
output, weights = attention(Q, K, V)

print("Attention Matrix (Softmax Probabilities):")
print(weights[0, 0].detach().numpy())
```

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Visualize Attention**: Expand the PyTorch code block above. Create an artificial $4 \times 4$ Attention Matrix where the word "bank" heavily attends to "river". Use `seaborn.heatmap` to plot the matrix and visualize the exact probabilities. 

### What's Next
| Next | Why |
|------|-----|
| [Transformer Architecture](./02-Transformer-Architecture.md) | We understand the mathematical engine (Attention). Now let's see how it fits into the massive framework of the full Transformer Encoder-Decoder architecture. |

---

[← Temporal Convolutional Networks](../08-Sequence-Models/04-Temporal-Convolutional-Networks.md) | [Back to Index](../README.md) | [Next: Transformer Architecture →](./02-Transformer-Architecture.md)
