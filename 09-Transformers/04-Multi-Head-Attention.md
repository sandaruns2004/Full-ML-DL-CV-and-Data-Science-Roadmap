# 04 - Multi-Head Attention

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Prerequisites**: 03-Self-Attention | **Estimated Reading Time**: 15 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition: The Panel of Experts](#2-intuition-the-panel-of-experts)
3. [Algorithm Workflow](#3-algorithm-workflow)
4. [Mathematics of Multi-Head Attention](#4-mathematics-of-multi-head-attention)
5. [Visualizing Multiple Heads](#5-visualizing-multiple-heads)
6. [Library Implementation (PyTorch)](#6-library-implementation-pytorch)
7. [Key Takeaways](#7-key-takeaways)
8. [Next Topic](#8-next-topic)

---

# 1. What Problem Does This Solve?

In the previous lesson, we learned that Self-Attention allows a word to look at other words to understand its context. 

### 🟢 Beginner
If the sentence is *"The angry dog barked loudly"*, the word "dog" needs to pay attention to "angry" (its mood) and "barked" (its action). 
If we only have one Attention mechanism (one "Head"), it is mathematically forced to split its attention: maybe 50% to "angry" and 50% to "barked". But what if the sentence is 50 words long? A single Head gets overwhelmed trying to track grammar, emotion, subject-verb agreement, and vocabulary all at the same time.

### 🟡 Intermediate
A single $W^Q, W^K, W^V$ projection matrix forces the network to learn a single representation of similarity. But language is multi-dimensional. Two words can be grammatically related (Subject $\to$ Verb), but semantically opposed (Hero $\to$ Villain). We need a mechanism that allows the network to calculate *multiple independent attention matrices* simultaneously.

### 🔴 Advanced
The Transformer solves this using **Multi-Head Attention**. It splits the embedding dimension $d_{model}$ into $h$ smaller chunks (Heads). It projects the input into $h$ different $Q, K, V$ spaces in parallel. After calculating $h$ independent Context Vectors, it concatenates them back together. This provides the model with multiple "representation subspaces", allowing it to attend to different linguistic phenomena at different positions simultaneously.

---

# 2. Intuition: The Panel of Experts

Imagine you are a CEO (the Transformer) trying to analyze a complex legal contract (the input sentence).

If you give the contract to just one general manager (Single-Head Attention), they will try to look for financial risks, grammatical errors, and legal loopholes all by themselves. They will likely miss something important.

Instead, you hire a **Panel of 8 Experts** (Multi-Head Attention):
- Expert 1 looks *strictly* for financial numbers.
- Expert 2 looks *strictly* at the legal jargon.
- Expert 3 checks the dates.
- Expert 4 checks the grammar.

Every expert reads the exact same contract. But because they have different training ($W_i^Q, W_i^K, W_i^V$ matrices), they "pay attention" to completely different words. 

Finally, they all put their individual reports together (Concatenation) and hand you a single, incredibly detailed summary.

---

# 3. Algorithm Workflow

```mermaid
flowchart TD
    Input[Input Sequence X] --> Split
    
    subgraph Multi-Head Attention (e.g., 3 Heads)
    Split --> H1[Head 1: W_q1, W_k1, W_v1]
    Split --> H2[Head 2: W_q2, W_k2, W_v2]
    Split --> H3[Head 3: W_q3, W_k3, W_v3]
    
    H1 --> SDPA1[Scaled Dot-Product Attention]
    H2 --> SDPA2[Scaled Dot-Product Attention]
    H3 --> SDPA3[Scaled Dot-Product Attention]
    end
    
    SDPA1 --> Concat
    SDPA2 --> Concat
    SDPA3 --> Concat
    
    Concat[Concatenate all Head Outputs] --> Linear[Final Linear Projection W_O]
    Linear --> Output[Final Context Vector]
```

---

# 4. Mathematics of Multi-Head Attention

Let the embedding dimension be $d_{model} = 512$.
Let the number of heads be $h = 8$.

Instead of doing one massive attention calculation with 512 dimensions, we split it into 8 parallel calculations of dimension $d_k = \frac{512}{8} = 64$.

For each head $i = 1, \dots, h$:
$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

Then, we concatenate the outputs of all 8 heads:
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

The final matrix $W^O$ (Output Projection) mixes the information from all 8 heads back together into a final vector of size 512.

*Note: Because the dimensions are split ($512 \to 8 \times 64$), the computational cost of Multi-Head Attention is exactly the same as Single-Head Attention! We get 8 experts for the price of 1.*

---

# 5. Visualizing Multiple Heads

If we visualize the attention weights of different heads for the sentence: *"The dog ran because it was scared"*, we will see specialization:

*   **Head 1 (Pronoun Resolution):** When processing "it", this head pays 99% attention to "dog".
*   **Head 2 (Subject-Verb):** When processing "ran", this head pays 90% attention to "dog".
*   **Head 3 (Causality):** When processing "because", this head pays attention to "scared".

Without Multi-Head Attention, a single head would have to average all these weights, washing out the strong, specific signals.

---

# 6. Library Implementation (PyTorch)

PyTorch provides a highly optimized implementation of this.

```python
import torch
import torch.nn as nn

# Configuration
batch_size = 2
seq_len = 10
embed_dim = 512 # Standard Transformer embedding size
num_heads = 8   # 512 / 8 = 64 dimensions per head

# Dummy input sequence (e.g., 10 words)
x = torch.randn(batch_size, seq_len, embed_dim)

# Initialize the PyTorch Multi-Head Attention Layer
# batch_first=True tells PyTorch our data is [Batch, Seq, Features]
mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

# Because this is Self-Attention, Query, Key, and Value are all the exact same input!
attn_output, attn_weights = mha(query=x, key=x, value=x)

print(f"Input Shape:  {x.shape}")           # [2, 10, 512]
print(f"Output Shape: {attn_output.shape}") # [2, 10, 512] -> Shape is preserved!
```

---

# 7. Key Takeaways

*   **Multi-Head Attention** solves the problem of a single attention head getting overwhelmed by complex sentences.
*   It splits the embedding dimension into $h$ smaller heads, allowing the network to look at $h$ different "representation subspaces" simultaneously (e.g., one head for grammar, one for vocabulary).
*   The outputs of all heads are concatenated and multiplied by a final output matrix $W^O$.
*   Because the math is parallelized and chunked, $h$ heads cost the same computationally as $1$ massive head.

---

# 8. Next Topic

We now have an AI that can look at every word in a sentence simultaneously and figure out the complex grammatical relationships between them. 

But there is a massive flaw. Because the matrix multiplications happen to all words *at the exact same time*, the AI has no idea what order the words are in! *"Dog bites man"* and *"Man bites dog"* yield the exact same mathematical result. 

We need a way to inject time back into the network without using RNN loops.

[← Self-Attention](03-Self-Attention.md) | [Back to Index](README.md) | [Next Topic: Positional Encoding →](05-Positional-Encoding.md)
