# 🧠 Mixture of Experts (MoE)

> **Prerequisites**: Core ML Concepts | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The MoE Architecture](#1-the-moe-architecture)
2. [The Routing Mechanism (Top-K Gating)](#2-the-routing-mechanism-top-k-gating)
3. [The Load Balancing Problem](#3-the-load-balancing-problem)
4. [PyTorch Implementation Sketch](#4-pytorch-implementation-sketch)
5. [Summary](#summary)

---

As Large Language Models (LLMs) scale up to trillions of parameters, training and inference become computationally prohibitive. A dense model activates every single parameter for every single token processed. 

**Mixture of Experts (MoE)** is a sparse architecture that allows the total number of parameters to increase massively, while keeping the compute cost (active parameters) roughly constant. It does this by activating only a small subset of the network for each token.

MoE is the architecture behind models like GPT-4, Mixtral 8x7B, and Google's Switch Transformer.

---

## 1. The MoE Architecture

In a standard Transformer block, the Feed-Forward Network (FFN) consists of two linear layers with a non-linear activation in between. 

In an MoE Transformer block, the single dense FFN is replaced by:
1.  **A set of $N$ "Expert" networks** (usually just standard FFNs).
2.  **A Routing Network (Gating Network)** that decides which tokens go to which experts.

Instead of all tokens passing through the same FFN, the router dynamically assigns each token to the best $K$ experts (where typically $K=1$ or $K=2$).

### Mathematical Formulation

Given an input token representation $x$ and $N$ expert networks $E_1(x), E_2(x), \dots, E_N(x)$, the output of the MoE layer is a weighted sum of the expert outputs:

$$ y = \sum_{i=1}^{N} G(x)_i E_i(x) $$

Where $G(x)_i$ is the routing score (gate value) assigned to expert $i$. 

Crucially, in a **Sparse MoE**, the gating network $G(x)$ is designed to output a sparse vector, where most elements are exactly zero. If $G(x)_i = 0$, then $E_i(x)$ does not need to be computed at all.

---

## 2. The Routing Mechanism (Top-K Gating)

How do we compute the sparse routing vector $G(x)$? The most common approach is **Top-K Gating**.

1.  **Linear Projection**: Multiply the input token $x$ by a learned weight matrix $W_g$ to get raw logits for each expert.
    $$ H(x) = x W_g $$
2.  **Add Noise (Optional but common during training)**: Add standard normal noise to encourage exploration of different experts.
    $$ H(x) = x W_g + \epsilon \cdot \text{Softplus}(x W_{noise}) $$
3.  **Top-K Selection**: Keep only the top $K$ values, and set the rest to $-\infty$. Let this operation be $\text{TopK}(v, k)$.
4.  **Softmax**: Apply the softmax function to normalize the chosen $K$ values to sum to 1. The $-\infty$ values become 0.

$$ G(x) = \text{Softmax}(\text{TopK}(H(x), K)) $$

### Why $K=2$?
While $K=1$ (Switch Routing) is the most efficient, $K=2$ (routing each token to the top two experts) is standard in models like Mixtral. Routing to at least 2 experts is critical because it allows the router network to receive meaningful gradients. If $K=1$, the router can't easily learn *how much better* the first choice was compared to the second.

---

## 3. The Load Balancing Problem

A naive MoE model will quickly suffer from **representation collapse**: the router might decide that Expert 1 is slightly better at everything. It will route all tokens to Expert 1.
*   Expert 1 gets all the gradients and becomes even better.
*   Experts 2 to $N$ receive no tokens, receive no gradients, and become "dead experts."

To prevent this, MoE models require an auxiliary loss function during training to force the router to distribute tokens evenly.

### Expert Capacity
During training, tokens are processed in batches. If we have $T$ tokens in a batch and $N$ experts, we ideally want each expert to receive $T/N$ tokens.

We define an **Expert Capacity** limit:
$$ \text{Capacity} = \frac{T \times K}{N} \times \text{CapacityFactor} $$

If an expert receives more tokens than its capacity, the excess tokens are **dropped** (their representation just passes through the residual connection without FFN processing). Dropped tokens hurt performance, so the model learns to avoid overflowing experts.

### Load Balancing Loss
We add a penalty to the main loss function. Let $f_i$ be the fraction of tokens routed to expert $i$, and $P_i$ be the average gating probability assigned to expert $i$ across the batch. 

The load balancing loss is:
$$ L_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i $$

This loss is minimized when the router uniformly distributes both the hard assignments ($f_i$) and the soft probabilities ($P_i$) across all $N$ experts.

---

## 4. PyTorch Implementation Sketch

Here is a simplified conceptual implementation of a Top-2 MoE layer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Standard Transformer FFN
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # The router/gate network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # The experts
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # [batch_size * seq_len, d_model]
        
        # 1. Get routing logits
        router_logits = self.gate(x_flat) # [num_tokens, num_experts]
        
        # 2. Get top-k experts
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        # Normalize weights to sum to 1
        routing_weights = F.softmax(routing_weights, dim=-1) 
        
        # 3. Compute expert outputs
        final_output = torch.zeros_like(x_flat)
        
        # In a real implementation, this loop is highly optimized or replaced 
        # with scatter/gather operations. Here we loop for clarity.
        for i, expert in enumerate(self.experts):
            # Find which tokens were routed to this expert
            expert_mask = (selected_experts == i)
            
            # If no tokens were routed to this expert, skip
            if not expert_mask.any():
                continue
                
            # Get the indices of the tokens routed to this expert
            token_indices, k_indices = torch.where(expert_mask)
            
            # Extract the actual tokens
            tokens_for_expert = x_flat[token_indices]
            
            # Extract their routing weights
            weights_for_expert = routing_weights[token_indices, k_indices].unsqueeze(-1)
            
            # Pass through expert and multiply by weight
            expert_out = expert(tokens_for_expert) * weights_for_expert
            
            # Add to final output
            final_output.index_add_(0, token_indices, expert_out)
            
        return final_output.view(batch_size, seq_len, d_model)
```

MoE achieves the seemingly impossible: decoupling model parameter count from inference compute cost. A model like Mixtral 8x7B has 47 billion total parameters but only activates ~13 billion parameters per token, allowing it to run as fast as a 13B model while achieving the performance of a much larger one.

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Implement the core concepts from scratch using NumPy.
- 🟡 **Intermediate**: Use scikit-learn or PyTorch to build a robust pipeline applying these methods.

### What's Next
| Next | Why |
|------|-----|
| [Multi-Modal Models](./08-Multi-Modal-Models.md) | Learn about combining Vision and Language in LLMs. |

---

[← LLM Fine Tuning And RLHF](./06-LLM-Fine-Tuning-And-RLHF.md) | [Back to Index](../README.md) | [Next: Multi Modal Models →](./08-Multi-Modal-Models.md)
