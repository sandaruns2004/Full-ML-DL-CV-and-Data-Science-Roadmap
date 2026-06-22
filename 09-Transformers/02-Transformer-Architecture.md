# 🏗️ The Transformer Architecture

> **Prerequisites**: Attention Mechanism | **Difficulty**: ⭐⭐⭐⭐⭐ Expert

---

## 📋 Table of Contents
1. [The Order Problem: Positional Encoding](#1-the-order-problem-positional-encoding)
2. [Multi-Head Self-Attention (MHSA)](#2-multi-head-self-attention-mhsa)
3. [The Feed Forward Network (FFN)](#3-the-feed-forward-network-ffn)
4. [The Encoder Block](#4-the-encoder-block)
5. [The Decoder Block & Masking](#5-the-decoder-block--masking)
6. [The Full Transformer Architecture](#6-the-full-transformer-architecture)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Order Problem: Positional Encoding

Because Attention computes the dot product between all words simultaneously, the Transformer has **no inherent concept of sequence order**. To a raw Attention matrix, the sentence *"Dog bites man"* is mathematically identical to *"Man bites dog"*.

To fix this, we must inject positional information into the input embeddings before they enter the network. This is called **Positional Encoding**.

The original 2017 paper used interwoven Sine and Cosine waves of different frequencies:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- $pos$: The position of the word in the sentence (0, 1, 2...).
- $i$: The dimension index of the embedding.

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        # Register as buffer (won't be updated by optimizer)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Add positional encoding to embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x
```

---

## 2. Multi-Head Self-Attention (MHSA)

Instead of performing a single attention function, the Transformer projects the queries, keys, and values $h$ times (multiple heads). This allows the model to jointly attend to information from different representation subspaces at different positions.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear projections and split into heads: (B, H, S, d_k)
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention on all heads
        out = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. Concatenate heads and apply final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out)
```

---

## 3. The Feed Forward Network (FFN)

After the attention layer, each token is passed through an identical, independent 2-layer MLP. This provides non-linearity to process the gathered contextual information.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
```

---

## 4. The Encoder Block

The Encoder block combines MHSA and FFN. Crucially, it surrounds each sub-layer with a **Residual Connection** and **Layer Normalization**.
- *Residual Connection*: $Output = x + \text{Sublayer}(x)$. Prevents vanishing gradients.
- *Layer Normalization*: Stabilizes training across the feature dimension.

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Self-Attention sub-layer
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-Forward sub-layer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
```

---

## 5. The Decoder Block & Masking

The Decoder generates output autoregressively (one word at a time). It has two main differences from the Encoder:

### 1. Masked Self-Attention
When generating word 4, the Decoder cannot look at future words (5, 6...). We apply a triangular mask to the self-attention matrix to set future positions to $-\infty$ before Softmax.

### 2. Cross-Attention
The Decoder has an extra attention layer where it looks back at the Encoder's output.
- **Queries ($Q$)**: Come from the *Decoder*.
- **Keys ($K$) & Values ($V$)**: Come from the *Encoder*.

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 1. Masked Self-Attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. Cross-Attention
        attn_out = self.cross_attn(q=x, k=enc_out, v=enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        
        # 3. Feed-Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x
```

---

## 6. The Full Transformer Architecture

We can now stack $N$ Encoders and $N$ Decoders to build the full model.

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        
        self.encoders = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, src, src_mask):
        x = self.dropout(self.pe(self.src_embedding(src)))
        for encoder in self.encoders:
            x = encoder(x, src_mask)
        return x
        
    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.dropout(self.pe(self.tgt_embedding(tgt)))
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask, tgt_mask)
        return x
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        out = self.fc_out(dec_out)
        return out

# --- Example Forward Pass ---
src_vocab = 1000
tgt_vocab = 1000
d_model = 512

model = Transformer(src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab, d_model=d_model)

# Dummy inputs (batch_size=2, seq_len=10)
src = torch.randint(0, src_vocab, (2, 10))
tgt = torch.randint(0, tgt_vocab, (2, 10))

# Forward pass
output = model(src, tgt)
print("Output shape:", output.shape) # Expected: (2, 10, 1000) -> (Batch, Seq_Len, Vocab_Size)
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Modify the `FeedForward` layer to use a GELU activation function instead of ReLU.
- 🟡 **Intermediate**: Implement the exact `tgt_mask` (a lower triangular matrix of 1s and 0s) to pass into the Transformer's forward function.
- 🔴 **Advanced**: Train a miniature version of this from-scratch Transformer on a simple dataset (e.g., reversing a sequence of numbers).

### What's Next
| Next | Why |
|------|-----|
| [BERT & Encoder Models](./03-BERT-And-Encoder-Models.md) | What happens if we take a Transformer, throw away the Decoder, and train the Encoder on the entire internet? You get BERT. |

---

[← The Attention Mechanism](01-Attention-Mechanism.md) | [Back to Index](../README.md) | [Next: BERT & Encoder-Only Models →](03-BERT-And-Encoder-Models.md)
