# 03 - Self-Attention

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Prerequisites**: 02-Attention-Mechanism | **Estimated Reading Time**: 25 Minutes

---

## 📋 Table of Contents
1. [Why Self-Attention Exists](#1-why-self-attention-exists)
2. [Intuition: The Context Problem](#2-intuition-the-context-problem)
3. [Token Relationships](#3-token-relationships)
4. [Mathematics of Self-Attention](#4-mathematics-of-self-attention)
5. [The Three Weight Matrices ($W^Q, W^K, W^V$)](#5-the-three-weight-matrices-wq-wk-wv)
6. [Library Implementation](#6-library-implementation)
7. [Key Takeaways](#7-key-takeaways)
8. [Next Topic](#8-next-topic)

---

# 1. Why Self-Attention Exists

In the previous lesson, we learned about standard Attention (often called **Cross-Attention**). In Cross-Attention, the Decoder looks at the Encoder. The Query comes from the French sentence being generated, and the Keys and Values come from the original English sentence.

### 🟢 Beginner
But what if we aren't translating? What if we just want an AI to read an English essay and truly understand it? There is no "French Decoder" to ask the questions. The English sentence must somehow look at *itself* to figure out its own meaning.

### 🟡 Intermediate
Standard Neural Networks (like MLPs or CNNs) process words in isolation. If you feed the word "bank" into a neural network, it outputs a static vector. But the word "bank" has no fixed meaning! It could mean a river edge, or a financial institution. We need an architecture where the representation of a word dynamically changes based on the other words surrounding it in the same sentence.

### 🔴 Advanced
**Self-Attention** is the mechanism that allows a sequence to attend to itself. For every token in the sequence, the model computes a Query, a Key, and a Value. Because all three vectors originate from the *exact same sequence*, the dot-product attention formula allows every token to mathematically integrate information from every other token in $O(1)$ sequential steps, creating heavily contextualized embeddings.

---

# 2. Intuition: The Context Problem

Consider these two sentences:

1. *"The **bank** is near the rushing river."*
2. *"The **bank** approved the 30-year loan."*

If you use a simple dictionary lookup (or Word2Vec embedding), the vector for the word "bank" is identical in both sentences. This is a massive problem.

With Self-Attention, when the model processes Sentence 1, the word "bank" looks at all the other words in the sentence. It sees "rushing" and "river". It pays high attention to them, and absorbs their "nature" meaning into itself.

When it processes Sentence 2, "bank" pays high attention to "approved" and "loan". It absorbs their "financial" meaning into itself.

By the time the Self-Attention layer is finished, the mathematical vector for "bank" in Sentence 1 is completely different from the vector for "bank" in Sentence 2. The words have become **Context-Aware**.

---

# 3. Token Relationships

Self-Attention also solves the problem of pronoun resolution (Coreference).

Consider the sentence: 
*"The animal didn't cross the street because **it** was too tired."*

What does the word **it** refer to? The animal, or the street? 

When the word "it" (acting as a Query) compares itself to the Keys of the rest of the sentence, the neural network's learned weights will result in a massive dot-product spike when "it" looks at "animal". 

If we change the sentence to:
*"The animal didn't cross the street because **it** was too wide."*

The weights will instantly shift, and the word "it" will pay maximum attention to the word "street".

---

# 4. Mathematics of Self-Attention

The formula is identical to the one in the previous lesson:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

The mind-bending difference is where $Q$, $K$, and $V$ come from.
In Translation (Cross-Attention):
- $Q$ comes from the Decoder (Language B).
- $K$ and $V$ come from the Encoder (Language A).

In Self-Attention:
- $Q$, $K$, and $V$ **all come from the exact same input sequence.**

If our input sentence is represented by a matrix $X$ (where each row is a word embedding):
$$Q = X \cdot W^Q$$
$$K = X \cdot W^K$$
$$V = X \cdot W^V$$

We simply multiply our sentence $X$ by three different learned weight matrices to generate the Queries, Keys, and Values.

---

# 5. The Three Weight Matrices ($W^Q, W^K, W^V$)

Why do we need to multiply $X$ by these matrices? Why not just use $X$ directly? ($Q=X, K=X, V=X$)

If we just used $X$, the dot product $X X^T$ would be a strict cosine similarity measure. The word "bank" would always pay the most attention to the word "bank". 

But in language, words don't just look for exact copies of themselves. 
- Verbs want to find Nouns.
- Adjectives want to find Nouns.
- Pronouns want to find Subjects.

The learned matrices $W^Q$, $W^K$, and $W^V$ act as **projections**. They rotate and stretch the word vectors into specialized spaces.
- $W^Q$ transforms "it" into a vector that essentially screams, *"I am looking for a singular noun!"*
- $W^K$ transforms "animal" into a vector that screams, *"I am a singular noun!"*
- Because they have been rotated into these matching states, their dot product ($Q K^T$) is huge, and they pay attention to each other.

---

# 6. Library Implementation

Here is how Self-Attention is built. Notice that we create the three Linear layers ($W^Q$, $W^K$, $W^V$), and then feed the *exact same input $x$* into all three!

```python
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        # These are the learned weight matrices (W^Q, W^K, W^V)
        self.q_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=False)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # x is the sequence (e.g., "The bank is near the river")
        
        # 1. Create Q, K, V from the EXACT SAME INPUT x
        q = self.q_proj(x) # [batch, seq_len, head_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        d_k = q.size(-1)
        
        # 2. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 3. Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4. Context Vector
        context = torch.matmul(attention_weights, v)
        
        return context, attention_weights

# Test the layer
batch = 1
seq_len = 5 # 5 words
embed_dim = 256
head_dim = 64

# Dummy sentence
sentence = torch.randn(batch, seq_len, embed_dim)

self_attn = SelfAttention(embed_dim, head_dim)
output, weights = self_attn(sentence)

print("Output shape:", output.shape) # The highly contextualized sentence!
```

---

# 7. Key Takeaways

*   **Self-Attention** allows a sequence to attend to itself, removing the need for separate Encoders and Decoders.
*   It solves the **Context Problem**. Words are no longer static dictionary definitions; their mathematical vectors absorb the meaning of the words surrounding them.
*   The input $X$ is multiplied by three learned matrices ($W^Q, W^K, W^V$) to allow different linguistic rules (like verbs attending to nouns) to be learned.

---

# 8. Next Topic

Self-Attention is incredible. But what if "it" needs to pay attention to "animal" (because it's the subject) AND "tired" (because it's the adjective)? A single attention mechanism might struggle to focus on both grammar and vocabulary at the same time.

To fix this, we split the attention into multiple parallel "Heads".

[← The Attention Mechanism](02-Attention-Mechanism.md) | [Back to Index](README.md) | [Next Topic: Multi-Head Attention →](04-Multi-Head-Attention.md)
