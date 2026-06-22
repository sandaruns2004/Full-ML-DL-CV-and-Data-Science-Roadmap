# 📖 GPT & Decoder-Only Models

> **Prerequisites**: Transformer Architecture, Attention Mechanism | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [The Philosophy of Decoder-Only Models](#1-the-philosophy-of-decoder-only-models)
2. [Autoregressive Generation Deep Dive](#2-autoregressive-generation-deep-dive)
3. [Deep Mathematics of Masked Self-Attention](#3-deep-mathematics-of-masked-self-attention)
4. [From-Scratch Implementation (NumPy)](#4-from-scratch-implementation-numpy)
5. [Scaling Laws and Emergent Abilities](#5-scaling-laws-and-emergent-abilities)
6. [Library Implementation (PyTorch & Hugging Face)](#6-library-implementation-pytorch--hugging-face)
7. [Visualizing Generation Probabilities](#7-visualizing-generation-probabilities)
8. [Real-World Project: Tiny-GPT](#8-real-world-project-tiny-gpt)
9. [Project Ideas & What's Next](#9-project-ideas--whats-next)

---

## 1. The Philosophy of Decoder-Only Models

While BERT (Encoder-only) excels at understanding bidirectional context for classification, it struggles with **text generation**. If you want a model to write an essay, translate a novel, or write Python code, you need a model that can predict the *next* word given all *previous* words.

Enter **GPT (Generative Pre-trained Transformer)** by OpenAI (2018). 

OpenAI's philosophy was simple: **Language modeling is all you need**. If you train a massive Neural Network to simply predict the next word across the entire internet, it will naturally have to learn grammar, facts, reasoning, and logic to make accurate predictions.

Instead of keeping the Encoder, they kept only the **Decoder** and removed the Cross-Attention layer (since there is no Encoder to attend to). This resulted in the **Decoder-Only Architecture**.

### The Core Difference: Unidirectional vs. Bidirectional
- **BERT (Bidirectional)**: Can look at future words. Used for reading/understanding.
- **GPT (Unidirectional / Autoregressive)**: Can only look at past words. Used for writing/generating.

---

## 2. Autoregressive Generation Deep Dive

**Autoregressive** means the model uses its own previous outputs as inputs for future predictions.

Let's say the prompt is: *"The quick brown"*

1. **Step 1**: Feed `["The", "quick", "brown"]` into the model.
2. **Step 2**: The model outputs a probability distribution over the entire vocabulary (e.g., 50,000 words).
3. **Step 3**: The word with the highest probability is "fox" ($P=0.85$). We sample "fox".
4. **Step 4**: We append "fox" to the input. The new input is `["The", "quick", "brown", "fox"]`.
5. **Step 5**: Feed this new input into the model to predict the next word ("jumps").

This process continues sequentially until the model outputs a special `[EOS]` (End of Sequence) token.

### Sampling Strategies

If we always pick the word with the highest probability (**Greedy Search**), the text often becomes repetitive and boring. To make text more natural, we use sampling techniques:

1. **Temperature ($T$)**: Modifies the softmax probabilities. 
   - $T < 1.0$: Makes the distribution sharper (more deterministic, less creative).
   - $T = 1.0$: Original distribution.
   - $T > 1.0$: Flattens the distribution (more random, highly creative but potentially nonsensical).
2. **Top-K Sampling**: We only sample from the top $K$ most likely next words.
3. **Top-p (Nucleus) Sampling**: We sample from the smallest set of words whose cumulative probability exceeds $p$ (e.g., $p=0.9$).

---

## 3. Deep Mathematics of Masked Self-Attention

In an Encoder, word $i$ can attend to word $j$ even if $j > i$. In a Decoder, this would allow "cheating" during training—the model could just look at the next word instead of predicting it.

To prevent this, we use **Masked Self-Attention**. We apply a lower-triangular mask $M$ to the attention scores before the softmax.

### 3.1 The Attention Equation

Given Queries $Q$, Keys $K$, and Values $V$ (all of shape $N \times d_k$, where $N$ is sequence length and $d_k$ is hidden dimension):

The unmasked attention scores are:
$$ S = \frac{Q K^T}{\sqrt{d_k}} $$

### 3.2 The Mask Matrix
We define a mask matrix $M \in \mathbb{R}^{N \times N}$:

$$ M_{i,j} = \begin{cases} 0 & \text{if } i \ge j \text{ (past and current words)} \\ -\infty & \text{if } i < j \text{ (future words)} \end{cases} $$

For a sequence of length 4, $M$ looks like this:
$$ M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix} $$

### 3.3 Masked Softmax
We add the mask to the scores before applying softmax:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V $$

Because $e^{-\infty} = 0$, the softmax probabilities for future words become exactly 0. Word $i$ will have a 0% probability of attending to word $j$ if $j > i$.

### 3.4 Loss Function: Causal Language Modeling
The model is trained using **Cross-Entropy Loss** applied to every token position simultaneously.

Given a sequence of tokens $x_1, x_2, \dots, x_N$, the probability of the sequence is:
$$ P(x) = \prod_{i=1}^N P(x_i | x_1, \dots, x_{i-1}; \theta) $$

The loss is the negative log-likelihood:
$$ \mathcal{L}(\theta) = -\sum_{i=1}^N \log P(x_i | x_1, \dots, x_{i-1}; \theta) $$

---

## 4. From-Scratch Implementation (NumPy)

To truly understand Masked Self-Attention, let's implement it in pure NumPy.

```python
import numpy as np

def softmax(x, axis=-1):
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MaskedSelfAttention:
    def __init__(self, d_model, d_k):
        self.d_model = d_model
        self.d_k = d_k
        
        # Initialize weights (Xavier/He initialization usually, simplified here)
        self.W_q = np.random.randn(d_model, d_k) * 0.01
        self.W_k = np.random.randn(d_model, d_k) * 0.01
        self.W_v = np.random.randn(d_model, d_k) * 0.01
        
    def forward(self, x):
        """
        x shape: (seq_len, d_model)
        """
        seq_len = x.shape[0]
        
        # 1. Linear projections
        Q = np.dot(x, self.W_q) # (seq_len, d_k)
        K = np.dot(x, self.W_k) # (seq_len, d_k)
        V = np.dot(x, self.W_v) # (seq_len, d_k)
        
        # 2. Compute unmasked scores
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k) # (seq_len, seq_len)
        
        # 3. Create and apply the causal mask
        # np.triu returns upper triangle. We want future tokens to be -inf
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores[mask == 1] = -1e9 # -infinity approximation
        
        # 4. Softmax to get attention weights
        attention_weights = softmax(scores) # (seq_len, seq_len)
        
        # 5. Multiply by Values
        output = np.dot(attention_weights, V) # (seq_len, d_k)
        
        return output, attention_weights

# --- Demonstration ---
# Let's say we have a sequence of 3 words, embedded in 4 dimensions
seq_len = 3
d_model = 4
d_k = 4

# Dummy embeddings for "The" (row 0), "quick" (row 1), "brown" (row 2)
X = np.random.randn(seq_len, d_model)

attention_layer = MaskedSelfAttention(d_model, d_k)
out, attn_weights = attention_layer.forward(X)

print("Attention Weights Matrix (Note the upper triangle is 0):")
print(np.round(attn_weights, 3))
```

*Output Example:*
```text
Attention Weights Matrix (Note the upper triangle is 0):
[[1.    0.    0.   ]   <-- "The" only attends to "The"
 [0.45  0.55  0.   ]   <-- "quick" attends to "The" and "quick"
 [0.30  0.20  0.50 ]]  <-- "brown" attends to all three past words
```

---

## 5. Scaling Laws and Emergent Abilities

In 2020, OpenAI published a seminal paper on **Scaling Laws for Neural Language Models**. They found that the performance (Cross-Entropy Loss) of GPT models scales predictably as a power-law based on three factors:
1. $N$: Number of model parameters.
2. $D$: Dataset size (number of tokens).
3. $C$: Compute used for training (FLOPs).

$$ L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} $$

### Emergent Abilities
As Decoder-only models scale up (e.g., GPT-3 with 175B parameters), they exhibit **emergent abilities**—skills they were never explicitly trained to do, which suddenly "turn on" at a certain scale:
- **Few-Shot Prompting**: Providing a few examples in the prompt is enough for the model to learn a new task on the fly, without any weight updates.
- **Chain-of-Thought Reasoning**: If you ask the model to "think step by step," its mathematical and logical reasoning drastically improves.
- **Translation & Coding**: Despite being trained primarily to predict English text, scaling up causes the model to naturally learn syntax for other languages and programming paradigms.

---

## 6. Library Implementation (PyTorch & Hugging Face)

While writing from scratch is great for learning, in reality, we use PyTorch and Hugging Face. Let's see how to build a miniature Decoder block in PyTorch, and then how to use a pre-trained model.

### 6.1 PyTorch Decoder Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CausalSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # ... (standard multi-head attention reshaping) ...
        # Simplified for brevity. The crucial part is applying the mask:
        
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        
        if mask is not None:
            # Apply causal mask
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # ... (multiply by values and project) ...
        return out
```

### 6.2 Generating Text with Hugging Face (GPT-2)

Let's use a pre-trained GPT-2 model to generate text, applying Top-K and Temperature sampling.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 1. Load pre-trained model and tokenizer
model_name = "gpt2" # 124M parameters
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. Encode the prompt
prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 3. Generate text autoregressively
torch.manual_seed(42) # For reproducibility
output = model.generate(
    input_ids,
    max_length=50,          # Stop at 50 tokens
    temperature=0.8,        # Slightly creative
    top_k=50,               # Restrict to top 50 tokens
    top_p=0.95,             # Nucleus sampling
    do_sample=True,         # Required to use temp/top_k
    pad_token_id=tokenizer.eos_token_id
)

# 4. Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

---

## 7. Visualizing Generation Probabilities

To truly understand Autoregressive generation, let's write code using `matplotlib` to visualize the probability distribution of the top 5 next-word predictions at a specific step.

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Setup
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

prompt = "To bake a chocolate cake, the first ingredient you need is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # Get the logits for the very last token
    next_token_logits = outputs.logits[0, -1, :]
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)

# Get Top 5 predictions
top_k = 5
top_probs, top_indices = torch.topk(probabilities, top_k)

# Convert to Python lists and decode tokens
probs_list = top_probs.numpy().tolist()
tokens_list = [tokenizer.decode([idx]) for idx in top_indices]

# Visualization
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-darkgrid')
bars = plt.barh(tokens_list[::-1], probs_list[::-1], color='#3498db')

plt.title(f'Top {top_k} Next Token Predictions', fontsize=16)
plt.xlabel('Probability', fontsize=12)
plt.ylabel('Token', fontsize=12)

# Add probability text to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.2%}', va='center', fontsize=11)

plt.tight_layout()
plt.show()
```

*Imagine the plot: You'll see words like " flour" (~45%), " sugar" (~20%), " chocolate" (~15%), etc. The model has learned baking facts purely through language modeling!*

---

## 8. Real-World Project: Tiny-GPT

**Goal**: Train a character-level mini-GPT on Shakespeare's text.
**Difficulty**: Advanced

1. Download the `tinyshakespeare` dataset.
2. Build a character-level tokenizer (vocabulary size ~65).
3. Construct a PyTorch Decoder-only Transformer with 4 layers, 4 heads, and $d_{model}=128$.
4. Train it using CrossEntropyLoss to predict the next character.
5. Watch as the model learns to format text like a play, use old English words, and eventually hallucinate realistic-looking Shakespearean dialogue!

*(Reference: Andrej Karpathy's `nanoGPT` repository is the gold standard for this project).*

---

## 9. Project Ideas & What's Next

### Project Ideas
- 🟢 **Prompt Engineering Sandbox**: Use the OpenAI API (or a local Hugging Face model) to test how changes in `Temperature` and `Top-p` affect the output of creative writing tasks vs. factual QA tasks.
- 🟡 **Custom Text Generator**: Fine-tune GPT-2 on your own chat history or favorite author's books using the `Trainer` API from Hugging Face.

### What's Next
| Next | Why |
|------|-----|
| [Vision Transformers (ViT)](./05-Vision-Transformers-ViT.md) | Transformers took over NLP. In 2020, they took over Computer Vision too. We'll see how to treat an image as a sequence of patches. |
| [LLM Fine-Tuning & RLHF](./06-LLM-Fine-Tuning-And-RLHF.md) | A base GPT model just predicts the next word (it doesn't answer questions well). We need RLHF to turn it into ChatGPT. |

---

[← BERT & Encoder-Only Models](03-BERT-And-Encoder-Models.md) | [Back to Index](../README.md) | [Next: Vision Transformers (ViT) →](05-Vision-Transformers-ViT.md)
