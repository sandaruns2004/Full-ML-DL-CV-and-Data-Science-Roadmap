# 09 - Sequence-To-Sequence Models (Seq2Seq)

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Prerequisites**: 08-Attention-Mechanisms | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Core Concepts](#3-core-concepts)
4. [Algorithm Workflow: The Encoder-Decoder](#4-algorithm-workflow-the-encoder-decoder)
5. [Training Hack: Teacher Forcing](#5-training-hack-teacher-forcing)
6. [Library Implementation](#6-library-implementation)
7. [Advantages and Limitations](#7-advantages-and-limitations)
8. [Interview Questions](#8-interview-questions)
9. [Key Takeaways](#9-key-takeaways)
10. [Next Topic](#10-next-topic)

---

# 1. What Problem Does This Solve?

Up until now, we have assumed that our sequences align perfectly. If we input 5 words, we output 1 prediction at the end (Many-to-One), or 5 predictions matching the inputs (Many-to-Many synced). But what if the input and output have completely different lengths?

### 🟢 Beginner
If I translate the English phrase *"I am cold"* (3 words) into Spanish, I get *"Tengo frío"* (2 words). A standard RNN expects to output exactly 1 word for every 1 word it reads. It cannot naturally read 3 words and output 2 words. We need an architecture that completely separates the "reading" phase from the "writing" phase.

### 🟡 Intermediate
Tasks like Machine Translation, Text Summarization, and Question Answering require an asynchronous sequence-to-sequence mapping. We need a model that encodes an input sequence of length $N$ into an abstract representation, and then decodes that representation into an output sequence of length $M$.

### 🔴 Advanced
The Seq2Seq architecture introduced by Ilya Sutskever et al. (2014) utilizes two separate recurrent networks: an **Encoder** that models the joint probability of the input sequence, and a **Decoder** that models the conditional probability of the output sequence given the Encoder's final state: $P(y_1, ..., y_{M} | x_1, ..., x_{N})$.

---

# 2. Intuition

Think of an Interpreter at the United Nations.
When a diplomat speaks in German, the Interpreter listens to the entire sentence first. They do not start shouting English words out loud the second the diplomat says their first German word. 

Instead, the Interpreter:
1. **Encodes**: Listens to the whole German sentence and compresses the *meaning* into their brain.
2. **Decodes**: Takes that abstract *meaning* and starts speaking the English sentence from scratch.

A Seq2Seq model acts exactly like this human interpreter, utilizing two separate neural networks for the two distinct phases.

---

# 3. Core Concepts

### 🟢 The Encoder
An RNN (usually an LSTM or GRU) that reads the input sequence one step at a time. It does not output any predictions. Its only job is to update its hidden state. Once it finishes reading the sentence, its final hidden state is called the **Context Vector**.

### 🟡 The Context Vector
This is the handoff. It is the mathematical representation of the entire input sentence's meaning. The Encoder hands this vector over to the Decoder. *(Note: If Attention is used, the Encoder hands over ALL of its hidden states, not just the final one).*

### 🔴 The Decoder
Another RNN. It takes the Context Vector as its initial state. Then, it starts generating words one by one. It uses its own predicted word from step $t$ as its input for step $t+1$. It stops generating when it predicts a special `<EOS>` (End of Sentence) token.

---

# 4. Algorithm Workflow: The Encoder-Decoder

```mermaid
flowchart LR
    subgraph Encoder (Language A)
    x1[I] --> E1[LSTM]
    x2[am] --> E2[LSTM]
    x3[cold] --> E3[LSTM]
    E1 --> E2 --> E3
    end
    
    E3 -.->|Context Vector| D1
    
    subgraph Decoder (Language B)
    D1[LSTM] --> y1[Tengo]
    D2[LSTM] --> y2[frío]
    D3[LSTM] --> y3[< EOS >]
    
    D1 --> D2 --> D3
    y1 -->|Input to Next| D2
    y2 -->|Input to Next| D3
    end
    
    style Encoder fill:#e3f2fd,stroke:#2196f3
    style Decoder fill:#fbe9e7,stroke:#ff5722
```

---

# 5. Training Hack: Teacher Forcing

During training, the Decoder uses its own output from time $t$ as the input for time $t+1$. 
But what if the Decoder makes a mistake early on? 
If it predicts "Banana" instead of "Tengo", it will feed "Banana" into the next step, and the entire rest of the sentence will be garbage. This makes training extremely slow and unstable.

**Teacher Forcing** is a training technique where we intercept the Decoder. Even if the Decoder predicts "Banana", we force-feed it the *actual correct ground-truth word* ("Tengo") as the input for the next step. 
This prevents early mistakes from ruining the entire training sequence.

---

# 6. Library Implementation

Implementing Seq2Seq in PyTorch requires creating two distinct modules.

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        # We only care about the final hidden state to pass to the decoder
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden, cell):
        # x is the previous word (or the Teacher Forced word)
        embedded = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(out.squeeze(1))
        return prediction, hidden, cell

# The main Seq2Seq model orchestrates passing the state from Encoder to Decoder
```

---

# 7. Advantages and Limitations

| Advantages | Limitations |
| ---------- | ----------- |
| Completely decouples input length from output length. | Without Attention, it suffers heavily from the Information Bottleneck on long sentences. |
| Can translate between entirely different vocabularies. | Training is slow because the Decoder must be run sequentially, step-by-step. |

---

# 8. Interview Questions

### Beginner
**Q: Why do we use two separate RNNs (Encoder and Decoder) instead of one?**
A: Because the input sequence and output sequence often have different lengths and completely different vocabularies (e.g., English vs. French). The Encoder's job is just to read and compress, while the Decoder's job is to unroll and generate.

### Intermediate
**Q: How does the Decoder know when to stop generating words?**
A: We train the model to output a special `<EOS>` (End of Sequence) token. During inference, we run the Decoder in a loop until it generates the `<EOS>` token, at which point we terminate the loop.

### Advanced
**Q: What is Teacher Forcing, and what is its main drawback (Exposure Bias)?**
A: Teacher Forcing is feeding the ground-truth previous word into the Decoder during training, rather than the Decoder's own prediction. The drawback is **Exposure Bias**: during training, the model is "spoon-fed" perfect inputs, but during inference, it must rely on its own imperfect predictions. If it makes a small mistake during inference, it has no idea how to recover because it never saw mistakes during training.

---

# 9. Key Takeaways

* **Seq2Seq** models use an **Encoder** to read a sequence and a **Decoder** to generate a new sequence.
* This allows mapping sequences of length $N$ to sequences of length $M$.
* The Encoder creates a **Context Vector** representing the meaning of the input.
* **Teacher Forcing** is used to speed up training by feeding the correct words to the Decoder regardless of its predictions.

---

# 10. Next Topic

Seq2Seq with Attention was the state-of-the-art for NLP from 2014 to 2017. But the fundamental flaw remained: RNNs process data *sequentially*, making them impossible to train efficiently on thousands of GPUs.

In 2017, researchers at Google asked: *What if we throw away the RNN entirely, and just keep the Attention?*

[← Attention Mechanisms](08-Attention-Mechanisms.md) | [Back to Index](README.md) | [Next Topic: Transformers →](10-Transformers.md)
