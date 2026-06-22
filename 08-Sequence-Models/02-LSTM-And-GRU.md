# 🧠 Long Short-Term Memory (LSTM) & GRUs

> **Prerequisites**: RNN Fundamentals | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Solution to Vanishing Gradients](#1-the-solution-to-vanishing-gradients)
2. [The Core Concept: The Cell State](#2-the-core-concept-the-cell-state)
3. [The Mathematics of LSTM Gates](#3-the-mathematics-of-lstm-gates)
4. [Gated Recurrent Units (GRU)](#4-gated-recurrent-units-gru)
5. [PyTorch Implementation Example](#5-pytorch-implementation-example)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The Solution to Vanishing Gradients

Vanilla RNNs suffer from **Vanishing Gradients** because they repeatedly multiply the hidden state by the exact same weight matrix $\mathbf{W}_{hh}$ at every time step. 

Hochreiter & Schmidhuber (1997) proposed a radical solution: **Long Short-Term Memory (LSTM) Networks**. 

Instead of a single hidden state $h^{\langle t \rangle}$, an LSTM maintains two entirely separate states:
1. $h^{\langle t \rangle}$: The **Hidden State** (short-term memory, outputted to the next layer).
2. $C^{\langle t \rangle}$: The **Cell State** (long-term memory, the "conveyor belt").

---

## 2. The Core Concept: The Cell State

The Cell State $C^{\langle t \rangle}$ runs straight down the entire sequence chain. It has only minor, linear interactions (pointwise addition and multiplication). 

**Because the derivative of addition is 1, gradients flow backwards across the Cell State completely uninterrupted**. The network essentially creates a "gradient superhighway" that bypasses the deep matrix multiplications that cause gradients to vanish.

To control what gets put on or taken off this conveyor belt, the LSTM uses **Gates**. A gate is simply a fully connected layer with a Sigmoid activation, outputting a vector of numbers strictly between $0$ (let nothing through) and $1$ (let everything through).

---

## 3. The Mathematics of LSTM Gates

At time step $t$, the LSTM takes the previous hidden state $h^{\langle t-1 \rangle}$ and the current input $x^{\langle t \rangle}$.

### Step 1: The Forget Gate ($f_t$)
Decides what information from the past Cell State $C^{\langle t-1 \rangle}$ should be thrown away. 
*Example: If we see a new subject in a sentence, we should forget the gender of the previous subject.*
$$f^{\langle t \rangle} = \sigma(\mathbf{W}_f [h^{\langle t-1 \rangle}, x^{\langle t \rangle}] + \mathbf{b}_f)$$

### Step 2: The Input Gate ($i_t$) & Candidate Cell ($\tilde{C}_t$)
Decides what new information we should add to the Cell State.
- The **Input Gate** determines *which* values to update.
- The **Candidate Cell** creates a vector of *new* candidate values (using Tanh to bound them between -1 and 1).
$$i^{\langle t \rangle} = \sigma(\mathbf{W}_i [h^{\langle t-1 \rangle}, x^{\langle t \rangle}] + \mathbf{b}_i)$$
$$\tilde{C}^{\langle t \rangle} = \tanh(\mathbf{W}_c [h^{\langle t-1 \rangle}, x^{\langle t \rangle}] + \mathbf{b}_c)$$

### Step 3: Updating the Cell State ($C_t$)
We multiply the old state by the forget gate (dropping old info), and add the new candidate values scaled by the input gate (adding new info).
$$C^{\langle t \rangle} = f^{\langle t \rangle} * C^{\langle t-1 \rangle} + i^{\langle t \rangle} * \tilde{C}^{\langle t \rangle}$$
*(Note: $*$ denotes Hadamard/element-wise multiplication).*

### Step 4: The Output Gate ($o_t$) & New Hidden State ($h_t$)
Decides what parts of the Cell State we are going to output as our new Hidden State $h^{\langle t \rangle}$. We push the Cell State through a Tanh to keep it bounded, and multiply it by the Output Gate.
$$o^{\langle t \rangle} = \sigma(\mathbf{W}_o [h^{\langle t-1 \rangle}, x^{\langle t \rangle}] + \mathbf{b}_o)$$
$$h^{\langle t \rangle} = o^{\langle t \rangle} * \tanh(C^{\langle t \rangle})$$

---

## 4. Gated Recurrent Units (GRU)

Introduced by Kyunghyun Cho et al. in 2014, the **GRU** is a streamlined, faster version of the LSTM.

### Key Differences from LSTM:
1. **No Cell State**: It merges $C^{\langle t \rangle}$ and $h^{\langle t \rangle}$ into a single hidden state vector.
2. **Fewer Gates**: It merges the Forget and Input gates into a single **Update Gate** ($z_t$). It also includes a **Reset Gate** ($r_t$) to determine how much of the past to ignore.

### The GRU Math:
$$z^{\langle t \rangle} = \sigma(\mathbf{W}_z [h^{\langle t-1 \rangle}, x^{\langle t \rangle}] + \mathbf{b}_z) \quad \text{(Update Gate)}$$
$$r^{\langle t \rangle} = \sigma(\mathbf{W}_r [h^{\langle t-1 \rangle}, x^{\langle t \rangle}] + \mathbf{b}_r) \quad \text{(Reset Gate)}$$
$$\tilde{h}^{\langle t \rangle} = \tanh(\mathbf{W}_h [r^{\langle t \rangle} * h^{\langle t-1 \rangle}, x^{\langle t \rangle}] + \mathbf{b}_h) \quad \text{(Candidate)}$$
$$h^{\langle t \rangle} = (1 - z^{\langle t \rangle}) * h^{\langle t-1 \rangle} + z^{\langle t \rangle} * \tilde{h}^{\langle t \rangle} \quad \text{(Final State)}$$

**Which should you use?**
- **GRU**: Faster to train, less memory, performs equally well on smaller datasets.
- **LSTM**: Strictly more powerful, more parameters. Better for very long, complex sequences (like Language Modeling).

---

## 5. PyTorch Implementation Example

Let's build a text classifier (Many-to-One) using a Bidirectional LSTM. A Bidirectional LSTM processes the sequence forwards and backwards simultaneously, allowing the network to understand context from both the past *and* the future.

```python
import torch
import torch.nn as nn

class TextClassifierLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        # 1. Embedding Layer: Turns word indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. LSTM Layer
        # bidirectional=True doubles the hidden dimension!
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=2,            # 2 stacked LSTMs
            bidirectional=True, 
            dropout=0.5,
            batch_first=True
        )
        
        # 3. Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        # text shape: [batch_size, sequence_length]
        
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch_size, seq_len, emb_dim]
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # hidden shape: [num_layers * 2, batch_size, hidden_dim]
        
        # We want the final hidden states from the top layer.
        # hidden[-2] is the forward final state, hidden[-1] is the backward final state.
        final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # final_hidden shape: [batch_size, hidden_dim * 2]
        
        return self.fc(self.dropout(final_hidden))

# Initialize
model = TextClassifierLSTM(vocab_size=10000, embedding_dim=100, hidden_dim=256, output_dim=1)
print(model)
```

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Sentiment Analysis**: Download the IMDB Movie Reviews dataset. Tokenize the text, pad the sequences to a fixed length, and train the `TextClassifierLSTM` above to predict positive/negative sentiment.

### What's Next
| Next | Why |
|------|-----|
| [Seq2Seq & Attention](./03-Seq2Seq-And-Attention.md) | We know how to classify sequences. But how do we *translate* them? Enter the Encoder-Decoder architecture. |

---

[← RNN Fundamentals](01-RNN-Fundamentals.md) | [Back to Index](../README.md) | [Next: Seq2Seq & Attention →](03-Seq2Seq-And-Attention.md)
