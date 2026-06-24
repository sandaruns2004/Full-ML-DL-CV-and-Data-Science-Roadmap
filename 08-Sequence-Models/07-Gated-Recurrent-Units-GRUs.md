# 07 - Gated Recurrent Units (GRUs)

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Prerequisites**: 06-Long-Short-Term-Memory-LSTMs | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Core Concepts](#3-core-concepts)
4. [Mathematics](#4-mathematics)
5. [Algorithm Workflow](#5-algorithm-workflow)
6. [Library Implementation](#6-library-implementation)
7. [Advantages and Limitations (LSTM vs GRU)](#7-advantages-and-limitations-lstm-vs-gru)
8. [Interview Questions](#8-interview-questions)
9. [Key Takeaways](#9-key-takeaways)
10. [Next Topic](#10-next-topic)

---

# 1. What Problem Does This Solve?

LSTMs successfully solved the Vanishing Gradient problem, but they introduced a new issue: **Computational Bloat**.

### 🟢 Beginner
LSTMs are like a factory with four distinct managers (the four gates/operations) checking every single item on the conveyor belt. This is very thorough, but it takes a lot of time and resources. What if we could merge some of these managers' jobs together to make the factory run faster, without losing quality?

### 🟡 Intermediate
An LSTM uses four separate weight matrices inside every single cell (Forget, Input, Output, and Candidate). For a network with large hidden dimensions, this leads to millions of parameters, requiring massive amounts of GPU memory and making training very slow.

### 🔴 Advanced
In 2014, Kyunghyun Cho et al. introduced the **Gated Recurrent Unit (GRU)**. Their goal was to streamline the LSTM. By coupling the Forget and Input mechanisms and removing the separate Cell State entirely, they created an architecture that maintains the linear error carousel (gradient preservation) but with significantly fewer matrix multiplications.

---

# 2. Intuition

Imagine you are reading a book and keeping a single notebook of facts.
When a new character is introduced, you need to write it down (Input). But you only have limited space on the page, so you have to erase an old, irrelevant character to make room (Forget).

Instead of making "Input" and "Forget" two completely separate decisions (like an LSTM does), the GRU links them into a single **Update** decision. 
*If I write 30% new information, I must erase exactly 30% of old information.* 

By linking these operations, the GRU saves an entire neural network layer's worth of computations.

---

# 3. Core Concepts

### 🟢 No Cell State
The GRU drops the $C^{\langle t \rangle}$ "conveyor belt" entirely. It merges long-term memory and short-term memory into a single Hidden State vector $h^{\langle t \rangle}$.

### 🟡 The Two Gates
While an LSTM has three gates (Forget, Input, Output), a GRU only has two:
1. **Update Gate**: Decides how much of the past memory to keep, and how much to overwrite with new information.
2. **Reset Gate**: Decides how much of the past memory to *completely ignore* when calculating the potential new information.

### 🔴 Linear Interpolation
The GRU maintains gradients through a process called **Linear Interpolation**. The new hidden state is calculated as: `(1 - Update) * Old_State + Update * New_Candidate`. This `+` sign is the exact mechanism that prevents gradients from vanishing, mimicking the LSTM's cell state addition.

---

# 4. Mathematics

At time step $t$, the GRU takes the previous hidden state $h^{\langle t-1 \rangle}$ and the current input $x^{\langle t \rangle}$.

**1. The Update Gate ($z_t$)**
Acts as a combination of the LSTM's Forget and Input gates.
- $z_t \approx 1$: Keep the old state.
- $z_t \approx 0$: Overwrite with new state.
$$z^{\langle t \rangle} = \sigma(\mathbf{W}_z [h^{\langle t-1 \rangle}, x^{\langle t \rangle}] + \mathbf{b}_z)$$

**2. The Reset Gate ($r_t$)**
Decides how much of the past memory to ignore when calculating the candidate.
$$r^{\langle t \rangle} = \sigma(\mathbf{W}_r [h^{\langle t-1 \rangle}, x^{\langle t \rangle}] + \mathbf{b}_r)$$

**3. The Candidate Hidden State ($\tilde{h}_t$)**
Notice that $r_t$ is element-wise multiplied with $h_{t-1}$. If $r_t \approx 0$, the network drops the past and only looks at the current input.
$$\tilde{h}^{\langle t \rangle} = \tanh(\mathbf{W}_h [r^{\langle t \rangle} * h^{\langle t-1 \rangle}, x^{\langle t \rangle}] + \mathbf{b}_h)$$

**4. The Final Hidden State ($h_t$)**
Linear interpolation between the old state and the candidate state.
$$h^{\langle t \rangle} = (1 - z^{\langle t \rangle}) * h^{\langle t-1 \rangle} + z^{\langle t \rangle} * \tilde{h}^{\langle t \rangle}$$

---

# 5. Algorithm Workflow

```mermaid
flowchart TD
    subgraph GRU Cell
    PrevH[h_t-1] --> Concat
    Input[x_t] --> Concat
    
    Concat --> UpdateGate[Update Gate z_t]
    Concat --> ResetGate[Reset Gate r_t]
    
    ResetGate --> Multiply1{x}
    PrevH --> Multiply1
    Multiply1 --> Concat2[Concat with x_t]
    
    Concat2 --> Candidate[Candidate Tanh]
    
    UpdateGate --> Interpolate[(1 - z)*h + z*Candidate]
    PrevH --> Interpolate
    Candidate --> Interpolate
    
    Interpolate --> NextH[h_t]
    end
```

---

# 6. Library Implementation

In PyTorch, swapping an LSTM for a GRU is simply a matter of changing the class name. Notice that unlike the LSTM, the GRU's `.forward()` method only returns the hidden state, not a tuple.

```python
import torch
import torch.nn as nn

class SequenceGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Simply use nn.GRU instead of nn.LSTM
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # GRU returns (out, h_n)
        out, h_n = self.gru(x)
        return self.fc(h_n[-1])
```

---

# 7. Advantages and Limitations (LSTM vs GRU)

There is no definitive theoretical winner between the two, but there are practical heuristics:

| Feature | LSTM | GRU |
| :--- | :--- | :--- |
| **Complexity** | High (3 Gates, 2 States) | Low (2 Gates, 1 State) |
| **Training Speed** | Slower | Faster |
| **Memory Footprint**| Large | Smaller |
| **Long-Range Performance**| Slightly better on very long sequences | Good, but can struggle slightly more |

---

# 8. Interview Questions

### Beginner
**Q: How many gates does a GRU have compared to an LSTM?**
A: A GRU has two gates (Update, Reset), whereas an LSTM has three (Forget, Input, Output).

### Intermediate
**Q: Does a GRU have a Cell State?**
A: No, the GRU merges the concept of long-term memory (Cell State) and short-term memory (Hidden State) into a single Hidden State vector.

### Advanced
**Q: Explain how the GRU Update gate prevents vanishing gradients.**
A: The Update gate $z_t$ creates a linear interpolation equation: $h_t = (1-z_t)h_{t-1} + z_t\tilde{h}_t$. Because $h_{t-1}$ is added linearly, its derivative with respect to $h_t$ is $1-z_t$. This additive path allows gradients to flow backwards through time without suffering from repeated matrix multiplication decay.

---

# 9. Key Takeaways

* The GRU is a **streamlined, faster version** of the LSTM.
* It merges the Cell State and Hidden State into a single vector.
* It uses only **two gates** (Update and Reset) instead of three.
* In practice, GRUs and LSTMs perform very similarly on most tasks, but GRUs are faster to train and use less memory.

---

# 10. Next Topic

We have solved the vanishing gradient problem. But we still have a massive bottleneck: Even with LSTMs and GRUs, the network has to compress an *entire sentence* down into a single fixed-size hidden state vector before making a prediction. 

Can a single vector truly capture the meaning of a 1,000-word essay? To solve this, we need a mechanism that lets the network "pay attention" to specific parts of the past.

[← Long Short-Term Memory (LSTMs)](06-Long-Short-Term-Memory-LSTMs.md) | [Back to Index](README.md) | [Next Topic: Attention Mechanisms →](08-Attention-Mechanisms.md)
