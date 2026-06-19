# 🗣️ Seq2Seq Architectures & Bahdanau Attention

> **Prerequisites**: LSTMs & GRUs | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [The Sequence-to-Sequence (Seq2Seq) Problem](#1-the-sequence-to-sequence-seq2seq-problem)
2. [The Encoder-Decoder Architecture](#2-the-encoder-decoder-architecture)
3. [The Information Bottleneck Problem](#3-the-information-bottleneck-problem)
4. [The Attention Mechanism (Bahdanau)](#4-the-attention-mechanism-bahdanau)
5. [Luong (Multiplicative) Attention](#5-luong-multiplicative-attention)
6. [Teacher Forcing during Training](#6-teacher-forcing-during-training)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Sequence-to-Sequence (Seq2Seq) Problem

Standard RNNs and LSTMs map one input to one output at the exact same time step. But what if the input and output sequences have **different lengths**, and their elements don't align one-to-one?

**Examples:**
- Machine Translation: "I am a student" (4 words) $\rightarrow$ "Je suis étudiant" (3 words).
- Text Summarization: A 1,000-word article $\rightarrow$ A 50-word summary.
- Question Answering: "What is the capital of France?" $\rightarrow$ "Paris".

To solve this, Ilya Sutskever et al. (2014) and Kyunghyun Cho et al. (2014) independently developed the **Sequence-to-Sequence (Seq2Seq)** framework.

---

## 2. The Encoder-Decoder Architecture

The Seq2Seq model is built using two distinct LSTMs: the **Encoder** and the **Decoder**.

### The Encoder
The Encoder reads the input sequence one word at a time. It does *not* output anything. Its sole purpose is to process the entire input sequence and compress all its meaning into a single, fixed-length vector called the **Context Vector** ($c$).

Usually, the Context Vector is simply the final hidden state of the Encoder:
$$c = h^{\langle T_x \rangle}_{encoder}$$

### The Decoder
The Decoder acts as a Language Model conditioned on the Context Vector. 
We initialize the Decoder's hidden state using the Encoder's Context Vector:
$$h^{\langle 0 \rangle}_{decoder} = c$$

At step $1$, we feed it a special `<SOS>` (Start of Sentence) token. It outputs a prediction for the first word. We then feed that predicted word into step $2$, and it predicts the second word. It stops when it outputs an `<EOS>` (End of Sentence) token.

---

## 3. The Information Bottleneck Problem

The Encoder-Decoder works decently for short sentences (e.g., $< 10$ words). However, it performs horribly on long sentences (e.g., 50 words).

**The Bottleneck**: The Encoder is forced to compress the exact meaning, grammar, subjects, and verbs of a 50-word sentence into a *single fixed-length vector* (e.g., a 256-dimensional vector). It's mathematically impossible to not lose information. By the time it reads the 50th word, it has largely forgotten the 1st word.

---

## 4. The Attention Mechanism (Bahdanau)

In 2014, Dzmitry Bahdanau proposed a brilliant solution: **Attention**.
Instead of throwing away all the intermediate hidden states of the Encoder, *keep all of them*. 
When the Decoder is generating the $t$-th output word, let it look at *all* the Encoder hidden states, and assign a **weight** (attention score) to each one based on how relevant it is right now.

### The Bahdanau Attention Math

Let $s^{\langle t-1 \rangle}$ be the Decoder's previous hidden state.
Let $h_j$ be the $j$-th hidden state of the Encoder.

**1. Calculate Alignment Scores ($e_{t,j}$)**:
How much should Decoder step $t$ care about Encoder step $j$?
We use a small Feedforward Neural Network (a single linear layer + Tanh) to calculate this:
$$e_{t,j} = \mathbf{v}_a^T \tanh(\mathbf{W}_a s^{\langle t-1 \rangle} + \mathbf{U}_a h_j)$$
*(This is called Additive Attention).*

**2. Calculate Attention Weights ($\alpha_{t,j}$)**:
We push the alignment scores through a Softmax function so they sum to 1. These are our probabilities/weights.
$$\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{k=1}^{T_x} \exp(e_{t,k})}$$

**3. Calculate the Context Vector ($c_t$)**:
The Context Vector for step $t$ is the weighted sum of all Encoder hidden states:
$$c_t = \sum_{j=1}^{T_x} \alpha_{t,j} h_j$$

**4. Decode**:
We concatenate the current input word $y_{t-1}$ with the Context Vector $c_t$, and feed that combined vector into the Decoder LSTM to get the new state $s^{\langle t \rangle}$.

With Attention, the Decoder doesn't rely on a single bottleneck vector. It dynamically builds a custom context vector for *every single output word*.

---

## 5. Luong (Multiplicative) Attention

Minh-Thang Luong et al. (2015) simplified Bahdanau Attention.

Instead of using a Feedforward Network to calculate the alignment scores (which is slow because it involves adding matrices), Luong proposed using a simple dot product:

$$e_{t,j} = (s^{\langle t-1 \rangle})^T \mathbf{W}_a h_j$$

Because matrix multiplication is highly optimized on GPUs, Luong Attention (Multiplicative) is much faster to compute than Bahdanau Attention (Additive), while achieving nearly identical accuracy. This dot-product approach laid the groundwork for the modern Transformer.

---

## 6. Teacher Forcing during Training

When training a Seq2Seq model, we encounter a problem:
If the Decoder makes a mistake at step 1 and predicts the wrong word, it feeds that wrong word into step 2. Step 2 will definitely predict the wrong word, which feeds into step 3, causing an escalating cascade of errors. The model will never learn anything.

**Teacher Forcing** is the standard training technique:
During *training*, instead of feeding the Decoder's own prediction back into itself, we feed the **ground truth** (the actual correct word from the training data) as the input to the next step, *regardless of what the Decoder predicted*.

This stabilizes training immensely. However, at *inference* (testing) time, we don't have the ground truth, so the Decoder must rely on its own predictions.

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Date Translation**: Train a Seq2Seq model with Attention to translate human-readable dates ("August 25th, 2021", "25 Aug 2021", "08/25/21") into standard ISO format ("2021-08-25"). Visualize the Attention Matrix to see exactly which parts of the input the model looks at when generating the year, month, and day!

### What's Next
| Next | Why |
|------|-----|
| [Temporal Convolutional Networks](./04-Temporal-Convolutional-Networks.md) | Can we process sequences without using recurrent loops at all? Yes. We can use 1D Convolutions. |

---

[← LSTM And GRU](./02-LSTM-And-GRU.md) | [Back to Index](../README.md) | [Next: Temporal Convolutional Networks →](./04-Temporal-Convolutional-Networks.md)
