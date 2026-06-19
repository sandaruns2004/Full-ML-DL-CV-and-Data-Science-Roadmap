# 🌍 Machine Translation (Seq2Seq & Attention)

> **Prerequisites**: LSTMs, Word Embeddings | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Sequence-to-Sequence (Seq2Seq) Problem](#1-the-sequence-to-sequence-seq2seq-problem)
2. [The Encoder-Decoder Architecture](#2-the-encoder-decoder-architecture)
3. [The Information Bottleneck](#3-the-information-bottleneck)
4. [The Attention Mechanism (Bahdanau & Luong)](#4-the-attention-mechanism-bahdanau--luong)
5. [Decoding Strategies (Beam Search)](#5-decoding-strategies-beam-search)
6. [Evaluating Translation: BLEU Score](#6-evaluating-translation-bleu-score)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Sequence-to-Sequence (Seq2Seq) Problem

Text Classification maps $N$ words to $1$ label.
Named Entity Recognition maps $N$ words to $N$ labels.

**Machine Translation** is much harder. It must map $N$ words to $M$ words. The input length and output length are almost never the same. Furthermore, the word order might be completely reversed (e.g., English Subject-Verb-Object vs. Japanese Subject-Object-Verb).

This requires a completely new paradigm: **Sequence-to-Sequence (Seq2Seq)** modeling. Before Transformers took over in 2017, the absolute pinnacle of Seq2Seq was the **Encoder-Decoder LSTM with Attention**.

---

## 2. The Encoder-Decoder Architecture

A standard Seq2Seq model consists of two separate RNNs (usually LSTMs or GRUs).

### 1. The Encoder
The Encoder reads the source sentence (e.g., French: *"Je suis étudiant"*). 
It processes the words one by one, updating its hidden state. 
When it reads the final word, we take the very last hidden state. This final state is called the **Context Vector**.
The Context Vector mathematically summarizes the *entire meaning* of the source sentence into a single dense vector.

### 2. The Decoder
The Decoder's job is to write the target sentence (e.g., English: *"I am a student"*).
It is initialized with the Encoder's Context Vector.
1. We feed it a special `<START>` token.
2. It outputs a probability distribution over the English vocabulary. The highest probability word is (hopefully) "I".
3. We take the word "I", and feed it back into the Decoder as the input for the next time step.
4. It outputs "am".
5. This autoregressive process continues until the Decoder outputs an `<END>` token.

---

## 3. The Information Bottleneck

The standard Encoder-Decoder has a fatal flaw: **The Context Vector is a fixed size** (e.g., 512 dimensions).

If the input sentence is 5 words long, compressing it into 512 numbers is easy. But if you try to translate a 50-word legal paragraph, forcing all that semantic meaning into the exact same 512 numbers causes massive information loss. The network literally "forgets" the beginning of the sentence by the time it reaches the end.

Translation quality of standard Seq2Seq models plummets rapidly for sentences longer than 20 words.

---

## 4. The Attention Mechanism (Bahdanau & Luong)

In 2014, Bahdanau et al. published a paper that changed AI forever: *"Neural Machine Translation by Jointly Learning to Align and Translate"*. They introduced **Attention**.

Instead of squashing the entire sentence into one Context Vector, Attention allows the Decoder to "look back" at the Encoder's outputs while it generates each word.

**How it works:**
1. The Encoder does not just pass its final hidden state. It keeps the hidden state produced at *every single word* ($h_1, h_2, h_3$).
2. The Decoder is currently trying to generate word $t$. It has its own current hidden state $s_t$.
3. **Alignment Scores**: We calculate a score between $s_t$ and every Encoder state $h_i$ (e.g., using a dot product).
4. **Attention Weights**: We pass these scores through a Softmax to get probabilities. (e.g., Word 1: 0.1, Word 2: 0.8, Word 3: 0.1).
5. **Context Vector**: We calculate a weighted average of the Encoder states using these probabilities.
6. The Decoder uses this custom Context Vector to generate the word.

Because the Attention Weights change at every time step, the Decoder can "focus" on different parts of the French sentence while generating different parts of the English sentence! This perfectly solved the Information Bottleneck.

*(Note: This RNN-based Attention is the direct ancestor of the Self-Attention mechanism used in Transformers!)*

---

## 5. Decoding Strategies (Beam Search)

During generation, how do we pick the next word?
- **Greedy Search**: Always pick the word with the highest probability. 
  - *Problem*: It's short-sighted. Picking the best word now might force the model into a grammatical dead-end later.

- **Beam Search**: The industry standard for translation. Instead of keeping just 1 path, Beam Search keeps the top $K$ most likely paths (the "beam width", usually $K=5$).
  - At step 1, it generates the top 5 starting words.
  - At step 2, for each of those 5 words, it generates the top 5 next words (25 total paths). It calculates the cumulative probability of all 25 paths, and throws away the bottom 20.
  - It repeats this until the `<END>` token is reached, ensuring a globally optimal sequence!

---

## 6. Evaluating Translation: BLEU Score

How do we evaluate if a translation is good? We can't use Accuracy, because there are many valid ways to translate a sentence.
- Human Reference: *"The quick brown fox jumped."*
- Model A: *"A fast brown fox jumps."* (Good)
- Model B: *"The the the the the."* (Bad)

We use the **BLEU (Bilingual Evaluation Understudy)** score.
BLEU relies on modified $n$-gram precision. It checks how many unigrams, bigrams, trigrams, and 4-grams in the Model's output perfectly match the Human Reference.

It also includes a **Brevity Penalty**. If the reference is 10 words, and the model just outputs "The", it would have 100% precision! The Brevity Penalty heavily punishes models that generate sentences shorter than the reference.

*BLEU scores range from 0 to 1 (or 0 to 100). A score above 30 is understandable, and a score above 50 is considered high-quality.*

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Translate with Hugging Face**: Use the `transformers` library to load `Helsinki-NLP/opus-mt-en-fr` (a massive pre-trained MarianMT translation model). Write a script that translates an English text file into French.
- 🟡 **Seq2Seq from Scratch**: Use PyTorch to build a character-level Encoder-Decoder GRU without attention. Train it on a dataset of English dates ("July 4th 2023") paired with ISO dates ("2023-07-04"). Watch the network learn the conversion rules purely from data!

### What's Next
| Next | Why |
|------|-----|
| [Modern NLP with Transformers](./07-Modern-NLP-With-Transformers.md) | The Encoder-Decoder with Attention was the king of NLP until 2017. Then, Google researchers asked: "If Attention is so powerful, why do we need the RNNs at all?" Thus, the Transformer was born. |

---

[← Named Entity Recognition](./05-Named-Entity-Recognition.md) | [Back to Index](../README.md) | [Next: Modern NLP With Transformers →](./07-Modern-NLP-With-Transformers.md)
