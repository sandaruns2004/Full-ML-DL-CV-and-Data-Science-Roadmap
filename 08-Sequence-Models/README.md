# 08 - Sequence Models & Transformers 📈

Welcome to **Module 08: Sequence Models & Transformers**. This module bridges the gap between traditional neural networks and state-of-the-art modern language models. 

You will learn how to process data where **order matters** (like text, speech, and time series), moving from foundational Recurrent Neural Networks (RNNs) all the way up to the Attention mechanisms that power today's Transformers (GPT, BERT, LLaMA).

> **Difficulty**: ⭐⭐⭐⭐☆ Intermediate to Advanced | **Prerequisites**: 06-Neural-Networks-Foundations | **Estimated Learning Time**: 40 Hours

---

## 🗺️ Study Roadmap

The curriculum is structured to follow the historical and architectural evolution of sequence models. You will clearly see the **Problem → Limitation → Solution** progression that led to modern AI.

### 🟢 Part 1: The Foundations of Sequence Data
1. [Introduction to Sequential Data](01-Introduction-To-Sequential-Data.md)
2. [Limitations of Traditional Neural Networks](02-Limitations-Of-Traditional-Neural-Networks.md)

### 🟡 Part 2: Recurrent Architectures
3. [Recurrent Neural Networks (RNNs)](03-Recurrent-Neural-Networks-RNNs.md)
4. [RNN Training & BPTT](04-RNN-Training-And-BPTT.md)
5. [Vanishing and Exploding Gradients](05-Vanishing-And-Exploding-Gradients.md)

### 🟡 Part 3: Solving the Memory Problem
6. [Long Short-Term Memory (LSTMs)](06-Long-Short-Term-Memory-LSTMs.md)
7. [Gated Recurrent Units (GRUs)](07-Gated-Recurrent-Units-GRUs.md)

### 🔴 Part 4: The Attention Revolution & Transformers
8. [Attention Mechanisms](08-Attention-Mechanisms.md)
9. [Sequence-to-Sequence Models](09-Sequence-To-Sequence-Models.md)
10. [Transformers](10-Transformers.md)
11. [Modern Transformer Architectures](11-Modern-Transformer-Architectures.md)

### 🔴 Part 5: Practical Applications & MLOps
12. [Time Series Forecasting](12-Time-Series-Forecasting.md)
13. [Sequence Model Evaluation](13-Sequence-Model-Evaluation.md)
14. [Common Failure Cases](14-Common-Failure-Cases.md)
15. [Modern Applications](15-Modern-Applications.md)
16. [Temporal Convolutional Networks](16-Temporal-Convolutional-Networks.md) *(Bonus)*

---

## 💻 Project Roadmap

Theory is nothing without implementation. This module features 8 carefully scoped projects designed to build your portfolio and cement your understanding.

| Project | Concept Focus | Technologies Used |
| :--- | :--- | :--- |
| **[01-Sentiment-Analysis-System](projects/01-Sentiment-Analysis-System)** | Text classification and embeddings | PyTorch, LSTM |
| **[02-SMS-Spam-Detection](projects/02-SMS-Spam-Detection)** | Sequence truncation and basic RNNs | PyTorch, RNN |
| **[03-Language-Translation-System](projects/03-Language-Translation-System)** | Encoder-Decoder architecture | Seq2Seq, Attention |
| **[04-Next-Word-Prediction](projects/04-Next-Word-Prediction)** | Language modeling basics | GRU / LSTM |
| **[05-Time-Series-Forecasting-Dashboard](projects/05-Time-Series-Forecasting-Dashboard)** | Forecasting real-world data | Streamlit, LSTM |
| **[06-Transformer-Text-Classifier](projects/06-Transformer-Text-Classifier)** | Transfer learning with foundation models | Hugging Face, BERT |
| **[07-Text-Summarization-System](projects/07-Text-Summarization-System)** | Generative sequence-to-sequence | Transformers |
| **[08-Mini-Transformer-From-Scratch](projects/08-Mini-Transformer-From-Scratch)** | Understanding attention mathematically | PyTorch, NumPy |

---

## 📓 Interactive Educational Notebooks

Check out the `notebooks/` folder for visual, interactive playgrounds:
- `Sequence_Modeling_Fundamentals_Lab.ipynb` (Understand sequences visually)
- `RNN_vs_LSTM_vs_GRU_Comparison.ipynb` (Train and compare architectures)
- `Attention_Visualization_Lab.ipynb` (Interactive attention heatmaps)
- `Transformer_Playground.ipynb` (Self-attention token relationship visualizations)
- `Time_Series_Forecasting_Lab.ipynb` (End-to-end forecasting pipeline)
- `Sequence_Models_Benchmark.ipynb` (Compare performance across datasets)

---

## 🎓 Skills Gained
By the end of this module, you will understand:
- How sequence data differs from tabular and image data.
- Why standard MLPs and CNNs fail on sequences.
- How LSTMs solve the vanishing gradient problem of RNNs.
- Why the phrase *"Attention Is All You Need"* changed AI forever.
- How to preprocess text, tokenize sequences, and use embeddings.
- How to evaluate models using metrics like BLEU, ROUGE, and Perplexity.
