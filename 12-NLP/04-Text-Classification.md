# 📊 Text Classification Pipelines

> **Prerequisites**: Word Embeddings, PyTorch Basics | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Text Classification Task](#1-the-text-classification-task)
2. [Building the NLP Pipeline](#2-building-the-nlp-pipeline)
3. [The Embedding Layer](#3-the-embedding-layer)
4. [Using CNNs for Text (1D Convolution)](#4-using-cnns-for-text-1d-convolution)
5. [Using RNNs for Text (LSTM/GRU)](#5-using-rnns-for-text-lstmgru)
6. [End-to-End Implementation (PyTorch)](#6-end-to-end-implementation-pytorch)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Text Classification Task

Text Classification is the foundational task in NLP. Given a sequence of text, assign it a discrete label.
- **Sentiment Analysis**: "This movie was terrible!" $\rightarrow$ `NEGATIVE`
- **Spam Detection**: "Claim your free iPhone!" $\rightarrow$ `SPAM`
- **Intent Recognition**: "Set a timer for 5 minutes." $\rightarrow$ `SET_TIMER`

While Classical NLP (TF-IDF + Naive Bayes) works well as a baseline, Deep Learning models (using Word Embeddings) achieve state-of-the-art results because they understand word order and semantic meaning.

---

## 2. Building the NLP Pipeline

Unlike image classification where pixels are already numbers, text requires a strict preprocessing pipeline before it touches a Neural Network.

1. **Tokenization**: Split text into discrete words/tokens. (`"I love ML" -> ["I", "love", "ML"]`)
2. **Vocabulary Building**: Assign a unique integer ID to every unique word in the dataset. (`{"I": 1, "love": 2, "ML": 3, "<PAD>": 0, "<UNK>": 4}`)
3. **Numericalization**: Convert the list of strings into a list of integers. (`["I", "love", "ML"] -> [1, 2, 3]`)
4. **Padding**: Neural networks require fixed-size inputs. If max sequence length is 5, pad short sentences with zeros. (`[1, 2, 3] -> [1, 2, 3, 0, 0]`)
5. **Embedding**: Pass the integers through an Embedding Layer to get dense vectors.

---

## 3. The Embedding Layer

In PyTorch, the `nn.Embedding` layer acts as a massive lookup table. 
If your vocabulary has 10,000 words, and you want 300-dimensional embeddings, the Embedding Layer holds a weight matrix of size $10,000 \times 300$.

When you pass the integer `42` into the layer, it simply returns the 42nd row of the weight matrix.

**Crucial Concept**: The weights in this Embedding Layer are completely random at initialization. They are updated via Backpropagation during training! The network learns the optimal Word Embeddings specifically for your classification task. 
*(Alternatively, you can load pre-trained GloVe vectors into this layer and freeze them).*

---

## 4. Using CNNs for Text (1D Convolution)

We usually think of Convolutional Neural Networks (CNNs) for images (2D data). But they are incredibly fast and effective for text (1D data)!

Instead of a $3 \times 3$ pixel square, a 1D filter slides across the sequence of words. 
If we use a filter size of 3, the CNN looks at 3 consecutive words (a trigram) at a time.

**Architecture (TextCNN)**:
1. Embedding Layer: $(Batch, Seq\_Len, Embed\_Dim)$
2. 1D Convolution (Multiple filters of sizes 2, 3, and 4 to capture bigrams, trigrams, etc.)
3. 1D Max Pooling (Extracts the single most important feature from each filter's output)
4. Concatenate outputs.
5. Fully Connected Layer $\rightarrow$ Sigmoid/Softmax.

CNNs are highly parallelizable (much faster than RNNs) and excel at extracting key phrases (e.g., detecting the phrase "not good" anywhere in the sentence).

---

## 5. Using RNNs for Text (LSTM/GRU)

While CNNs are fast, they lack memory. They just look at isolated chunks of words. 
Recurrent Neural Networks (RNNs), specifically LSTMs and GRUs, process words sequentially from left to right. They maintain a hidden state that carries the context of the entire sentence so far.

**Architecture (Bi-LSTM)**:
1. Embedding Layer.
2. Bidirectional LSTM: One LSTM reads left-to-right, another reads right-to-left. Their hidden states are concatenated.
3. Extract the final hidden state (which now contains the compressed meaning of the entire sentence).
4. Fully Connected Layer $\rightarrow$ Sigmoid/Softmax.

LSTMs are slower but vastly superior for complex sentences with long-term dependencies (e.g., "The movie, despite having a great cast and massive budget, was ultimately terrible." — The LSTM remembers "terrible" refers back to "movie").

---

## 6. End-to-End Implementation (PyTorch)

Let's implement a complete pipeline and a TextCNN classifier.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. The Preprocessing Simulation ---
# In reality, you'd use torchtext or huggingface tokenizers for this
vocab_size = 5000
embed_dim = 100
max_seq_len = 50
num_classes = 2 # e.g., Positive vs Negative sentiment

# Dummy input: Batch of 32 sentences, already tokenized and padded to length 50
x_batch = torch.randint(0, vocab_size, (32, max_seq_len)) 

# --- 2. The TextCNN Architecture ---
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Convolutional Layers (Filters of size 2, 3, and 4)
        # in_channels = embed_dim (for TextCNN, channels = embedding dimension)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=4)
        
        # 3. Dropout and Fully Connected
        self.dropout = nn.Dropout(0.5)
        # 64 channels * 3 filter sizes = 192 features
        self.fc = nn.Linear(192, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        
        # 1. Embed
        x = self.embedding(x) # (Batch, Seq_Len, Embed_Dim)
        
        # 2. PyTorch Conv1d expects shape: (Batch, Channels, Length)
        # We must permute the sequence and embedding dimensions
        x = x.permute(0, 2, 1) # (Batch, Embed_Dim, Seq_Len)
        
        # 3. Convolutions + ReLU
        c2 = F.relu(self.conv2(x)) # (Batch, 64, Seq_Len - 1)
        c3 = F.relu(self.conv3(x)) # (Batch, 64, Seq_Len - 2)
        c4 = F.relu(self.conv4(x)) # (Batch, 64, Seq_Len - 3)
        
        # 4. Global Max Pooling 1D
        # This squeezes the sequence dimension out, keeping the max feature map
        p2 = F.max_pool1d(c2, c2.size(2)).squeeze(2) # (Batch, 64)
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2) # (Batch, 64)
        p4 = F.max_pool1d(c4, c4.size(2)).squeeze(2) # (Batch, 64)
        
        # 5. Concatenate and Classify
        cat = torch.cat([p2, p3, p4], dim=1) # (Batch, 192)
        cat = self.dropout(cat)
        logits = self.fc(cat) # (Batch, 2)
        
        return logits

# Test the model
model = TextCNN(vocab_size, embed_dim, num_classes)
output = model(x_batch)
print(f"Output shape: {output.shape}") # Expected: (32, 2)
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **IMDB Sentiment Baseline**: Download the IMDB movie reviews dataset. Implement the preprocessing pipeline using PyTorch's `torchtext` library, and train a basic LSTM model to achieve >85% accuracy.
- 🟡 **GloVe Transfer Learning**: Update the PyTorch `nn.Embedding` layer to load pre-trained GloVe 300d vectors. Set `embedding.weight.requires_grad = False` to freeze them. Train the model and see how much faster it converges compared to random initialization!

### What's Next
| Next | Why |
|------|-----|
| [Named Entity Recognition](./05-Named-Entity-Recognition.md) | Text classification assigns one label to the whole sentence. What if we want to assign a label to *every single word* in the sentence (e.g., identifying Names, Dates, and Locations)? That's Sequence Labeling (NER). |

---

[← Word Embeddings (Word2Vec & GloVe)](03-Word-Embeddings.md) | [Back to Index](../README.md) | [Next: NER →](05-Named-Entity-Recognition.md)
