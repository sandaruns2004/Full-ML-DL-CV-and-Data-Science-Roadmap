# 🧠 Word Embeddings (Word2Vec & GloVe)

> **Prerequisites**: Classical NLP, Neural Network Basics | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Flaw of One-Hot Encoding](#1-the-flaw-of-one-hot-encoding)
2. [What is a Word Embedding?](#2-what-is-a-word-embedding)
3. [Word2Vec: Continuous Bag of Words (CBOW)](#3-word2vec-continuous-bag-of-words-cbow)
4. [Word2Vec: Skip-Gram](#4-word2vec-skip-gram)
5. [The Mathematics of Skip-Gram](#5-the-mathematics-of-skip-gram)
6. [GloVe and FastText](#6-glove-and-fasttext)
7. [Library Implementation (Gensim)](#7-library-implementation-gensim)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. The Flaw of One-Hot Encoding

In Classical NLP (BoW/TF-IDF), we represent words using **One-Hot Encoding**. If our vocabulary has 10,000 words, the word "Cat" might be a vector of 9,999 zeros and a single `1` at index 42. The word "Dog" might have a `1` at index 815.

Mathematically, the dot product between the "Cat" vector and the "Dog" vector is exactly $0$. 
The dot product between the "Cat" vector and the "Car" vector is also exactly $0$.

One-Hot Encoding treats every word as entirely independent. It has absolutely no concept of synonyms, semantics, or relationships. The network has to learn the meaning of "Dog" and "Puppy" completely separately.

---

## 2. What is a Word Embedding?

A **Word Embedding** is a dense vector of floating-point numbers (usually length 100 to 300) that captures the semantic meaning of a word.

Instead of a 10,000-length vector of zeros and ones, "Cat" becomes a 300-length vector like: `[0.45, -0.12, 0.88, ...]`.

Because the vector contains meaning, words with similar meanings will be grouped together in this 300-dimensional space. The Cosine Similarity between "Cat" and "Dog" will be very high ($\approx 0.9$), while the similarity between "Cat" and "Car" will be low ($\approx 0.1$).

But how do we find these magical numbers? We use a Neural Network, relying on a linguistic theory by John Firth (1957): 
> *"You shall know a word by the company it keeps."*

---

## 3. Word2Vec: Continuous Bag of Words (CBOW)

In 2013, Google researcher Tomas Mikolov introduced **Word2Vec**. He proposed two simple Neural Network architectures to learn word embeddings: CBOW and Skip-Gram.

**The CBOW Objective**: Predict a target word given its surrounding context words.

Example Sentence: *"The quick brown fox jumps over the lazy dog."*
Target Word: **fox**
Context Words (window size=2): **[quick, brown, jumps, over]**

**The Architecture:**
1. Input Layer: The one-hot vectors of the 4 context words.
2. Hidden Layer: A linear layer (no activation function) with $N$ neurons (e.g., $N=300$). This matrix is $V \times N$.
3. Output Layer: A softmax layer with $V$ neurons, predicting the target word.

By forcing the network to predict missing words from millions of sentences, the weights in the Hidden Layer naturally become the Word Embeddings!

---

## 4. Word2Vec: Skip-Gram

Skip-Gram is the exact inverse of CBOW, and it is generally considered the superior architecture because it works better for rare words.

**The Skip-Gram Objective**: Predict the surrounding context words given a single target word.

Target Word: **fox**
Predict Context Words: **[quick, brown, jumps, over]**

The network takes the one-hot vector for "fox", compresses it into the 300-dimensional hidden layer, and then uses that hidden vector to predict the probabilities of the surrounding words.

### Negative Sampling
Calculating the Softmax denominator over a vocabulary of 50,000 words for every single word in Wikipedia is computationally impossible.

Word2Vec solves this using **Negative Sampling**. Instead of predicting the exact word, we turn it into a Binary Classification task:
1. Give the network a True pair: `("fox", "brown")` $\rightarrow$ Target output: `1`
2. Give the network 5 Random (Negative) pairs: `("fox", "apple")`, `("fox", "computer")` $\rightarrow$ Target output: `0`

The network just uses Sigmoid to classify if the pair is real context or random garbage. This drastically speeds up training.

---

## 5. The Mathematics of Skip-Gram

Let $\mathbf{v}_w$ be the input embedding (hidden layer) for target word $w$.
Let $\mathbf{u}_c$ be the output embedding for context word $c$.

The probability of seeing context word $c$ given target word $w$ is defined via softmax:
$$ P(c | w) = \frac{\exp(\mathbf{v}_w \cdot \mathbf{u}_c)}{\sum_{i \in V} \exp(\mathbf{v}_w \cdot \mathbf{u}_i)} $$

The loss function for a sequence of words is to maximize the log probability of all context windows:
$$ \mathcal{L} = \sum_{t=1}^{T} \sum_{-m \le j \le m, j \neq 0} \log P(w_{t+j} | w_t) $$

Notice the dot product $\mathbf{v}_w \cdot \mathbf{u}_c$. To maximize this probability, the network must make the dot product large for words that appear together. A large dot product means the vectors are pointing in the same direction. Thus, similar words get similar vectors!

---

## 6. GloVe and FastText

Word2Vec is great, but researchers quickly improved upon it:

### GloVe (Global Vectors for Word Representation)
Developed at Stanford. Word2Vec only looks at local windows (5 words at a time). GloVe builds a massive **Co-occurrence Matrix** of the entire dataset (how often word $i$ appears next to word $j$ across all of Wikipedia) and uses Matrix Factorization (SVD) to find the embeddings. It combines the benefits of local context windows with global statistical counts.

### FastText
Developed at Facebook. Word2Vec cannot handle Out-Of-Vocabulary (OOV) words. If it sees the word "apple", it has a vector. If it sees "apples", but "apples" wasn't in the training data, it crashes.

FastText breaks words down into **character n-grams**. 
"apple" $\rightarrow$ `["<ap", "app", "ppl", "ple", "le>"]`
The final embedding for a word is the sum of its character n-gram embeddings. This means FastText can generate a valid embedding for a misspelled word or a brand new word by looking at its sub-components!

---

## 7. Library Implementation (Gensim)

We don't need to code Word2Vec in PyTorch. The Python library `gensim` is the industry standard for classical word embeddings.

```python
import gensim.downloader as api
from gensim.models import Word2Vec

# --- 1. Train your own Word2Vec ---
# A small custom dataset
sentences = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["the", "smart", "dog", "barks", "at", "the", "fox"]
]

# Train a Skip-Gram model (sg=1) with vector size 10
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, sg=1)

print("Vector for 'fox':")
print(model.wv['fox'])

print("\nSimilarity between 'fox' and 'dog':")
print(model.wv.similarity('fox', 'dog'))

# --- 2. Use Pre-trained GloVe Embeddings ---
# Download Stanford's GloVe vectors (trained on 2 billion tweets)
# Note: This downloads a large file (~100MB for the 25d version)
print("\nDownloading GloVe...")
glove_vectors = api.load("glove-twitter-25")

# The famous king - man + woman = queen analogy
print("\nSolving: King - Man + Woman = ?")
result = glove_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(result) # Should output something like [('queen', 0.85)]
```

---

## 8. Project Ideas & What's Next

### Project Ideas
- 🟢 **Word Similarity Visualizer**: Download a pre-trained Word2Vec model. Pick 100 random words, extract their 300-dimensional vectors, and use PCA or t-SNE (from `sklearn.manifold`) to reduce them to 2 dimensions. Plot them using `matplotlib` to visually see how animals group together, countries group together, etc.
- 🟡 **Document Similarity Search**: Convert entire documents into vectors by taking the average of all the Word2Vec embeddings of the words in the document. Then, write a script where a user inputs a query sentence, and you return the most semantically similar document using Cosine Similarity.

### What's Next
| Next | Why |
|------|-----|
| [Text Classification](./04-Text-Classification.md) | Now that we can convert words into dense mathematical vectors that capture semantic meaning, we can feed these vectors into deep Neural Networks (RNNs, LSTMs, CNNs) to classify text sentiment! |

---

[← Classical NLP](./02-Classical-NLP.md) | [Back to Index](../README.md) | [Next: Text Classification →](./04-Text-Classification.md)
