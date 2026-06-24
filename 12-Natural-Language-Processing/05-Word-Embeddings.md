# 05 - Word Embeddings

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Prerequisites**: 04-Feature-Extraction | **Estimated Reading Time**: 30 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Distributed Representations](#2-distributed-representations)
3. [Word2Vec (The Breakthrough)](#3-word2vec-the-breakthrough)
4. [GloVe and FastText](#4-glove-and-fasttext)
5. [Visualizing the Latent Space](#5-visualizing-the-latent-space)
6. [Library Implementation (Gensim / spaCy)](#6-library-implementation-gensim--spacy)
7. [Key Takeaways](#7-key-takeaways)
8. [Next Topic](#8-next-topic)

---

# 1. What Problem Does This Solve?

In the previous lesson, we used Bag of Words and TF-IDF to convert text into numbers. This is called **One-Hot Encoding**.

### 🟢 Beginner
In a TF-IDF matrix, every word gets its own dedicated column. 
*   `Cat` = `[1, 0, 0, 0]`
*   `Dog` = `[0, 1, 0, 0]`
*   `Car` = `[0, 0, 1, 0]`

If we mathematically calculate the similarity between `Cat` and `Dog` using these vectors, the answer is `0%`. The computer thinks a Dog is just as similar to a Cat as it is to a Car! 

### 🟡 Intermediate
One-Hot Encoding suffers from two fatal flaws:
1.  **Sparsity:** A vector for a single word has 50,000 zeros and a single `1`. It wastes massive amounts of memory.
2.  **Zero Semantic Meaning:** It treats every word as a completely independent variable. It does not understand synonyms.

### 🔴 Advanced
To solve this, we must map words into a **Dense Vector Space** (usually 300 dimensions) where the geometric distance between two vectors perfectly represents their semantic similarity. This dense mathematical mapping is called a **Word Embedding**, and it was one of the most important breakthroughs in the history of Deep Learning.

---

# 2. Distributed Representations

"You shall know a word by the company it keeps." — *John Rupert Firth (1957)*

This quote is the foundation of Word Embeddings. How do humans know that "Dog" and "Puppy" mean the same thing? Because we see them used in the exact same sentences.
*   "The **dog** chewed on the bone."
*   "The **puppy** chewed on the bone."

If we build a Neural Network and train it to read millions of Wikipedia articles, and force it to predict missing words in a sentence, the Neural Network will naturally learn that "dog" and "puppy" are interchangeable. 

To achieve this, the network will assign them very similar mathematical coordinates in its internal memory (the Latent Space). 

Instead of `[0, 1, 0, 0]`, "Dog" becomes a dense vector of 300 floating-point numbers:
`Dog = [0.24, -0.81, 0.12, 0.99, ...]`
`Puppy = [0.22, -0.79, 0.15, 0.98, ...]`

Because the numbers are nearly identical, the geometric distance between them is nearly zero. The computer now understands meaning!

---

# 3. Word2Vec (The Breakthrough)

In 2013, researchers at Google (led by Tomas Mikolov) invented **Word2Vec**. It was the first algorithm that could calculate these dense embeddings incredibly fast.

Word2Vec has two architectural flavors:

### 1. CBOW (Continuous Bag of Words)
The model tries to predict a **target word** based on its surrounding **context words**.
*   *Input:* `["The", "cat", "sat", "on", "the", ___]`
*   *Output to predict:* `"mat"`

### 2. Skip-Gram
The exact opposite of CBOW. The model takes a **single word** and tries to predict the **surrounding context words**.
*   *Input:* `"mat"`
*   *Output to predict:* `["The", "cat", "sat", "on", "the"]`

By training a shallow neural network on billions of words using Skip-Gram, the network's hidden layer weights *become* the Word Embeddings.

---

# 4. GloVe and FastText

Word2Vec was revolutionary, but it spawned two massive improvements:

**GloVe (Global Vectors for Word Representation):**
Developed by Stanford, GloVe mathematically combines the local context tracking of Word2Vec with the global statistical tracking of traditional matrices (like Matrix Factorization). It often produces slightly more stable embeddings than Word2Vec.

**FastText:**
Developed by Facebook (Meta), FastText solved Word2Vec's biggest flaw: **Out of Vocabulary (OOV) words**.
If Word2Vec has never seen the word `"unhappiness"`, it crashes. 
FastText breaks words down into subword N-Grams (e.g., `un`, `happi`, `ness`) and learns embeddings for the subwords. When it sees a brand new word, it simply adds up the vectors of its subwords. It can guess the mathematical meaning of words it has never seen before!

---

# 5. Visualizing the Latent Space

Because embeddings capture semantic meaning mathematically, you can perform actual algebra on words!

The most famous equation in all of Machine Learning is:
$$ \text{Vector}(\text{King}) - \text{Vector}(\text{Man}) + \text{Vector}(\text{Woman}) \approx \text{Vector}(\text{Queen}) $$

If you take the coordinate for "King", subtract the "maleness" direction, and add the "femaleness" direction, you land exactly on the coordinate for "Queen".

**Similarity Map:**
If we compress these 300-dimensional vectors down to 2 dimensions (using PCA or t-SNE) and plot them on a graph, we see natural clustering:
*   Countries (France, Germany, Japan) cluster in the top left.
*   Foods (Apple, Bread, Cheese) cluster in the bottom right.
*   Verbs (Run, Jump, Swim) cluster in the center.

---

# 6. Library Implementation (Gensim / spaCy)

You rarely train Word2Vec from scratch. Instead, you download pre-trained embeddings (trained by Google or Stanford on billions of words) and use them in your code.

```python
import spacy

# Load the medium or large spaCy model (which includes GloVe vectors)
# Note: en_core_web_sm does NOT include true vectors, you must use _md or _lg
nlp = spacy.load("en_core_web_md")

word1 = nlp("dog")
word2 = nlp("puppy")
word3 = nlp("car")

# Calculate Cosine Similarity
print(f"Dog vs Puppy: {word1.similarity(word2):.2f}") # Output: ~0.85
print(f"Dog vs Car: {word1.similarity(word3):.2f}")   # Output: ~0.20

# View the actual 300-dimensional vector
print(word1.vector[:5]) # Prints the first 5 numbers of the embedding
```

---

# 7. Key Takeaways

*   **One-Hot Encoding (TF-IDF)** suffers from sparsity and completely fails to capture the semantic meaning of words.
*   **Word Embeddings** map words into a dense mathematical space where distance equals semantic similarity.
*   **Word2Vec** (Google) learns embeddings by predicting missing words (CBOW) or predicting context (Skip-Gram).
*   **FastText** (Facebook) improves on Word2Vec by learning subword embeddings, solving the Out-of-Vocabulary problem.
*   Embeddings are so accurate you can perform **Vector Arithmetic** on them (King - Man + Woman = Queen).

---

# 8. Next Topic

Now that we have successfully converted our text into dense, meaningful mathematical vectors, we can actually start solving real-world problems.

In the next lesson, we will build an NLP pipeline to automatically read documents and categorize them into different classes (like Spam vs Not Spam).

[← Feature Extraction](04-Feature-Extraction.md) | [Back to Index](README.md) | [Next Topic: Text Classification →](06-Text-Classification.md)
