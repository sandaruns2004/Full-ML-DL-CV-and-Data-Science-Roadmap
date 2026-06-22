# 📚 Classical NLP (Bag of Words & TF-IDF)

> **Prerequisites**: Text Preprocessing, Scikit-Learn | **Difficulty**: ⭐⭐☆☆☆ Beginner

---

## 📋 Table of Contents
1. [The Challenge of Text Data](#1-the-challenge-of-text-data)
2. [Bag of Words (BoW)](#2-bag-of-words-bow)
3. [N-Grams: Capturing Local Context](#3-n-grams-capturing-local-context)
4. [TF-IDF: Term Frequency-Inverse Document Frequency](#4-tf-idf-term-frequency-inverse-document-frequency)
5. [Topic Modeling (Latent Dirichlet Allocation)](#5-topic-modeling-latent-dirichlet-allocation)
6. [Library Implementation (Scikit-Learn)](#6-library-implementation-scikit-learn)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Challenge of Text Data

Machine Learning algorithms (like Logistic Regression or Random Forests) only understand numbers. They cannot process strings like `"I love machine learning"`.

**Natural Language Processing (NLP)** is the subfield of AI focused on enabling computers to understand human language. To do this, we must first convert text into numbers—a process called **Vectorization** or **Feature Extraction**.

Before Neural Networks and Transformers took over, the field relied on "Classical" NLP techniques, which treated documents as collections of word counts.

---

## 2. Bag of Words (BoW)

The simplest way to convert text to numbers is the **Bag of Words (BoW)** model.

**The Concept:**
1. Collect all unique words across all documents in your dataset. This is your **Vocabulary**.
2. For each document, create a vector of length $V$ (where $V$ is the size of the vocabulary).
3. Count how many times each word appears in the document and put that count in the vector.

**Example:**
* Doc 1: "The cat sat on the mat."
* Doc 2: "The dog sat on the log."

*Vocabulary*: `["the", "cat", "sat", "on", "mat", "dog", "log"]`

*BoW Vectors*:
* Vector 1: `[2, 1, 1, 1, 1, 0, 0]`
* Vector 2: `[2, 0, 1, 1, 0, 1, 1]`

**The Problem with BoW**: 
It throws away all grammar and word order. "John loves Mary" and "Mary loves John" have the exact same BoW vector, even though they mean opposite things!

---

## 3. N-Grams: Capturing Local Context

To fix the word order problem in BoW, we can use **N-Grams**. Instead of counting single words (unigrams), we count sequences of $N$ words.

**Example for "The cat sat":**
* **Unigrams (1-grams)**: `["The", "cat", "sat"]`
* **Bigrams (2-grams)**: `["The cat", "cat sat"]`
* **Trigrams (3-grams)**: `["The cat sat"]`

Using Bigrams, "John loves Mary" (`["John loves", "loves Mary"]`) is entirely different from "Mary loves John" (`["Mary loves", "loves John"]`).

*Trade-off*: Adding Bigrams and Trigrams exponentially increases the size of your vocabulary vector, leading to the **Curse of Dimensionality** and extremely sparse matrices (mostly zeros).

---

## 4. TF-IDF: Term Frequency-Inverse Document Frequency

In standard BoW, words like "the", "a", and "is" will always have the highest counts. However, they carry almost no meaning. Conversely, a rare word like "Quantum" might only appear once, but it tells you exactly what the document is about.

**TF-IDF** solves this by weighting words based on their rarity.

### The Mathematics

**1. Term Frequency (TF)**: How often does word $t$ appear in document $d$?
$$ TF(t, d) = \frac{\text{Count of word } t \text{ in document } d}{\text{Total words in document } d} $$

**2. Inverse Document Frequency (IDF)**: How rare is word $t$ across all $N$ documents in the corpus?
$$ IDF(t) = \log \left( \frac{\text{Total number of documents } N}{\text{Number of documents containing word } t} \right) $$
*(Note: If a word appears in every document, the fraction is 1, and $\log(1) = 0$. The word is perfectly penalized!)*

**3. Final TF-IDF Score**:
$$ \text{TF-IDF}(t, d) = TF(t, d) \times IDF(t) $$

A word gets a high TF-IDF score if it appears *frequently* in a *specific* document, but *rarely* in the *overall corpus*.

---

## 5. Topic Modeling (Latent Dirichlet Allocation)

If you have 10,000 news articles but no labels, how do you group them by topic? 
You use Unsupervised Learning via **Latent Dirichlet Allocation (LDA)**.

LDA assumes that:
1. Every document is a mixture of multiple topics (e.g., 70% Politics, 30% Economy).
2. Every topic is a mixture of words (e.g., Politics = "election", "vote", "president").

LDA uses Bayesian probability to reverse-engineer these topics based on the co-occurrence of words in documents.

---

## 6. Library Implementation (Scikit-Learn)

Let's implement BoW, TF-IDF, and LDA using Python's `scikit-learn`.

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

corpus = [
    "The cat sat on the mat.",
    "The dog chased the cat.",
    "Quantum physics is the study of matter and energy.",
    "The study of physics requires math."
]

# 1. Bag of Words (with Unigrams and Bigrams)
print("--- Bag of Words (Bigrams) ---")
vectorizer = CountVectorizer(ngram_range=(1, 2))
X_bow = vectorizer.fit_transform(corpus)
print(pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out()))

# 2. TF-IDF
print("\n--- TF-IDF ---")
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)
print(pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out()))

# 3. Topic Modeling (LDA)
print("\n--- Topic Modeling (LDA) ---")
# We want to find 2 underlying topics
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X_bow) # LDA usually takes BoW, not TF-IDF

# Display the top words for each topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    # Sort the words by weight and take the top 3
    top_words_idx = topic.argsort()[:-4:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Spam Filter**: Download a dataset of Spam and Ham (Safe) emails. Use `TfidfVectorizer` to convert the emails to numbers, and train a `MultinomialNB` (Naive Bayes) classifier to detect spam.
- 🟡 **Movie Review Topics**: Download IMDB movie reviews. Run LDA to find the top 5 hidden topics in the reviews. Look at the words generated for each topic—can you figure out if the topic represents Horror, Comedy, or Romance?

### What's Next
| Next | Why |
|------|-----|
| [Word Embeddings](./03-Word-Embeddings.md) | TF-IDF is powerful, but it doesn't understand that "King" and "Royalty" mean similar things. To capture the actual *semantic meaning* of words, we need Neural Networks and Word Embeddings (Word2Vec). |

---

[← Text Preprocessing](01-Text-Preprocessing.md) | [Back to Index](../README.md) | [Next: Word Embeddings (Word2Vec & GloVe) →](03-Word-Embeddings.md)
