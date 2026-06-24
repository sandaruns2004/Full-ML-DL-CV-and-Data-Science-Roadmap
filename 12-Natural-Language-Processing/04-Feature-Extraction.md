# 04 - Feature Extraction

> **Difficulty**: ⭐⭐☆☆☆ Intermediate | **Prerequisites**: 03-Tokenization | **Estimated Reading Time**: 25 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Bag of Words (BoW)](#2-bag-of-words-bow)
3. [N-Grams (Adding Context)](#3-n-grams-adding-context)
4. [TF-IDF (Term Frequency - Inverse Document Frequency)](#4-tf-idf-term-frequency---inverse-document-frequency)
5. [Code Implementation (Scikit-Learn)](#5-code-implementation-scikit-learn)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

We have successfully tokenized our text into a list of strings: `["the", "cat", "sat"]`.

### 🟢 Beginner
Machine Learning algorithms (like Logistic Regression, Random Forests, or Neural Networks) are essentially giant math equations. You cannot multiply the string `"cat"` by `3.14`. Before we can train an AI to classify a document, we must convert our text tokens into numbers.

### 🟡 Intermediate
This process is called **Feature Extraction** (or Text Vectorization). We need a systematic way to map every unique word in our vocabulary to a specific column in a matrix, allowing us to represent an entire document as a single array of numbers (a vector).

### 🔴 Advanced
Classical NLP relies on **Sparse Vector Representations**. In these representations, the size of the vector is equal to the size of the entire vocabulary (often 50,000+ dimensions). Most of the numbers in the vector will be `0` (hence, "sparse"), because a single sentence only contains a tiny fraction of the English language. We will explore the three fundamental classical approaches: Bag of Words, N-Grams, and TF-IDF.

---

# 2. Bag of Words (BoW)

The simplest way to convert text to numbers is to just count how many times each word appears.

Imagine our entire training dataset consists of only two short sentences:
1.  "The cat sat on the hat."
2.  "The dog ate the hat."

**Step 1: Build the Vocabulary**
We find every unique word across all documents (ignoring punctuation for this example).
`Vocabulary = ["the", "cat", "sat", "on", "hat", "dog", "ate"]` (Size = 7)

**Step 2: Count the Frequencies**
We represent each sentence as a 7-dimensional vector, counting the occurrences of each word.

| Document | "the" | "cat" | "sat" | "on" | "hat" | "dog" | "ate" |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Sentence 1 | 2 | 1 | 1 | 1 | 1 | 0 | 0 |
| Sentence 2 | 2 | 0 | 0 | 0 | 1 | 1 | 1 |

*Sentence 1 Vector:* `[2, 1, 1, 1, 1, 0, 0]`

**The Flaw of BoW:**
It is called a "Bag" of words because it completely destroys the *order* of the words. 
"John loves Mary" and "Mary loves John" will produce the exact same mathematical vector, even though they mean completely different things.

---

# 3. N-Grams (Adding Context)

To fix the word-order problem of BoW, we can group adjacent words together into single tokens. These groups are called **N-Grams**.

*   **Unigrams (N=1):** `["The", "cat", "sat"]` (Standard BoW)
*   **Bigrams (N=2):** `["The cat", "cat sat"]`
*   **Trigrams (N=3):** `["The cat sat"]`

If we use Bigrams, "John loves Mary" `["John loves", "loves Mary"]` is now mathematically distinct from "Mary loves John" `["Mary loves", "loves John"]`.

**The Flaw of N-Grams:**
As you increase N, your vocabulary size explodes exponentially. If you have 10,000 unique words, you have $10,000 \times 10,000 = 100,000,000$ possible bigrams. Your matrix will become so large that your computer will run out of RAM.

---

# 4. TF-IDF (Term Frequency - Inverse Document Frequency)

Look at our BoW example above. The word "the" appears twice in both sentences. Because it has the highest count (2), a naive ML algorithm might assume "the" is the most important word in the dataset. 

We know "the" is useless. We want to penalize words that appear too frequently, and boost the mathematical weight of rare, highly specific words (like "Quantum" or "Dog").

**TF-IDF** solves this mathematically:
$$ \text{TF-IDF} = \text{Term Frequency (TF)} \times \text{Inverse Document Frequency (IDF)} $$

1.  **TF (Term Frequency):** How many times does the word appear in *this specific document*? (Higher is better).
2.  **IDF (Inverse Document Frequency):** How many documents *in the entire dataset* contain this word? (Lower is better).
    $$ \text{IDF} = \log\left(\frac{\text{Total Documents}}{\text{Documents containing the word}}\right) $$

If the word "the" appears in 100% of the documents, the $\log(1)$ is `0`. 
Therefore, the TF-IDF score for "the" becomes `0`. The math automatically deletes useless stopwords!

---

# 5. Code Implementation (Scikit-Learn)

In production, you don't calculate TF-IDF by hand. `scikit-learn` provides highly optimized vectorizers.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "The cat sat on the hat.",
    "The dog ate the hat."
]

# Initialize TF-IDF (We can even include bigrams automatically!)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Convert the text into a sparse mathematical matrix
X = vectorizer.fit_transform(corpus)

# Look at the vocabulary features
print("Vocabulary:", vectorizer.get_feature_names_out())

# Look at the final TF-IDF scores for the first sentence
print("TF-IDF Vector for Sentence 1:\n", X.toarray()[0])
```

---

# 6. Key Takeaways

*   **Feature Extraction** converts text strings into numerical vectors so Machine Learning algorithms can process them.
*   **Bag of Words (BoW)** counts word frequencies but destroys word order.
*   **N-Grams** group adjacent words to preserve local context, but exponentially increase memory usage.
*   **TF-IDF** is the gold standard of classical NLP. It automatically penalizes common stopwords and boosts the mathematical weight of rare, highly informative words.

---

# 7. Next Topic

TF-IDF is brilliant, but it has a fatal flaw: **It does not understand meaning.**

In a TF-IDF matrix, the word "Dog" and the word "Puppy" are treated as two completely unrelated columns. The math has no idea that they are essentially the same animal. 

To teach a computer the *meaning* of words, we have to abandon counting completely. We have to enter the era of Deep Learning and Neural **Word Embeddings**.

[← Tokenization](03-Tokenization.md) | [Back to Index](README.md) | [Next Topic: Word Embeddings →](05-Word-Embeddings.md)
