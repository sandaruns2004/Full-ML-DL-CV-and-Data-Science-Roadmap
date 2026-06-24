# 02 - Text Preprocessing

> **Difficulty**: ⭐⭐☆☆☆ Beginner | **Prerequisites**: 01-Introduction-To-NLP | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Cleaning and Normalization](#2-cleaning-and-normalization)
3. [Stopword Removal](#3-stopword-removal)
4. [Stemming vs. Lemmatization](#4-stemming-vs-lemmatization)
5. [Code Implementation (NLTK & spaCy)](#5-code-implementation-nltk--spacy)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

If you scrape 10,000 product reviews from Amazon, the raw data will look something like this:
`"<p>OMG!!! The battery life on this phone is AWFUL... i h8 it.</p> 😡"`

### 🟢 Beginner
If we feed this raw text directly into a machine learning algorithm, it will treat `"AWFUL..."` and `"awful"` as two completely different words because of the capital letters and the punctuation. The HTML tag `<p>` and the emoji `😡` will confuse the math. 

### 🟡 Intermediate
**Text Preprocessing** is the data-cleaning phase of NLP. Before we do any complex math or AI modeling, we must normalize the text to reduce the vocabulary size and remove irrelevant noise, ensuring the algorithm focuses only on the semantic meaning.

### 🔴 Advanced
While Modern Deep Learning architectures (like LLMs) can often handle raw, uncleaned text natively via Subword Tokenization, traditional statistical NLP pipelines (like TF-IDF or Naive Bayes) will fail catastrophically if the text isn't aggressively preprocessed. Knowing *when* to clean text and *when* to leave it raw is a critical engineering decision.

---

# 2. Cleaning and Normalization

The first step in any traditional NLP pipeline is standardizing the raw string.

**1. Lowercasing:**
Converting all characters to lowercase ensures that `"Apple"`, `"APPLE"`, and `"apple"` are treated as the exact same word. *(Exception: If you are doing Named Entity Recognition, capital letters are very important clues for finding names, so you would skip this step).*

**2. Removing Noise:**
*   **HTML Tags:** Stripping out `<br>`, `<p>`, or `<a>` tags left over from web scraping.
*   **Punctuation:** Removing commas, periods, and exclamation marks.
*   **Special Characters & Emojis:** Removing non-ASCII characters if they don't add value to your specific task.

---

# 3. Stopword Removal

**Stopwords** are the most common words in a language: *the, is, in, at, of, and, a.*

In many NLP tasks (like classifying whether a document is about Sports or Politics), stopwords add absolutely no value. The word "the" appears equally in sports articles and political articles.

By removing stopwords, we drastically shrink the size of our dataset, making our algorithms run much faster and reducing memory usage.

*(Warning: If you are doing Sentiment Analysis, do not blindly remove stopwords. The word "not" is technically a stopword, but removing it changes "I am not happy" into "I am happy"!).*

---

# 4. Stemming vs. Lemmatization

Humans use different forms of the same word for grammatical reasons:
*   *Run*
*   *Running*
*   *Ran*

To a computer, these are three different variables. We want to reduce all of them to their base root form. There are two ways to do this:

### Stemming (The Brute Force Approach)
Stemming uses hardcoded rules to chop off the ends of words. It is very fast, but very dumb.
*   "Running" $\to$ chops off "ning" $\to$ "Run"
*   "Generously" $\to$ chops off "ously" $\to$ "Gener" (Not a real word!)

### Lemmatization (The Intelligent Approach)
Lemmatization uses a massive dictionary and morphological analysis to find the actual dictionary root (the **Lemma**) of the word.
*   "Running" $\to$ looks up verb forms $\to$ "Run"
*   "Better" $\to$ looks up adjectives $\to$ "Good"

**Which to use?**
Use Lemmatization 90% of the time. It is slightly slower than Stemming, but much more accurate because it always results in real English words.

---

# 5. Code Implementation (NLTK & spaCy)

Here is how you perform these steps using the two most popular classical NLP libraries in Python.

### Using NLTK (Natural Language Toolkit)
NLTK is great for academic research and learning the fundamentals.

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')

text = "The foxes are RUNNING quickly!"

# 1. Lowercase
text = text.lower() # "the foxes are running quickly!"

# 2. Remove Punctuation (using Python string methods)
text = "".join([char for char in text if char not in string.punctuation])

# 3. Remove Stopwords
stop_words = set(stopwords.words('english'))
words = text.split()
filtered_words = [word for word in words if word not in stop_words]

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]

print(lemmatized) 
# Output: ['fox', 'running', 'quickly'] 
# (Note: NLTK needs explicit Part-of-Speech tags to lemmatize 'running' to 'run')
```

### Using spaCy
`spaCy` is the industry standard for production classical NLP. It is incredibly fast and does all of the above steps automatically in a single line of code.

```python
import spacy

# Load the small English pipeline
nlp = spacy.load("en_core_web_sm")

text = "The foxes are RUNNING quickly!"

# Process the text (This automatically tokenizes, tags, and lemmatizes)
doc = nlp(text)

# Extract lemmas, ignoring punctuation and stopwords
clean_tokens = [
    token.lemma_.lower() 
    for token in doc 
    if not token.is_stop and not token.is_punct
]

print(clean_tokens)
# Output: ['fox', 'run', 'quickly'] 
# (Notice spaCy correctly lemmatized 'running' to 'run' automatically!)
```

---

# 6. Key Takeaways

*   **Text Preprocessing** reduces noise and shrinks vocabulary size.
*   **Lowercasing** standardizes text but destroys capitalization features.
*   **Stopword Removal** speeds up processing by deleting common words, but can destroy semantic meaning in sentiment tasks.
*   **Lemmatization** safely reduces words to their dictionary roots, whereas **Stemming** brutally chops off suffixes.
*   `spaCy` is generally preferred over `NLTK` for production engineering due to its speed and automated pipelines.

---

# 7. Next Topic

In our code examples above, you might have noticed we had to split the sentence into individual words (e.g., `text.split()`) before we could remove stopwords or lemmatize them.

This process of chopping a string into smaller pieces is called **Tokenization**. While `split()` works for basic English, it fails completely for languages without spaces (like Chinese), or for complex punctuation. Let's look at how Modern NLP actually tokenizes text.

[← Introduction To NLP](01-Introduction-To-NLP.md) | [Back to Index](README.md) | [Next Topic: Tokenization →](03-Tokenization.md)
