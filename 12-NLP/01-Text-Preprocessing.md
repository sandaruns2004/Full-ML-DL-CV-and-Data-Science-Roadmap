# 📝 NLP: Text Preprocessing

> **Prerequisites**: Python Basics | **Difficulty**: ⭐⭐☆☆☆ Beginner

---

## 📋 Table of Contents
1. [Why Text Preprocessing?](#1-why-text-preprocessing)
2. [Step 1: Text Cleaning and Normalization](#2-step-1-text-cleaning-and-normalization)
3. [Step 2: Tokenization](#3-step-2-tokenization)
4. [Step 3: Stemming vs. Lemmatization](#4-step-3-stemming-vs-lemmatization)
5. [Complete Pipeline using NLTK & SpaCy](#5-complete-pipeline-using-nltk--spacy)

---

## 1. Why Text Preprocessing?

Machine learning models cannot read raw text. They require numbers (vectors). Before we can turn text into numbers, we must standardize the text. 

Raw text is incredibly messy:
- "I love dogs!" vs "i love dogs." (capitalization and punctuation differences)
- "running", "ran", "runs" (different forms of the same root concept)
- HTML tags, URLs, and emojis

Text preprocessing reduces the vocabulary size and collapses similar linguistic structures together, making it easier for models to learn patterns.

---

## 2. Step 1: Text Cleaning and Normalization

The first step is removing noise that doesn't contribute to the underlying meaning of the text.

### Common Cleaning Steps:
1. **Lowercasing**: Converting everything to lowercase. (`"Hello"` $\to$ `"hello"`)
2. **Removing HTML/URLs**: Stripping out `<br>`, `http://...`
3. **Removing Punctuation**: `!`, `?`, `,`, `.`
4. **Removing Stop Words**: Filtering out extremely common words that carry little semantic weight (`the`, `is`, `at`, `which`, `on`).
5. **Unicode Normalization**: Ensuring characters like `é` are represented consistently. Unicode allows the same visual character to be represented by different underlying byte sequences. Libraries like `unicodedata` (using NFKD or NFKC form) resolve this so the network doesn't treat them as unique characters.

```python
import re

text = "The quick brown fox jumps over the lazy dog! Visit http://fox.com"

# 1. Lowercase
text = text.lower()

# 2. Remove URLs (using Regex)
text = re.sub(r'http\S+', '', text)

# 3. Remove punctuation
text = re.sub(r'[^\w\s]', '', text)

print(text)
# Output: "the quick brown fox jumps over the lazy dog visit "
```

---

## 3. Step 2: Tokenization

**Tokenization** is the process of splitting a long string of text into smaller units, called "tokens".

### Word Tokenization
Splitting text by spaces and punctuation.
- `"I love NLP."` $\to$ `["I", "love", "NLP", "."]`

### Subword Tokenization (Modern Approach)
Modern models (like BERT, GPT) do not use word tokenization because of the Out-Of-Vocabulary (OOV) problem. If a model has never seen the word "unbelievable", it fails. 
Instead, they break words into smaller subwords:
- `"unbelievable"` $\to$ `["un", "believ", "able"]`

**Major Subword Algorithms:**
1. **Byte-Pair Encoding (BPE)**: Used by GPT models. It starts with a base vocabulary of single characters and iteratively merges the most frequently occurring adjacent pairs into new tokens.
2. **WordPiece**: Used by BERT. Similar to BPE, but instead of merging based on pure frequency, it merges pairs that maximize the likelihood of the training data when added to the vocabulary.
3. **SentencePiece**: Used by T5 and LLaMA. Unlike BPE/WordPiece which require pre-tokenization (splitting by spaces first), SentencePiece treats spaces just like any other character (often represented as `_`). This is crucial for languages like Chinese or Japanese that don't use spaces between words!

---

## 4. Step 3: Stemming vs. Lemmatization

Both techniques aim to reduce words to their base or root form.

### Stemming (The Dumb Approach)
Stemming applies crude heuristic rules to chop off the ends of words.
- `"running"` $\to$ `"run"`
- `"studies"` $\to$ `"studi"` (Not a real word!)

It is extremely fast but often linguistically incorrect. The most famous algorithm is the **Porter Stemmer**.

### Lemmatization (The Smart Approach)
Lemmatization uses a dictionary and morphological analysis to return the proper dictionary base form (the lemma).
- `"running"` (verb) $\to$ `"run"`
- `"better"` (adjective) $\to$ `"good"`
- `"studies"` $\to$ `"study"`

It is slower but much more accurate.

---

## 5. Complete Pipeline using NLTK & SpaCy

While you can write regex, it's better to use dedicated libraries like NLTK or SpaCy. SpaCy is generally preferred for production because it is faster and more accurate.

### Using SpaCy for an End-to-End Pipeline

```python
# pip install spacy
# python -m spacy download en_core_web_sm
import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

raw_text = "The children are playing happily in the park! They've ran 5 miles."

# Process the text
doc = nlp(raw_text)

processed_tokens = []

for token in doc:
    # Filter out punctuation and stop words
    if not token.is_punct and not token.is_stop:
        # Get the lemma (base form) and lowercase it
        lemma = token.lemma_.lower()
        processed_tokens.append(lemma)

print(processed_tokens)
# Output: ['child', 'play', 'happily', 'park', 'run', '5', 'mile']
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Write a script that takes a raw Wikipedia article and returns a list of lowercase, lemmatized tokens with stop words removed.
- 🟡 **Intermediate**: Implement a simple Byte-Pair Encoding (BPE) algorithm from scratch that merges the most frequent pairs of characters in a small corpus.

### What's Next
Now that our text is cleaned and split into root tokens, we need to convert these tokens into numbers.

| Next Topic | Why |
|------------|-----|
| [Classical NLP: BoW and TF-IDF](./02-Classical-NLP.md) | How to count tokens and create frequency vectors to feed into classical ML models like Naive Bayes. |

---

[← Foundation Models CV](../11-CV/11-Foundation-Models-CV.md) | [Back to Index](../README.md) | [Next: Classical NLP (Bag of Words & TF-IDF) →](02-Classical-NLP.md)
