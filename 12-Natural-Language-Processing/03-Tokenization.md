# 03 - Tokenization

> **Difficulty**: ⭐⭐☆☆☆ Beginner | **Prerequisites**: 02-Text-Preprocessing | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Is Tokenization?](#1-what-is-tokenization)
2. [Word Tokenization (The Classic Approach)](#2-word-tokenization-the-classic-approach)
3. [Character Tokenization (The Failed Approach)](#3-character-tokenization-the-failed-approach)
4. [Subword Tokenization (The Modern Standard)](#4-subword-tokenization-the-modern-standard)
5. [Byte Pair Encoding (BPE)](#5-byte-pair-encoding-bpe)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Is Tokenization?

In the previous lesson, we cleaned the text. But a clean string like `"the cat sat"` is still just one long continuous string of characters to a computer.

### 🟢 Beginner
To count how many times the word "cat" appears, we must chop the string into individual pieces. This process is called **Tokenization**. A **Token** is the fundamental atomic unit of NLP. Depending on the algorithm, a token could be a sentence, a word, or even a single character.

### 🟡 Intermediate
If we just use Python's `.split(" ")`, we will fail. 
*   *Input:* `"Don't do it, Mr. Smith!"`
*   *Naïve Split:* `["Don't", "do", "it,", "Mr.", "Smith!"]`
Notice how the punctuation got glued to the words (`"it,"` and `"Smith!"`). We need a smarter algorithm that understands English punctuation rules.

### 🔴 Advanced
Modern Deep Learning models (like GPT-4) do not tokenize by whole words. They use **Subword Tokenization**. This solves the "Out of Vocabulary" (OOV) problem, allowing models to process rare, misspelled, or entirely made-up words efficiently without requiring infinitely large dictionaries.

---

# 2. Word Tokenization (The Classic Approach)

For decades, NLP relied on Word Tokenization. The goal was to split the string so that every English word and every piece of punctuation became its own token.

*   *Input:* `"Don't do it, Mr. Smith!"`
*   *Proper Tokenization:* `["Do", "n't", "do", "it", ",", "Mr.", "Smith", "!"]`

Notice how `"Don't"` was intelligently split into `"Do"` and `"n't"`, while `"Mr."` was kept together (the period wasn't split off because it's part of the abbreviation).

**Limitations of Word Tokenization:**
1.  **Massive Vocabularies:** If you build a dictionary of every single English word, including names, medical terms, and typos, your dictionary (Vocabulary) will have millions of entries. This makes Neural Networks impossibly slow.
2.  **Out of Vocabulary (OOV):** If the model encounters a word during testing that wasn't in its training dictionary (e.g., a new slang word like `"rizz"`), it crashes or assigns it a useless `<UNK>` (Unknown) token.
3.  **Language Dependence:** It relies on spaces. Languages like Chinese and Japanese do not use spaces between words!

---

# 3. Character Tokenization (The Failed Approach)

If vocabularies are too big, why don't we just tokenize by individual characters?
The English alphabet only has 26 letters. We would have a tiny vocabulary!

*   *Input:* `"Cat"`
*   *Tokens:* `["C", "a", "t"]`

**Why it failed:**
Characters don't have semantic meaning. The letter `"C"` means nothing on its own. Furthermore, a single sentence would now be hundreds of tokens long. The Neural Network would instantly run out of short-term memory before it finished reading the first paragraph.

---

# 4. Subword Tokenization (The Modern Standard)

We need a compromise. 
*   Word tokenization has too much meaning, but the vocabulary is too big.
*   Character tokenization has a tiny vocabulary, but no meaning.

**Subword Tokenization** is the perfect middle ground. It operates on the principle that frequently used words should remain whole words, but rare words should be chopped up into smaller, meaningful chunks.

*   *Input:* `"unhappiness"`
*   *Subword Tokens:* `["un", "happi", "ness"]`

Now, if the model sees the completely made-up word `"un-sandwich-ness"`, it doesn't crash. It breaks it into `["un", "sandwich", "ness"]`. It knows `"un"` means "not", and `"ness"` means "state of". It can mathematically deduce the meaning of a word it has never seen before!

---

# 5. Byte Pair Encoding (BPE)

How do we actually build a Subword Tokenizer? The industry standard algorithm (used by GPT-3, GPT-4, and LLaMA) is **Byte Pair Encoding (BPE)**.

BPE is a data compression algorithm that learns from the training data.

**How BPE Works (Simplified):**
1.  Start by treating every single character in the training dataset as a token.
2.  Scan the dataset and find the most frequent pair of adjacent tokens.
    *   *Example:* You notice `"e"` and `"r"` appear next to each other millions of times.
3.  Merge them into a new single token: `"er"`. Add `"er"` to the vocabulary.
4.  Repeat the process. You scan again and find that `"t"` and `"h"` appear together often. Merge them into `"th"`.
5.  Scan again. You find `"th"` and `"e"` appear together constantly. Merge them into `"the"`.
6.  Stop when your vocabulary reaches a specific size (usually ~50,000 tokens).

Because BPE is driven purely by statistical frequency, it is completely language-agnostic. It works flawlessly on English, Chinese, Python code, and HTML.

---

# 6. Key Takeaways

*   **Tokenization** is the process of chopping strings into atomic units that algorithms can process.
*   **Word Tokenization** (used in classical NLP) uses punctuation rules, but suffers from massive vocabularies and the Out-of-Vocabulary (OOV) problem.
*   **Subword Tokenization** solves the OOV problem by breaking rare words down into meaningful sub-components (like prefixes and suffixes).
*   **BPE (Byte Pair Encoding)** is the industry-standard algorithm used by Modern LLMs to automatically discover the optimal subword tokens based on statistical frequency.

---

# 7. Next Topic

Now that we have successfully chopped our text into tokens, we have a list of strings: `["The", "cat", "sat"]`. 

But Machine Learning algorithms like Logistic Regression and Random Forests cannot multiply strings. They only accept numbers. 

In the next lesson, we will explore the classical NLP techniques used to convert these string tokens into numerical feature vectors.

[← Text Preprocessing](02-Text-Preprocessing.md) | [Back to Index](README.md) | [Next Topic: Feature Extraction →](04-Feature-Extraction.md)
