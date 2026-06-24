# 10 - Tokenization & Embeddings

> **Difficulty**: ⭐⭐☆☆☆ Beginner/Intermediate | **Prerequisites**: None | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Word-Level vs Character-Level Tokenization](#2-word-level-vs-character-level-tokenization)
3. [The Solution: Subword Tokenization](#3-the-solution-subword-tokenization)
4. [Byte-Pair Encoding (BPE)](#4-byte-pair-encoding-bpe)
5. [Word Embeddings](#5-word-embeddings)
6. [Library Implementation (Hugging Face)](#6-library-implementation-hugging-face)
7. [Common Failure Cases](#7-common-failure-cases)
8. [Interview Questions](#8-interview-questions)
9. [Key Takeaways](#9-key-takeaways)
10. [Next Topic](#10-next-topic)

---

# 1. What Problem Does This Solve?

Neural networks do not understand English. They do not understand the letter "A" or the word "Apple". Neural networks only understand floating-point numbers.

### 🟢 Beginner
If you want to feed a book into an AI, you first have to convert every word into a number. 
Maybe "Apple" becomes `1`, "Banana" becomes `2`, and "Cat" becomes `3`. 
This dictionary mapping text to numbers is called **Tokenization**.

### 🟡 Intermediate
But how do we chop up the text? Do we assign a number to every *word*? There are millions of words in English, and new words are invented every day (e.g., "iPhone", "TikTok"). We would run out of memory. 
Do we assign a number to every *letter*? If we do that, the sequence length becomes massive, and we learned that Transformers scale terribly with long sequences ($O(N^2)$).

### 🔴 Advanced
Modern LLMs solve this using **Subword Tokenization** (like BPE or SentencePiece). They chop text into chunks that are smaller than words, but larger than characters. Furthermore, replacing a word with a simple integer (`3`) destroys its semantic meaning. We must project that integer into a dense vector space using an **Embedding Layer**, so the mathematical distance between "Apple" and "Banana" is smaller than the distance between "Apple" and "Car".

---

# 2. Word-Level vs Character-Level Tokenization

Why don't the simple methods work?

| Method | Example | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Word-Level** | `["I", "love", "coding"]` | Short sequences. | Massive vocabulary. Cannot handle Out-of-Vocabulary (OOV) words or typos. "run", "runs", and "running" are treated as completely different unrelated words. |
| **Character-Level**| `["I", " ", "l", "o", "v", "e", ...] ` | Tiny vocabulary (26 letters). Never gets an OOV error. | Sequences are way too long for Transformers. Letters hold no semantic meaning on their own. |

---

# 3. The Solution: Subword Tokenization

Subword tokenization is the goldilocks solution. 

It splits rare words into smaller, meaningful chunks, but keeps common words whole.

Example: **"unhappiness"**
- A word-level tokenizer might crash if "unhappiness" isn't in its dictionary.
- A subword tokenizer splits it into: `["un", "happi", "ness"]`

Now, the network can understand the prefix "un" (meaning not), the root "happi", and the suffix "ness". It can instantly understand newly invented words if it understands their subword roots.

---

# 4. Byte-Pair Encoding (BPE)

**BPE** is the most popular subword tokenization algorithm (used by GPT-2, GPT-3, and GPT-4).

How it works (Simplified):
1. Start by splitting all text into individual characters: `["a", "p", "p", "l", "e"]`.
2. Look at the dataset and find the most frequently adjacent pair of characters. Let's say `("e", "r")` appears a million times.
3. Merge them into a new token: `"er"`. Add `"er"` to your vocabulary.
4. Repeat this process recursively. Maybe `"est"` becomes a token. Maybe `"ing"` becomes a token.
5. Stop when you reach your desired vocabulary size (usually around 50,000 tokens for LLMs).

*Note: You can easily see BPE in action. OpenAI provides a free tool at [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app/) where you can see exactly how GPT-4 chops up your sentences.*

---

# 5. Word Embeddings

Once we have our integer tokens (e.g., "Apple" $\to$ `354`), we cannot feed `354` directly into a Neural Network. 
If "Banana" is `102` and "Car" is `355`, the network will assume Apple (`354`) is mathematically closer to Car (`355`) than to Banana (`102`).

We must convert the integer into an **Embedding Vector**.
An embedding is a list of floating-point numbers (e.g., 512 numbers long) where each dimension represents a hidden semantic concept (like gender, royalty, color, or age).

$$\text{Embedding}(\text{"King"}) - \text{Embedding}(\text{"Man"}) + \text{Embedding}(\text{"Woman"}) \approx \text{Embedding}(\text{"Queen"})$$

In PyTorch, `nn.Embedding(vocab_size, d_model)` is essentially a massive lookup table. It is trained simultaneously with the rest of the Transformer via backpropagation.

---

# 6. Library Implementation (Hugging Face)

Here is how to use the tokenizer that powers GPT-2.

```python
# pip install transformers
from transformers import GPT2Tokenizer

# Load the BPE tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Transformers are revolutionizing artificial intelligence!"

# 1. Encode into Subword Integer IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)
# Output: [41502, 373, 389, 21919, ... ]

# 2. See the actual subword string chunks!
chunks = tokenizer.convert_ids_to_tokens(token_ids)
print("Subword Chunks:", chunks)
# Notice how 'revolutionizing' is chopped into 'revol', 'ution', 'izing'
# Notice the special 'Ġ' character indicating a space before the word!

# 3. Decode back to English
print("Decoded:", tokenizer.decode(token_ids))
```

---

# 7. Common Failure Cases

Tokenization is often the hidden source of bizarre LLM behavior.

1.  **Arithmetic Failures**: Why do LLMs struggle with math? Because BPE tokenizes numbers arbitrarily based on frequency. `123` might be one token, but `124` might be split into `["12", "4"]`. The LLM cannot "see" individual digits cleanly.
2.  **Rhyming Failures**: BPE destroys phonetic information. The LLM does not see letters, it sees chunks. It is very hard for an LLM to write a poem that perfectly rhymes if the rhyming words are chunked differently.
3.  **Non-English Languages**: BPE is heavily biased toward English. A 10-word English sentence might be 12 tokens. A 10-word Arabic sentence might be 40 tokens. This makes API costs 4x more expensive for non-English users!

---

# 8. Interview Questions

### Beginner
**Q: What is the difference between Tokenization and Embedding?**
A: Tokenization is the process of chopping text into chunks and assigning an integer ID to each chunk. Embedding is the process of converting that integer ID into a dense vector of floating-point numbers that capture semantic meaning.

### Intermediate
**Q: Why do modern LLMs use Subword Tokenization (like BPE) instead of Word-Level tokenization?**
A: Subword tokenization limits the total vocabulary size (saving memory) while effectively handling Out-of-Vocabulary (OOV) words by breaking them down into known subword roots, prefixes, and suffixes.

### Advanced
**Q: Explain how Byte-Pair Encoding (BPE) is trained.**
A: BPE starts with a base vocabulary of individual characters. It scans the training corpus to find the most frequently occurring pair of adjacent symbols. It merges this pair into a new single symbol, adds it to the vocabulary, and repeats this process iteratively until a target vocabulary size is reached.

---

# 9. Key Takeaways

*   **Tokenization** is the first step of any NLP pipeline. It maps text to integers.
*   **Subword Tokenization (BPE)** is the industry standard. It balances the small vocabulary of character-level methods with the semantic richness of word-level methods.
*   **Embeddings** map these discrete integers into a continuous vector space where mathematical distances represent semantic similarities.
*   Many weird LLM bugs (bad at math, bad at rhyming) are actually Tokenization bugs.

---

# 10. Next Topic

We now know how to architect the Transformer and how to tokenize the data. The final theoretical step before we enter the modern era of Generative AI is understanding exactly how these massive models are trained.

[← The GPT Family](09-GPT-Family.md) | [Back to Index](README.md) | [Next Topic: Transformer Training & Fine-Tuning →](11-Transformer-Training.md)
