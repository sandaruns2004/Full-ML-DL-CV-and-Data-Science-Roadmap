# 10 - Language Models

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Prerequisites**: 05-Word-Embeddings | **Estimated Reading Time**: 30 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [What is a Language Model?](#2-what-is-a-language-model)
3. [N-Gram Language Models (The Old Way)](#3-n-gram-language-models-the-old-way)
4. [Neural Language Models](#4-neural-language-models)
5. [The Autoregressive Generation Loop](#5-the-autoregressive-generation-loop)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

Up until this point, we have focused on **Discriminative NLP** (Classification, NER, POS Tagging). We give the computer text, and it gives us a label.

### 🟢 Beginner
What if we want the computer to *generate* text? When you type a text message on your phone, the keyboard automatically predicts the next word you are going to type. If you type `"I am going to the"`, the keyboard suggests `"store"`, `"bank"`, or `"gym"`. How does it know that?

### 🟡 Intermediate
To predict the next word, a computer must mathematically understand the **Probability Distribution** of the English language. It must know that $P(\text{store} | \text{going to the})$ is very high, but $P(\text{elephant} | \text{going to the})$ is almost zero.

### 🔴 Advanced
A **Language Model (LM)** is a probability distribution over sequences of words. Building a Language Model is the Holy Grail of Artificial Intelligence. If a machine can perfectly predict the next word in *any* context (a physics textbook, a Python script, a romantic poem), then it has effectively achieved Artificial General Intelligence (AGI). Modern Large Language Models (LLMs) like GPT-4 are simply massive implementations of this exact concept.

---

# 2. What is a Language Model?

Formally, a Language Model assigns a probability $P$ to a sequence of words $W$.

$P(W) = P(w_1, w_2, w_3, ..., w_n)$

By using the Chain Rule of Probability, we can break this down into the probability of predicting the *next* word given all the *previous* words:

$P(\text{"The cat sat"}) = P(\text{"The"}) \times P(\text{"cat"} | \text{"The"}) \times P(\text{"sat"} | \text{"The cat"})$

There are two primary ways to calculate these probabilities: The Classical way (N-Grams) and the Modern way (Neural Networks).

---

# 3. N-Gram Language Models (The Old Way)

Before Deep Learning, we built Language Models by simply counting words in a massive dataset (like all of Wikipedia).

Imagine we want to calculate $P(\text{"sat"} | \text{"The cat"})$.
Using a **Trigram Model** (looking at 3 words at a time), we would count:
1.  How many times does the exact phrase `"The cat"` appear in Wikipedia? (Let's say 1,000 times).
2.  How many times does the exact phrase `"The cat sat"` appear? (Let's say 200 times).

Probability = $200 / 1000 = 0.20$ (or 20%).

**The Fatal Flaw of N-Grams:**
You cannot use a 100-Gram model to remember long contexts. If you want to predict the next word based on the previous 100 words, you would have to search Wikipedia for an exact match of that 100-word sentence. It will never exist! The count will be 0. 

Because of this, N-Gram models have a "Memory" of only 3 or 4 words. They cannot generate coherent paragraphs.

---

# 4. Neural Language Models

To remember long contexts without needing an infinitely large counting matrix, we use **Neural Networks**.

Instead of counting exact word matches, we pass the text through **Word Embeddings** and into a sequence model (like an LSTM or a Transformer).

1.  *Input:* The vector embeddings for `["The", "cat"]`.
2.  *Hidden State:* The Neural Network compresses the meaning of the input into a dense memory vector.
3.  *Output:* The network projects this memory vector onto the entire Vocabulary (e.g., 50,000 words) and outputs a probability score for every single word in the dictionary.

Because Neural Networks learn continuous mathematics (embeddings) rather than discrete counting, they can guess the next word even if they have *never seen the exact sentence before*.

---

# 5. The Autoregressive Generation Loop

How does a Language Model write a whole essay? It uses an **Autoregressive Loop**. It generates one word at a time, appends it to the input, and runs the model again.

**Step 1:**
*   Input: `["The"]`
*   Model Output (Probabilities): `cat (40%)`, `dog (30%)`, `man (10%)`
*   Action: We sample the highest probability (`"cat"`).

**Step 2:**
*   Input: `["The", "cat"]` (We feed the output back into the input!)
*   Model Output: `sat (60%)`, `ran (20%)`, `slept (10%)`
*   Action: We sample `"sat"`.

**Step 3:**
*   Input: `["The", "cat", "sat"]`
*   Model Output: `on (90%)`, `under (5%)`
*   Action: We sample `"on"`.

This loop continues until the model outputs a special `<STOP>` token. This exact same loop is how ChatGPT generates its responses!

---

# 6. Key Takeaways

*   **Language Models** calculate the probability of a sequence of words, allowing them to predict the next word.
*   **N-Gram LMs** use simple counting and probability ratios, but fail because they cannot remember long-term context (the "curse of dimensionality").
*   **Neural LMs** use Word Embeddings and deep networks to calculate probabilities mathematically, allowing for infinite context tracking.
*   **Autoregressive Generation** is the loop of predicting one word, adding it to the input, and repeating the process to generate long-form text.

---

# 7. Next Topic

Language Models are powerful, but we don't always need to generate text. Sometimes we just want to understand the overarching themes of thousands of documents simultaneously.

In the next lesson, we will step away from Supervised Deep Learning and look at an Unsupervised NLP technique: **Topic Modeling**.

[← Part-Of-Speech Tagging](09-Part-Of-Speech-Tagging.md) | [Back to Index](README.md) | [Next Topic: Topic Modeling →](11-Topic-Modeling.md)
