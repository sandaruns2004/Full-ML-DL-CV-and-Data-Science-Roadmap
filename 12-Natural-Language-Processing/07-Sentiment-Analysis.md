# 07 - Sentiment Analysis

> **Difficulty**: ⭐⭐☆☆☆ Intermediate | **Prerequisites**: 06-Text-Classification | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Rule-Based Methods (VADER)](#2-rule-based-methods-vader)
3. [Machine Learning Approaches](#3-machine-learning-approaches)
4. [Deep Learning & LLM Approaches](#4-deep-learning--llm-approaches)
5. [Industry Applications](#5-industry-applications)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

Text Classification helps us categorize documents by topic (e.g., Sports vs. Tech). But for businesses, the *topic* of a review is less important than the *emotion* behind it.

### 🟢 Beginner
If an airline receives 10,000 tweets a day, they cannot hire enough humans to read them all. They need an automated way to instantly identify which tweets are from angry customers whose flights were canceled, so customer support can prioritize them over tweets from happy customers.

### 🟡 Intermediate
**Sentiment Analysis** (Opinion Mining) is a specialized subset of Text Classification. Its goal is to extract the subjective polarity of a text.
*   *Polarity:* Positive (+1), Neutral (0), or Negative (-1).
*   *Emotion:* Anger, Joy, Sadness, Fear.

### 🔴 Advanced
Sentiment Analysis is notoriously difficult because human emotion is complex. Algorithms struggle with sarcasm ("Oh great, another delay!"), negation ("I am not very happy"), and domain-specific vocabulary (In finance, a "bullish" market is positive, but in a china shop, a "bull" is negative). Over the last 20 years, we have evolved from strict rule-based lexicons to context-aware Deep Learning models to solve these nuances.

---

# 2. Rule-Based Methods (VADER)

Before Machine Learning was accessible, researchers built Sentiment Lexicons. They literally sat down and assigned a score to thousands of English words.
*   `Excellent` = +3.0
*   `Good` = +1.5
*   `Terrible` = -3.0

The algorithm simply scans a sentence, looks up every word in the dictionary, and calculates the average score.

**VADER (Valence Aware Dictionary and sEntiment Reasoner)** is the most famous rule-based model. It is specifically tuned for Social Media.
*   It understands capitalization: `"GOOD"` is scored higher than `"good"`.
*   It understands punctuation: `"Good!!!"` is scored higher than `"Good"`.
*   It understands emojis: `:)` is positive, `>:[` is negative.

*Pros:* Requires zero training data. Incredibly fast.
*Cons:* Cannot understand sarcasm. Struggles with complex grammar.

---

# 3. Machine Learning Approaches

To understand grammar, we must train a model using supervised learning.

1.  **Dataset:** Collect 50,000 movie reviews (e.g., the IMDB Dataset) labeled as Positive or Negative.
2.  **Feature Extraction:** Convert the reviews into TF-IDF vectors (usually including Bigrams to catch simple negations like "not good").
3.  **Modeling:** Train a Logistic Regression or Naive Bayes classifier on the vectors.

Because the model learns from the data, it will automatically figure out that the phrase "waste of time" has a strong mathematical correlation with the Negative class, even if the individual words "waste" and "time" aren't inherently negative.

---

# 4. Deep Learning & LLM Approaches

TF-IDF and Naive Bayes still struggle with long-distance context and sarcasm. 

**The Deep Learning Era (LSTMs & Transformers):**
By using Word Embeddings, an LSTM or Transformer model actually reads the sentence in order. It understands that in the sentence *"The movie was absolutely terrible, I loved every second of it"*, the positive ending completely overrides the negative beginning (irony/campiness).

**The Modern Era (LLMs):**
Today, the state-of-the-art approach to Sentiment Analysis is **Prompt Engineering an LLM** (like GPT-4 or Llama-3). 
You don't need to train a model or build a pipeline. You simply send an API request:
*   *Prompt:* `Analyze the sentiment of this review and output a JSON with the keys 'polarity' and 'emotion': "The battery life is awful but the screen is gorgeous."`
*   *Output:* `{"polarity": "mixed", "emotion": "frustrated"}`

LLMs perform **Aspect-Based Sentiment Analysis (ABSA)** natively. They can understand that the user is simultaneously negative about the battery and positive about the screen.

---

# 5. Industry Applications

*   **Brand Monitoring:** Tracking Twitter during a product launch. If the sentiment graph suddenly spikes negative, the PR team is alerted immediately.
*   **Algorithmic Trading (Finance):** Hedge funds use sentiment analysis to read thousands of financial news articles and Bloomberg terminal headlines per second. If the sentiment around a specific company turns heavily negative, algorithms automatically sell the stock before humans even finish reading the headline.
*   **Customer Service:** Analyzing the transcript of a live phone call. If the customer's sentiment drops into "Extreme Anger", the software alerts a human manager to take over the call from the automated bot.

---

# 6. Key Takeaways

*   **Sentiment Analysis** categorizes text based on emotion and polarity (Positive/Negative/Neutral).
*   **Rule-Based Models (VADER)** are fast and require no training data, relying on hardcoded dictionaries and heuristics (like emojis and ALL CAPS).
*   **Machine Learning Models (TF-IDF + SVM)** are better at capturing dataset-specific nuances.
*   **LLMs** are the modern gold standard, naturally capable of Aspect-Based Sentiment Analysis and understanding complex sarcasm without any fine-tuning.

---

# 7. Next Topic

We have mastered categorizing the overall *meaning* or *emotion* of a document. 

But what if we don't care about the emotion? What if we want to extract hard facts? What if we have a 10-page legal contract, and we just want to extract the names of the people involved, the dates, and the dollar amounts?

In the next lesson, we will explore **Named Entity Recognition (NER)**.

[← Text Classification](06-Text-Classification.md) | [Back to Index](README.md) | [Next Topic: Named Entity Recognition →](08-Named-Entity-Recognition-NER.md)
