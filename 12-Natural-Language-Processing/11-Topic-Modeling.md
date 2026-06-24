# 11 - Topic Modeling

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Prerequisites**: 04-Feature-Extraction | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [What is Topic Modeling?](#2-what-is-topic-modeling)
3. [Latent Dirichlet Allocation (LDA)](#3-latent-dirichlet-allocation-lda)
4. [Non-Negative Matrix Factorization (NMF)](#4-non-negative-matrix-factorization-nmf)
5. [Visualizing Topics (pyLDAvis)](#5-visualizing-topics-pyldavis)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

Text Classification is great when you *know* the categories in advance (e.g., "Spam" vs "Not Spam"). This requires thousands of human-labeled examples.

### 🟢 Beginner
What if you download a massive database of 100,000 leaked emails, and you have absolutely no idea what they are about? You don't have categories, and you certainly don't have the time to read them to figure it out.

### 🟡 Intermediate
You need an algorithm that can automatically read the 100,000 emails, discover the hidden themes inside them, and group the emails into "Topics" without any human supervision. This is an **Unsupervised Learning** task called **Topic Modeling**.

### 🔴 Advanced
Topic Modeling algorithms analyze the co-occurrence of words across a corpus. If the words `"stock"`, `"market"`, and `"dividend"` frequently appear in the same documents, the algorithm deduces that they form a mathematical "Topic." The two most common classical algorithms for this are **LDA (Probabilistic)** and **NMF (Linear Algebra)**.

---

# 2. What is Topic Modeling?

It is crucial to understand that a Topic Modeling algorithm does not know English. It will never output a label like `"Topic 1: Finance"`.

Instead, it outputs a list of the most important words that make up that topic. It is up to the human Data Scientist to look at the words and assign a human-readable name.

*   *Algorithm Output for Topic 1:* `["stock", "market", "trade", "bull", "dividend"]` $\to$ Human labels it: **Finance**
*   *Algorithm Output for Topic 2:* `["cpu", "ram", "motherboard", "intel", "amd"]` $\to$ Human labels it: **Hardware**

Furthermore, a single document is not restricted to just one topic. An article about the CEO of Intel buying stocks might be mathematically represented as: `[60% Topic 2, 40% Topic 1]`.

---

# 3. Latent Dirichlet Allocation (LDA)

**LDA** is the most famous Topic Modeling algorithm. It is a generative probabilistic model.

**The Intuition:**
LDA assumes that every document was written through a specific "generative process":
1.  Before writing the document, the author chose a mix of topics (e.g., 70% Politics, 30% Sports).
2.  For every word the author wrote, they first rolled a loaded die to pick one of those topics.
3.  Then, they rolled a second loaded die (specific to that topic) to pick the actual word.

LDA uses Bayseian statistics to reverse-engineer this process. It looks at the final documents and works backward to figure out what the original "loaded dice" (the topics and the word distributions) must have been.

*Pros:* Excellent at handling overlapping topics.
*Cons:* Mathematically complex and slow to train on massive datasets.

---

# 4. Non-Negative Matrix Factorization (NMF)

**NMF** takes a completely different approach. It uses linear algebra instead of probability.

**The Intuition:**
We start with a massive TF-IDF matrix ($V$) where rows are Documents and columns are Words.
NMF attempts to factorize (split) this giant matrix into two much smaller matrices:
1.  **$W$ (Document-Topic Matrix):** How much of each Topic is in each Document?
2.  **$H$ (Topic-Word Matrix):** How much of each Word is in each Topic?

$V \approx W \times H$

By forcing all numbers to be positive (Non-Negative), the algorithm naturally creates parts-based representations that group co-occurring words together into topics.

*Pros:* Incredibly fast. Often produces more coherent and interpretable topics than LDA on shorter texts (like news headlines).
*Cons:* Less robust theoretical foundation than LDA's probability models.

---

# 5. Visualizing Topics (pyLDAvis)

Because Topic Modeling is unsupervised, you cannot calculate "Accuracy". The only way to know if your model worked is to visualize the topics and see if they make sense to a human.

The industry-standard visualization tool is **`pyLDAvis`**.

It produces an interactive web-based dashboard:
1.  **Left Panel (Inter-Topic Distance Map):** Shows all topics as bubbles on a 2D graph. If two bubbles overlap, it means those topics share a lot of the same words. A good model has spread-out, distinct bubbles.
2.  **Right Panel (Term Frequency):** When you click a bubble, it shows a bar chart of the most important words for that specific topic.

*Note: You will implement and visualize this in the `Topic_Modeling_Lab.ipynb` notebook!*

---

# 6. Key Takeaways

*   **Topic Modeling** is an Unsupervised NLP task used to discover hidden themes in a large collection of unlabeled text.
*   **LDA** uses Bayesian probability to reverse-engineer the topic distributions.
*   **NMF** uses linear algebra to factorize a TF-IDF matrix into document-topic and topic-word matrices.
*   Algorithms output clusters of words; humans must interpret those words to name the topic.
*   Evaluation is highly subjective and relies heavily on interactive visualizations like `pyLDAvis`.

---

# 7. Next Topic

We have covered how to classify text (Sentiment) and cluster text (Topic Modeling). But these are primarily analytical tasks.

What if we want to change the text entirely? What if we want to convert English into French? In the next lesson, we will explore the evolution of **Machine Translation**.

[← Language Models](10-Language-Models.md) | [Back to Index](README.md) | [Next Topic: Machine Translation →](12-Machine-Translation.md)
