# 14 - Text Summarization

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Prerequisites**: 13-Question-Answering-Systems | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Extractive Summarization (The Highlighter Method)](#2-extractive-summarization-the-highlighter-method)
3. [Abstractive Summarization (The Writer Method)](#3-abstractive-summarization-the-writer-method)
4. [Evaluation Metrics (ROUGE)](#4-evaluation-metrics-rouge)
5. [Industry Applications](#5-industry-applications)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

We produce more text daily than any human could read in a lifetime. Financial analysts receive hundreds of 50-page earnings reports every morning. Lawyers must review thousands of pages of case law.

### 🟢 Beginner
We need an algorithm that can read a 10-page document and automatically generate a 1-paragraph summary containing only the most important bullet points, saving humans hours of reading time.

### 🟡 Intermediate
**Text Summarization** is the process of distilling the most important information from a source text to produce an abridged version. Just like Question Answering, there are two completely different architectural approaches to solving this: Extractive and Abstractive.

### 🔴 Advanced
Summarization is exceptionally difficult to evaluate. Unlike Text Classification where the prediction is either 100% right or 100% wrong, a summary is highly subjective. Two humans will write two completely different summaries of the exact same article. We must use specialized N-Gram overlap metrics (like **ROUGE**) to mathematically score how "good" a generated summary is.

---

# 2. Extractive Summarization (The Highlighter Method)

Imagine you give a student a textbook and a yellow highlighter. You tell them to highlight the 3 most important sentences on the page.

That is exactly how **Extractive Summarization** works. It does *not* generate new words. It simply ranks every existing sentence in the document by "importance" and returns the top 3 sentences exactly as they were written.

**How it works (TextRank):**
A popular classical algorithm is **TextRank** (based on Google's PageRank algorithm for websites).
1.  Split the document into a list of sentences.
2.  Convert every sentence into a TF-IDF vector or Word Embedding.
3.  Calculate the Cosine Similarity between every single sentence and every other sentence.
4.  If Sentence A is highly similar to many other sentences in the document, it must be a "central theme" of the document. It gets a high score.
5.  Return the top N highest-scoring sentences.

*Pros:* Incredibly fast. Zero hallucinations (it only outputs text that is literally in the document).
*Cons:* The summary is often disjointed. It doesn't read like a flowing paragraph, it reads like a list of disconnected quotes.

---

# 3. Abstractive Summarization (The Writer Method)

Imagine you give a student a textbook, have them read it, take the textbook away, and tell them to write a summary from memory in their own words.

That is **Abstractive Summarization**. The model reads the document, understands the meaning, and generates a brand new paragraph containing words that might not even exist in the original text.

**How it works (Seq2Seq):**
Just like Machine Translation, this requires an Encoder-Decoder neural network.
*   *Encoder:* Reads the massive 10-page document and compresses it into a single Context Vector.
*   *Decoder:* Generates a 1-paragraph summary conditioned on that Context Vector.

Modern models like **T5 (Text-To-Text Transfer Transformer)** and **BART** are heavily fine-tuned specifically for this task.

*Pros:* Highly coherent, human-like summaries that flow naturally.
*Cons:* Extremely computationally expensive. Prone to **Hallucinations** (generating facts that sound plausible but were never mentioned in the original document).

---

# 4. Evaluation Metrics (ROUGE)

How do we mathematically prove that a summary is "good"? We use the **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) metric.

To use ROUGE, you must have a "Reference Summary" written by a human expert.
You compare the Machine's Summary to the Human's Summary and count the overlapping N-Grams.

*   **ROUGE-1:** Measures the overlap of single words (Unigrams).
*   **ROUGE-2:** Measures the overlap of two-word phrases (Bigrams).
*   **ROUGE-L:** Measures the Longest Common Subsequence (how many words match *in order*, even if they are spread apart).

If the machine writes `"The president gave a speech"` and the human wrote `"The president delivered a speech"`, ROUGE-1 will be very high (they share 4 words), but ROUGE-2 will be slightly lower (because the bigrams "gave a" and "delivered a" do not match).

---

# 5. Industry Applications

*   **News Aggregation:** Apps like Google News automatically scrape thousands of long-form articles and generate 3-bullet-point summaries for users scrolling on their phones.
*   **Medical Records:** Automatically summarizing a patient's 10-year medical history into a single paragraph for an ER doctor who needs immediate context.
*   **Meeting Transcripts:** Recording a 1-hour Zoom meeting, converting the audio to text, and using Abstractive Summarization to automatically email the "Key Action Items" to all participants.

---

# 6. Key Takeaways

*   **Text Summarization** condenses long documents into short, informative summaries.
*   **Extractive Summarization** (like TextRank) scores and extracts exact sentences from the original text. It is safe but disjointed.
*   **Abstractive Summarization** (like T5 or BART) uses Encoder-Decoder models to write new sentences from scratch. It is fluent but prone to hallucinations.
*   **ROUGE** is the standard mathematical metric used to evaluate summaries by calculating N-Gram overlap against a human reference.

---

# 7. Next Topic

Throughout the last few lessons (Translation, QA, Summarization), we keep mentioning "Encoder-Decoder" models, "Sequence-to-Sequence" models, and "Transformers". 

These advanced Deep Learning architectures completely destroyed classical NLP pipelines in the late 2010s. In the next lesson, we will formally introduce the architecture that changed NLP forever: **The Transformer**.

[← Question Answering Systems](13-Question-Answering-Systems.md) | [Back to Index](README.md) | [Next Topic: Modern NLP With Transformers →](15-Modern-NLP-With_Transformers.md)
