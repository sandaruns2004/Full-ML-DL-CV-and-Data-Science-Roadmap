# 13 - Question Answering Systems

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Prerequisites**: 12-Machine-Translation | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Extractive QA (Reading Comprehension)](#2-extractive-qa-reading-comprehension)
3. [Abstractive QA (Generative)](#3-abstractive-qa-generative)
4. [Open-Domain QA (Search + Reading)](#4-open-domain-qa-search--reading)
5. [Industry Applications](#5-industry-applications)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

If you want to know the capital of France, you go to Google and type "What is the capital of France?". Google doesn't just give you a list of 10 links to read. At the very top of the page, in a giant bold box, it simply says **"Paris"**.

### 🟢 Beginner
Search engines used to be "Document Retrieval" systems. They matched the keywords in your search to the keywords in a web page. Today, users don't want a document. They want an *answer*.

### 🟡 Intermediate
**Question Answering (QA)** is the NLP task of automatically answering a question asked in natural language. Instead of returning a document, the system must read the document, locate the specific fact, and return the exact answer.

### 🔴 Advanced
There are two distinct architectural approaches to QA. **Extractive QA** uses classification to highlight the exact sub-string in a provided text that contains the answer. **Abstractive QA** uses a Seq2Seq Language Model to write a completely new, conversational answer based on the facts in the text. 

---

# 2. Extractive QA (Reading Comprehension)

Extractive QA models are like high school students taking a Reading Comprehension test. You give the model a paragraph of text (the Context) and a Question. The model must highlight the answer *exactly as it is written* in the text.

*   *Context:* `"The Apollo 11 mission landed on the Moon in 1969. Neil Armstrong was the first to step onto the surface."`
*   *Question:* `"When did Apollo 11 land?"`
*   *Output:* `"1969"`

**How it works (Token Classification):**
Extractive QA is mathematically identical to Named Entity Recognition (NER). 
The Neural Network reads the paragraph and outputs two probabilities for every single token:
1.  Is this token the **Start** of the answer?
2.  Is this token the **End** of the answer?

The model scans the paragraph and assigns the highest "Start" probability to `1969`, and the highest "End" probability to `1969`. It returns that exact slice of text.

---

# 3. Abstractive QA (Generative)

Sometimes, the exact answer is not neatly written in a single sentence. It might be scattered across the document.

*   *Context:* `"John bought 3 apples on Monday. On Tuesday, he ate 1 apple. On Wednesday, he bought 5 more apples."`
*   *Question:* `"How many apples does John have left?"`

An Extractive QA model will fail here, because the exact string `"7"` does not exist in the text. 

**Abstractive QA** solves this. It uses an Encoder-Decoder architecture (just like Machine Translation). 
It encodes the Context and the Question into a dense vector, and the Decoder generates a brand new sentence from scratch: `"John currently has 7 apples."`

This is exactly how modern systems like ChatGPT work when you ask them questions about a document you uploaded.

---

# 4. Open-Domain QA (Search + Reading)

Extractive and Abstractive QA both assume you already *have* the document that contains the answer. 

But what if you ask an AI: *"Who won the Superbowl in 2012?"* without providing a document?

The system must perform **Open-Domain QA**. This requires a two-step pipeline:

1.  **The Retriever:** A fast search engine (using TF-IDF or Word Embeddings) searches Wikipedia and returns the top 5 articles most likely to contain the answer.
2.  **The Reader:** A QA model (Extractive or Abstractive) reads those 5 articles, extracts the specific answer, and presents it to the user.

This exact architecture—Retriever + Reader—is the foundation of **Retrieval-Augmented Generation (RAG)**, which is how modern enterprise AI systems securely answer questions about private company documents without hallucinating.

---

# 5. Industry Applications

*   **Intelligent FAQ Bots:** Instead of making customers scroll through a massive FAQ page, a QA bot allows them to type their problem in plain English and instantly returns the specific solution.
*   **Medical Assistance:** Doctors can ask a clinical QA system: *"What are the contraindications for combining Drug A and Drug B?"* The system retrieves the latest medical journals and highlights the exact paragraph discussing the interaction.
*   **Voice Assistants:** Siri and Alexa rely heavily on Open-Domain QA to answer trivia and factual questions on the fly.

---

# 6. Key Takeaways

*   **Question Answering (QA)** moves beyond document retrieval to provide exact answers to natural language questions.
*   **Extractive QA** highlights the exact sub-string in a provided text that contains the answer. It is a Token Classification task.
*   **Abstractive QA** generates a completely new sentence based on the facts in the text. It uses an Encoder-Decoder generation model.
*   **Open-Domain QA** combines a fast Search Engine (Retriever) with a precise QA model (Reader) to find answers in massive databases like Wikipedia.

---

# 7. Next Topic

Question Answering is essentially the process of aggressively condensing a document down into a single sentence answer.

What if we want to condense a 10-page document into a 1-page summary? In the next lesson, we will look at the NLP task of **Text Summarization**.

[← Machine Translation](12-Machine-Translation.md) | [Back to Index](README.md) | [Next Topic: Text Summarization →](14-Text-Summarization.md)
