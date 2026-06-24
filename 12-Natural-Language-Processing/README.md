# 🗣️ Module 12: Natural Language Processing (NLP)

Welcome to **Module 12: Natural Language Processing**. 

Computers natively speak in numbers. Humans speak in messy, sarcastic, unstructured strings of text. This module is the bridge between the two. 

Here, you will learn how to teach a machine to read, understand, classify, translate, and generate human language. You will start with classical rule-based algorithms from the 1990s and work your way up to the modern Transformer architectures that power systems like ChatGPT.

---

## 📚 Curriculum Structure

This module follows a strict **Beginner $\to$ Intermediate $\to$ Advanced** progression. NLP is cumulative; you must understand how to tokenize text before you can understand how an LLM works.

### The Foundations (Text $\to$ Math)
1.  [**01 - Introduction To NLP**](01-Introduction-To-NLP.md) - Why human language is mathematically difficult (Ambiguity, Context).
2.  [**02 - Text Preprocessing**](02-Text-Preprocessing.md) - Cleaning, Lowercasing, Stopwords, Stemming, and Lemmatization.
3.  [**03 - Tokenization**](03-Tokenization.md) - Word, Character, and Subword Tokenization (BPE).
4.  [**04 - Feature Extraction**](04-Feature-Extraction.md) - Bag of Words, N-Grams, and TF-IDF vectors.
5.  [**05 - Word Embeddings**](05-Word-Embeddings.md) - Word2Vec, GloVe, and dense semantic latent spaces.

### Discriminative NLP (Reading & Categorizing)
6.  [**06 - Text Classification**](06-Text-Classification.md) - Categorizing documents using Naive Bayes and LSTMs.
7.  [**07 - Sentiment Analysis**](07-Sentiment-Analysis.md) - Extracting emotion and polarity (VADER vs Deep Learning).
8.  [**08 - Named Entity Recognition (NER)**](08-Named-Entity-Recognition-NER.md) - Extracting facts, names, and organizations using IOB tagging.
9.  [**09 - Part-Of-Speech Tagging**](09-Part-Of-Speech-Tagging.md) - Understanding English grammar and dependency trees.
10. [**11 - Topic Modeling**](11-Topic-Modeling.md) - Unsupervised clustering of massive datasets using LDA and NMF.

### Generative NLP (Writing & Translating)
11. [**10 - Language Models**](10-Language-Models.md) - N-Grams and the Autoregressive generation loop.
12. [**12 - Machine Translation**](12-Machine-Translation.md) - Rule-Based, Statistical, and Seq2Seq Neural translation.
13. [**13 - Question Answering Systems**](13-Question-Answering-Systems.md) - Extractive vs Abstractive QA, and RAG architectures.
14. [**14 - Text Summarization**](14-Text-Summarization.md) - Extractive vs Abstractive summarization, evaluated using ROUGE.

### The Modern Era & Production
15. [**15 - Modern NLP With Transformers**](15-Modern-NLP-With_Transformers.md) - Self-Attention, BERT, GPT, and T5.
16. [**16 - NLP In Production**](16-NLP-In-Production.md) - Model serving, Latency, Data Drift, and engineering failure cases.

---

## 💻 Labs & Notebooks

Inside the `notebooks/` directory, you will find interactive Jupyter notebooks designed to let you visualize the NLP pipelines and algorithms discussed in the theory files:

*   `NLP_Fundamentals_Lab.ipynb`
*   `TFIDF_And_Text_Features_Lab.ipynb`
*   `Word_Embeddings_Explorer.ipynb`
*   `Sentiment_Analysis_Lab.ipynb`
*   `Named_Entity_Recognition_Lab.ipynb`
*   `Topic_Modeling_Lab.ipynb`
*   `NLP_Pipeline_Benchmark.ipynb`

---

## 🚀 Projects

Inside the `projects/` directory, you will build 10 end-to-end applications demonstrating these architectures in production-style environments:

1.  `01-Spam-Email-Detection-System`
2.  `02-Sentiment-Analysis-Platform`
3.  `03-News-Topic-Classifier`
4.  `04-Resume-Parser`
5.  `05-Intelligent-FAQ-System`
6.  `06-Research-Paper-Analyzer`
7.  `07-News-Summarization-System`
8.  `08-NLP-Search-Engine`
9.  `09-Customer-Feedback-Analytics-Dashboard`
10. `10-End-to-End-NLP-Pipeline`

Happy Parsing!
