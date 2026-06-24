# 08 - BERT & Encoder Models

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Prerequisites**: 06-Transformer-Architecture | **Estimated Reading Time**: 25 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Why BERT Was Created](#2-why-bert-was-created)
3. [Bidirectional Learning](#3-bidirectional-learning)
4. [Masked Language Modeling (MLM)](#4-masked-language-modeling-mlm)
5. [Next Sentence Prediction (NSP)](#5-next-sentence-prediction-nsp)
6. [Library Implementation (Hugging Face)](#6-library-implementation-hugging-face)
7. [Applications & Limitations](#7-applications--limitations)
8. [Interview Questions](#8-interview-questions)
9. [Key Takeaways](#9-key-takeaways)
10. [Next Topic](#10-next-topic)

---

# 1. What Problem Does This Solve?

If you want an AI to read a legal contract and extract the names of the signing parties (Named Entity Recognition), what architecture do you use?

### 🟢 Beginner
You don't need the AI to *write* a new contract (Generation). You just need it to deeply *read and comprehend* the existing contract. Using a full Translator (Encoder + Decoder) is completely unnecessary because you aren't translating anything.

### 🟡 Intermediate
Before 2018, researchers trained specific, custom LSTMs from scratch for every specific task. If you wanted a sentiment classifier, you built a Sentiment LSTM. If you wanted a QA bot, you built a QA LSTM. This was incredibly inefficient because every model had to relearn the basic rules of English grammar from scratch.

### 🔴 Advanced
**BERT (Bidirectional Encoder Representations from Transformers)**, introduced by Google in 2018, solved this through **Transfer Learning**. Google took just the **Encoder** half of the Transformer and trained it on the entire English Wikipedia (3.3 billion words) using a self-supervised objective. This resulted in a massive "Foundation Model" that deeply understood English. You could then download BERT and "fine-tune" it on just 1,000 legal contracts, and it would achieve state-of-the-art accuracy instantly.

---

# 2. Why BERT Was Created

BERT proved that **Language Understanding** could be pre-trained. 

Instead of treating "Sentiment Analysis" and "Question Answering" as completely different AI problems, BERT treated them as the exact same problem: *Understanding context*.

If an AI perfectly understands the English language, you only need to add a single tiny Linear layer to the top of it to turn it into a sentiment classifier.

---

# 3. Bidirectional Learning

The most important word in BERT is **Bidirectional**. 

Consider the sentence: *"He went to the bank to deposit his check."*

If you read this sentence left-to-right (like a standard RNN or GPT), when you hit the word **bank**, you only know *"He went to the..."* You don't know if he went to a river bank or a financial bank until you read the word "deposit" later. 

Because BERT is an **Encoder**, it uses unmasked Self-Attention. It looks at the entire sentence simultaneously. When processing the word "bank", BERT instantly looks *forward* at the word "deposit" and *backward* at the word "He". 

It processes context from both directions simultaneously. This creates infinitely richer word embeddings than models that only look backward.

---

# 4. Masked Language Modeling (MLM)

If you have 3 billion words of Wikipedia text, how do you train the model? There are no "labels". You can't do supervised learning.

Google invented a genius self-supervised task: **Masked Language Modeling**.
They took a sentence and randomly replaced 15% of the words with a `<MASK>` token.

**Input:** *"The dog [MASK] over the fence."*
**Target:** *"jumped"*

Because BERT is bidirectional, it looks at "The dog" and "over the fence" simultaneously, and uses its FFN to guess the missing word. 

By forcing the network to play billions of rounds of "Fill-in-the-blank", the network naturally learns grammar, facts, and logic.
- *"The capital of France is [MASK]"* $\to$ Learns facts.
- *"He [MASK] to the store"* $\to$ Learns verb tenses.

---

# 5. Next Sentence Prediction (NSP)

To help BERT understand the relationship between multiple sentences (crucial for Question Answering), Google added a second training task.

BERT is fed two sentences: A and B. It must output a binary classification: `IsNext` or `NotNext`.

*   **Input**: `[CLS]` The man went to the store. `[SEP]` He bought a gallon of milk. `[SEP]`
*   **Target**: `IsNext`

*   **Input**: `[CLS]` The man went to the store. `[SEP]` Penguins live in Antarctica. `[SEP]`
*   **Target**: `NotNext`

*(Note: Modern variations of BERT, like RoBERTa, actually removed the NSP task because researchers realized MLM alone was powerful enough if trained longer).*

---

# 6. Library Implementation (Hugging Face)

Today, nobody writes BERT from scratch. We use the Hugging Face `transformers` library, which contains pre-trained weights downloaded directly from Google.

Here is how we use BERT for Sequence Classification (e.g., Sentiment Analysis).

```python
# pip install transformers torch
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Load the pre-trained Tokenizer and the Model
# 'bert-base-uncased' is the standard small BERT model (all lowercase)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 2. Prepare the input text
text = "The new Transformer architecture is absolutely mind-blowing!"
inputs = tokenizer(text, return_tensors="pt")

# inputs contains 'input_ids' (the integer tokens) and 'attention_mask'
print("Input IDs:", inputs['input_ids'])

# 3. Pass through BERT
# BERT will automatically process the text and route the [CLS] token through a classifier head
with torch.no_grad():
    outputs = model(**inputs)

# 4. Get the predictions
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

print(f"Predicted Class: {predicted_class} (0=Negative, 1=Positive)")
```

---

# 7. Applications & Limitations

### Applications
*   **Text Classification**: Spam detection, sentiment analysis, topic tagging.
*   **Named Entity Recognition (NER)**: Finding names, dates, and companies in legal texts.
*   **Extractive Question Answering**: Highlighting the exact sentence in a Wikipedia article that answers a user's question.

### Limitations
*   **Cannot Generate Text**: Because BERT relies on bidirectional context, it cannot be used to write a story or act as a chatbot. If you try to predict the next word, it will fail because it hasn't seen the "right-side" context yet.
*   **Heavy Compute for Inference**: BERT is very large (110M to 340M parameters). Running inference on every user request can be expensive without distillation (e.g., DistilBERT).

---

# 8. Interview Questions

### Beginner
**Q: What does BERT stand for?**
A: Bidirectional Encoder Representations from Transformers.

### Intermediate
**Q: Explain Masked Language Modeling (MLM).**
A: MLM is a self-supervised training technique where 15% of the input words are randomly replaced with a `[MASK]` token. The model must use the surrounding bidirectional context to predict the original missing word, forcing it to learn language semantics.

### Advanced
**Q: Why does BERT prepend a `[CLS]` token to every input sequence?**
A: In Self-Attention, every token aggregates information from every other token. By the final layer, the `[CLS]` token's embedding has mathematically absorbed the holistic context of the entire sentence. Therefore, we can just grab the `[CLS]` token's vector and feed it into a linear layer for sequence-level classification tasks.

---

# 9. Key Takeaways

*   **BERT** is an **Encoder-only** architecture.
*   It utilizes **Bidirectional Attention**, meaning it reads the entire sentence at once (left-to-right and right-to-left).
*   It was trained using **Masked Language Modeling** (fill-in-the-blank) on massive datasets.
*   It excels at reading comprehension tasks like classification and extraction, but it **cannot generate text**.

---

# 10. Next Topic

BERT proved that the Encoder was incredible at reading. But what if we wanted an AI that was incredible at *writing*? 

While Google was ripping the Decoder off the Transformer to build BERT, a small startup called OpenAI was ripping the Encoder off to build something else.

[← Building A Transformer](07-Building-A-Transformer-Step-By-Step.md) | [Back to Index](README.md) | [Next Topic: The GPT Family →](09-GPT-Family.md)
