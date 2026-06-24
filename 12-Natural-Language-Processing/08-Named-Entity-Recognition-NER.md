# 08 - Named Entity Recognition (NER)

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Prerequisites**: 06-Text-Classification | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [What is an Entity?](#2-what-is-an-entity)
3. [How NER Works (Token Classification)](#3-how-ner-works-token-classification)
4. [The IOB Tagging Format](#4-the-iob-tagging-format)
5. [Code Implementation (spaCy)](#5-code-implementation-spacy)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

Imagine you are a recruiter who receives 5,000 PDF resumes a day. 

### 🟢 Beginner
You don't want to know if the resume is "Positive" or "Negative". You don't want to categorize the resume as "Sports" or "Politics". You want to automatically extract specific pieces of information: The person's Name, their University, the Companies they worked for, and their Years of Experience.

### 🟡 Intermediate
To do this, we need an algorithm that can scan a sentence and locate the "proper nouns" and specific categories of information. This is called **Information Extraction**. The foundational task of Information Extraction is **Named Entity Recognition (NER)**.

### 🔴 Advanced
NER is significantly more complex than standard Text Classification. In standard classification, one document = one label. In NER, every single *token* in the document gets its own label. This requires Sequence-to-Sequence models (like LSTMs, CRFs, or Transformers) that understand context. For example, the word "Apple" could be a fruit (No Entity) or a company (`ORG`), depending entirely on the surrounding words.

---

# 2. What is an Entity?

An **Entity** is a real-world object, such as a person, location, organization, or product, that can be denoted with a proper name.

Standard NER categories include:
*   `PERSON`: People, including fictional. (e.g., *Elon Musk*, *Harry Potter*)
*   `ORG`: Companies, agencies, institutions. (e.g., *Google*, *FBI*, *Stanford*)
*   `GPE`: Geopolitical Entities (countries, cities, states). (e.g., *France*, *New York*)
*   `DATE`: Absolute or relative dates or periods. (e.g., *2023*, *next Tuesday*)
*   `MONEY`: Monetary values, including unit. (e.g., *$10,000*, *fifty euros*)

---

# 3. How NER Works (Token Classification)

Unlike Sentiment Analysis which outputs a single label for a whole sentence, NER outputs a label for **every single word**.

*   *Input:* `Tim Cook is the CEO of Apple Inc. in California.`
*   *Output:*
    *   `Tim` $\to$ `PERSON`
    *   `Cook` $\to$ `PERSON`
    *   `is` $\to$ `O` (Outside / Not an entity)
    *   `the` $\to$ `O`
    *   `CEO` $\to$ `O`
    *   `of` $\to$ `O`
    *   `Apple` $\to$ `ORG`
    *   `Inc.` $\to$ `ORG`
    *   `in` $\to$ `O`
    *   `California` $\to$ `GPE`

---

# 4. The IOB Tagging Format

Notice in the example above that `Tim` and `Cook` are both labeled as `PERSON`. How does the computer know if this is two different people ("Tim" and "Cook") or one person with a first and last name ("Tim Cook")?

To solve this, NLP engineers use the **IOB (Inside, Outside, Beginning)** format.
*   `B-` denotes the **Beginning** of an entity.
*   `I-` denotes the **Inside** (continuation) of an entity.
*   `O` denotes **Outside** (not an entity).

Let's re-tag the sentence using IOB:
*   `Tim` $\to$ `B-PERSON`
*   `Cook` $\to$ `I-PERSON` (This tells the system "Cook" belongs to the previous "Tim")
*   `is` $\to$ `O`
*   `the` $\to$ `O`
...and so on.

---

# 5. Code Implementation (spaCy)

`spaCy` has a phenomenally fast and accurate pre-trained NER model out of the box.

```python
import spacy

# Load the pre-trained English pipeline
nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion."

# Process the text
doc = nlp(text)

# Iterate over the detected entities
for ent in doc.ents:
    print(f"Text: {ent.text:15} | Label: {ent.label_}")

'''
Output:
Text: Apple           | Label: ORG
Text: U.K.            | Label: GPE
Text: $1 billion      | Label: MONEY
'''
```

### Visualizing Entities
If you run this in a Jupyter Notebook, spaCy provides `displacy`, a beautiful visualizer that highlights entities directly in the text with colored backgrounds.

```python
from spacy import displacy
displacy.render(doc, style="ent")
```

---

# 6. Key Takeaways

*   **Information Extraction** pulls hard facts out of unstructured text.
*   **Named Entity Recognition (NER)** is the task of locating and classifying named entities into pre-defined categories (like `PERSON`, `ORG`, `GPE`).
*   NER is a **Token Classification** task; the model must output a prediction for every single word in the sequence.
*   The **IOB Format** is used to tell the model where a multi-word entity begins and ends.
*   `spaCy` is the industry standard for production-grade, fast NER on classical text.

---

# 7. Next Topic

We now know how to extract the names of people and companies. But what if we want to extract the *actions* they are taking? Or the *descriptions* of the products? 

To do that, we have to teach the computer the fundamental grammar of the English language. In the next lesson, we will look at **Part-Of-Speech Tagging** (identifying Nouns, Verbs, and Adjectives).

[← Sentiment Analysis](07-Sentiment-Analysis.md) | [Back to Index](README.md) | [Next Topic: Part-Of-Speech Tagging →](09-Part-Of-Speech-Tagging.md)
