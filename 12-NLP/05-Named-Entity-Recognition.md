# 🏷️ Named Entity Recognition (NER) & Sequence Labeling

> **Prerequisites**: Text Classification, LSTMs | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [What is Sequence Labeling?](#1-what-is-sequence-labeling)
2. [The BIO Tagging Format](#2-the-bio-tagging-format)
3. [Conditional Random Fields (CRFs)](#3-conditional-random-fields-crfs)
4. [The BiLSTM-CRF Architecture](#4-the-bilstm-crf-architecture)
5. [Library Implementation (SpaCy & Hugging Face)](#5-library-implementation-spacy--hugging-face)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. What is Sequence Labeling?

Standard Text Classification assigns a single label to an entire sentence (e.g., Sentiment = Positive). 
**Sequence Labeling** assigns a discrete label to *every single word* in a sentence.

The most famous Sequence Labeling task is **Named Entity Recognition (NER)**. The goal of NER is to locate and classify proper nouns in text into predefined categories such as Person names, Organizations, Locations, Medical Codes, Time expressions, etc.

**Example Task:**
> *Sentence*: "Tim Cook announced a new Apple store in Paris yesterday."
> *Output*: 
> - Tim Cook $\rightarrow$ `PERSON`
> - Apple $\rightarrow$ `ORGANIZATION`
> - Paris $\rightarrow$ `LOCATION`
> - yesterday $\rightarrow$ `DATE`
> - announced, a, new, store, in $\rightarrow$ `O` (Outside / No Entity)

---

## 2. The BIO Tagging Format

How do we represent multi-word entities (like "Tim Cook" or "New York City") in a dataset? 
If we just label "New", "York", and "City" all as `LOCATION`, the model wouldn't know if it's one city called "New York City", or three different locations next to each other.

We solve this using the **BIO Format** (Begin, Inside, Outside).

Every entity category is split into two tags: `B-` (Begin) and `I-` (Inside).
Words that do not belong to any entity are tagged `O` (Outside).

**Example:**
| Word | Tag | Explanation |
|------|-----|-------------|
| Tim | `B-PER` | Beginning of a Person entity |
| Cook | `I-PER` | Inside (continuation) of a Person entity |
| flew | `O` | Outside |
| to | `O` | Outside |
| New | `B-LOC` | Beginning of a Location entity |
| York | `I-LOC` | Inside a Location entity |
| City | `I-LOC` | Inside a Location entity |

By predicting these BIO tags, a model can perfectly extract multi-word entities and handle adjacent entities of the same type.

---

## 3. Conditional Random Fields (CRFs)

If you just use a standard LSTM to predict BIO tags, it will make dumb mistakes. For example, it might predict `[B-PER, I-LOC]`. This is completely impossible! An `Inside-Location` tag cannot follow a `Begin-Person` tag.

A **Conditional Random Field (CRF)** is a classical machine learning model that learns the *transition probabilities* between tags. 

A CRF maintains a $K \times K$ transition matrix (where $K$ is the number of possible tags). During training, it learns that:
- The probability of transitioning from `B-PER` to `I-PER` is very high.
- The probability of transitioning from `B-PER` to `I-LOC` is exactly $0$.
- The probability of transitioning from `O` to `I-PER` is exactly $0$.

Instead of greedily picking the best tag for each word independently, the CRF calculates the probability of the *entire sequence of tags* and uses the **Viterbi Algorithm** (Dynamic Programming) to find the globally optimal, logically sound sequence of tags.

---

## 4. The BiLSTM-CRF Architecture

For years, the undisputed state-of-the-art for NER was the **BiLSTM-CRF** architecture (Huang et al., 2015).

1. **Embedding Layer**: Converts words into vectors. (Often combines Word2Vec with Character-level CNN embeddings to handle typos).
2. **Bi-directional LSTM**: Reads the sentence forwards and backwards, outputting a context-aware hidden state for every single word.
3. **Linear Projection**: Maps the LSTM hidden states to the number of BIO tags (e.g., 9 tags). These are called the *emission scores*.
4. **CRF Layer**: Takes the emission scores from the LSTM and applies the learned transition matrix. Uses the Viterbi algorithm to output the final sequence of tags.

*(Note: While Transformers like BERT have largely replaced BiLSTM-CRFs, modern NER models still often stick a CRF layer on top of BERT!)*

---

## 5. Library Implementation (SpaCy & Hugging Face)

Nobody writes BiLSTM-CRFs from scratch anymore. The modern NLP ecosystem offers incredibly powerful out-of-the-box NER models.

### SpaCy (Lightning Fast, Production Ready)
SpaCy uses highly optimized CNN architectures. It's the industry standard for production pipelines where speed is critical.

```python
import spacy

# Load the small English model (must run `python -m spacy download en_core_web_sm` first)
nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion."
doc = nlp(text)

# Iterate over detected entities
print(f"{'Entity':<15} | {'Label':<15} | {'Description'}")
print("-" * 50)
for ent in doc.ents:
    print(f"{ent.text:<15} | {ent.label_:<15} | {spacy.explain(ent.label_)}")
```

*Output:*
```text
Entity          | Label           | Description
--------------------------------------------------
Apple           | ORG             | Companies, agencies, institutions, etc.
U.K.            | GPE             | Countries, cities, states
$1 billion      | MONEY           | Monetary values, including unit
```

### Hugging Face (State-of-the-Art Transformers)
If you need maximum accuracy and don't care about CPU inference speed, use a pre-trained Transformer.

```python
from transformers import pipeline

# Load a pre-trained BERT model specifically fine-tuned for NER
ner_pipeline = pipeline("ner", grouped_entities=True)

text = "My name is Sarah and I work at Google in Mountain View, California."
results = ner_pipeline(text)

for entity in results:
    print(f"[{entity['entity_group']}] {entity['word']} (Score: {entity['score']:.2f})")
```

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Redacting PII**: Write a Python script that takes a folder of text documents, runs SpaCy NER to find all `PERSON`, `ORG`, and `LOC` entities, and replaces them with `[REDACTED]` to anonymize the dataset.
- 🟡 **Custom Medical NER**: Use the `spacy` training API to fine-tune a blank NER model. Create a small dataset using the `DocBin` format with custom BIO tags like `B-DISEASE` and `B-MEDICATION` to extract medical info from doctor's notes.

### What's Next
| Next | Why |
|------|-----|
| [Machine Translation](./06-Machine-Translation.md) | We've looked at assigning a label to a sequence. But what if we want to turn one sequence into an entirely new sequence (like English to French)? We need Encoder-Decoder Seq2Seq models. |

---

[← Text Classification](./04-Text-Classification.md) | [Back to Index](../README.md) | [Next: Machine Translation →](./06-Machine-Translation.md)
