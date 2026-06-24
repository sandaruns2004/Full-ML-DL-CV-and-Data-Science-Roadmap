# 09 - Part-Of-Speech Tagging

> **Difficulty**: ⭐⭐☆☆☆ Beginner | **Prerequisites**: 08-Named-Entity-Recognition-NER | **Estimated Reading Time**: 15 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [What is a POS Tag?](#2-what-is-a-pos-tag)
3. [Why is this difficult? (Ambiguity)](#3-why-is-this-difficult-ambiguity)
4. [Dependency Parsing](#4-dependency-parsing)
5. [Code Implementation (spaCy)](#5-code-implementation-spacy)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

We have successfully extracted the *names* of things (NER). Now we need to extract the *actions* and the *descriptions*.

### 🟢 Beginner
If a customer writes a review saying: "The battery is terrible, but the screen is amazing."
We want the computer to automatically extract that `battery` is linked to `terrible`, and `screen` is linked to `amazing`. To do this, the computer must understand basic English grammar. It must know which words are Nouns, which are Adjectives, and which are Verbs.

### 🟡 Intermediate
**Part-Of-Speech (POS) Tagging** is the process of labeling each word in a text with its corresponding grammatical part of speech. It is a fundamental step in building **Knowledge Graphs** and extracting relationships from text.

### 🔴 Advanced
Like NER, POS Tagging is a Token Classification task. A model must scan the sequence and assign a grammatical label to every token. Because many English words can act as both Nouns and Verbs depending entirely on context, this requires sequence models (Hidden Markov Models in the 90s, LSTMs in the 2010s, and Transformers today).

---

# 2. What is a POS Tag?

The Universal POS tag set includes:
*   `NOUN`: book, girl, cat, tree
*   `VERB`: run, eat, jump, is
*   `ADJ`: big, old, green, terrible
*   `ADV`: very, tomorrow, quickly
*   `PRON`: I, you, he, she, they
*   `ADP`: in, to, during (Adpositions / Prepositions)
*   `CONJ`: and, or, but (Conjunctions)

When we pass a sentence into a POS Tagger, it outputs a tuple for every word:
`"The red car drives fast."`
$\to$ `[("The", DET), ("red", ADJ), ("car", NOUN), ("drives", VERB), ("fast", ADV)]`

---

# 3. Why is this difficult? (Ambiguity)

Consider the word **"book"**.

1.  "Please read the **book**." $\to$ Here, "book" is a `NOUN`.
2.  "Please **book** the flight." $\to$ Here, "book" is a `VERB`.

A simple dictionary lookup fails completely. The tagger must look at the surrounding words. In sentence 2, "book" follows the word "Please" and precedes the determiner "the", which strongly implies an action is being requested.

---

# 4. Dependency Parsing

Knowing that "car" is a Noun and "red" is an Adjective is useful. But how does the computer know that "red" is specifically describing the "car", and not something else in the sentence?

This is solved by **Dependency Parsing**. 

Dependency Parsing draws a literal tree connecting the words in a sentence based on their grammatical relationships. It finds the "Root" verb of the sentence, and then draws directional arrows to the subjects and objects.

By combining POS Tagging with Dependency Parsing, you can write Python scripts that say:
`Find every NOUN in the document. Find the ADJECTIVES that have a dependency arrow pointing to that NOUN. Extract them as a pair.`

This allows you to automatically extract phrases like `("battery", "terrible")` and `("screen", "amazing")` from millions of reviews without human intervention.

---

# 5. Code Implementation (spaCy)

`spaCy` performs Tokenization, POS Tagging, NER, and Dependency Parsing all at the same time, under the hood, the moment you pass a string into the `nlp()` object.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "The red car drives very fast."
doc = nlp(text)

print(f"{'Word':10} | {'POS':8} | {'Dependency'}")
print("-" * 35)

for token in doc:
    # token.pos_ gives the Part of Speech
    # token.dep_ gives the Dependency tag
    # token.head.text gives the word it is attached to
    print(f"{token.text:10} | {token.pos_:8} | {token.dep_} (attached to '{token.head.text}')")
```

Output:
```text
Word       | POS      | Dependency
-----------------------------------
The        | DET      | det (attached to 'car')
red        | ADJ      | amod (attached to 'car')
car        | NOUN     | nsubj (attached to 'drives')
drives     | VERB     | ROOT (attached to 'drives')
very       | ADV      | advmod (attached to 'fast')
fast       | ADV      | advmod (attached to 'drives')
.          | PUNCT    | punct (attached to 'drives')
```
*Notice how the computer correctly figured out that "red" is attached to "car", and "fast" is attached to "drives"!*

---

# 6. Key Takeaways

*   **POS Tagging** labels every word with its grammatical category (Noun, Verb, Adjective).
*   It is heavily affected by **Ambiguity** (words having different tags depending on context), requiring sequence-aware models.
*   **Dependency Parsing** draws a tree connecting words based on their grammatical relationships.
*   Combining POS tags and Dependency Parsing allows us to extract highly specific relationships and facts from unstructured text.

---

# 7. Next Topic

Up until now, we have focused entirely on *understanding* and *extracting* information from text that already exists. We have been doing Discriminative AI.

But what if we want the computer to *write its own text*? What if we want it to write a poem, or finish our sentences in an email, or generate a brand new news article? 

To do that, we must move into Generative AI, and we must build a **Language Model**.

[← Named Entity Recognition](08-Named-Entity-Recognition-NER.md) | [Back to Index](README.md) | [Next Topic: Language Models →](10-Language-Models.md)
