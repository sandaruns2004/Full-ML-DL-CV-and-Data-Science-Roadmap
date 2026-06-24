# 13 - Sequence Model Evaluation

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Prerequisites**: 09-Sequence-To-Sequence-Models | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Core Concepts](#3-core-concepts)
4. [Mathematics: BLEU, ROUGE, and Perplexity](#4-mathematics-bleu-rouge-and-perplexity)
5. [Library Implementation](#5-library-implementation)
6. [Failure Cases](#6-failure-cases)
7. [Interview Questions](#7-interview-questions)
8. [Key Takeaways](#8-key-takeaways)
9. [Next Topic](#9-next-topic)

---

# 1. What Problem Does This Solve?

Evaluating an image classifier is easy: it either guessed "Cat" or it didn't. Accuracy is simple. But how do you evaluate a translated sentence?

### 🟢 Beginner
If the correct translation is *"The cat sat on the mat"*, and the AI generates *"On the mat sat the cat"*, is the AI wrong? Technically, the word order is completely different, so standard "Accuracy" would say the AI scored 0%. But to a human, the translation is perfectly fine! We need metrics that understand fuzzy text matching.

### 🟡 Intermediate
Sequence generation tasks (like Translation, Summarization, and Chatbots) suffer from the **One-to-Many Problem**. There are dozens of valid ways to express the same thought. We cannot use strict 1:1 cross-entropy classification accuracy to evaluate the final generated sequence. 

### 🔴 Advanced
We need objective, automated statistical metrics to evaluate the overlap of n-grams between a generated candidate sequence and a set of reference sequences. Furthermore, during training, we need an intrinsic mathematical metric to evaluate how "confused" a language model is regarding its probability distributions.

---

# 2. Intuition

Imagine you are a teacher grading two students on an essay summarization task.
- Student A copies exact phrases from the original text but jumbles the order slightly.
- Student B writes a completely original, beautiful summary using synonyms.

An automated script grading based on "exact word matches" will give Student A an A+, and Student B an F. 
To fix this, we grade based on **N-grams** (chunks of 1, 2, or 3 words). If Student A has the exact 1-word matches, but terrible 3-word matches (bad fluency), their score drops. We evaluate based on overlapping chunks of meaning.

---

# 3. Core Concepts

### 🟢 BLEU (Bilingual Evaluation Understudy)
The gold standard for evaluating **Machine Translation**. It counts how many n-grams in the generated translation actually appear in the human reference translation. It penalizes translations that are too short.

### 🟡 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
The gold standard for evaluating **Text Summarization**. While BLEU focuses on Precision (did the words the AI generated actually exist in the reference?), ROUGE focuses on Recall (did the AI capture all the important words that were in the reference?).

### 🔴 Perplexity (PPL)
An intrinsic metric used during the training of Language Models. It literally measures how "surprised" or "perplexed" the model is by the validation dataset. A lower perplexity means the model is confidently predicting the correct next words.

---

# 4. Mathematics: BLEU, ROUGE, and Perplexity

**1. BLEU Score (Precision Focus)**
BLEU looks at the intersection of n-grams.
$$P_n = \frac{\sum \text{Count}_{\text{match}}(n\text{-gram})}{\sum \text{Count}_{\text{total}}(n\text{-gram})}$$
*Brevity Penalty (BP)*: If the AI just outputs one correct word ("The") and stops, $P_1$ is 100%. BP mathematically punishes the score if the generated length is shorter than the reference length.

**2. ROUGE-N (Recall Focus)**
$$ROUGE_N = \frac{\sum \text{Count}_{\text{match}}(n\text{-gram})}{\sum \text{Count}_{\text{reference}}(n\text{-gram})}$$
If the human summary had 10 important bigrams, and the AI only generated 2 of them, the ROUGE-2 score is low (20%).

**3. Perplexity**
Mathematically, it is the exponentiation of the Cross-Entropy Loss ($H$).
$$\text{Perplexity} = e^{H(p, q)}$$
If an AI is guessing the next word out of a vocabulary of 50,000 words, a Perplexity of `10` means the AI is so confident that it has narrowed its guess down to a 1-in-10 coin flip.

---

# 5. Library Implementation

Instead of calculating these manually, we use Hugging Face's `evaluate` library.

```python
# pip install evaluate
import evaluate

# 1. Calculate BLEU for Translation
bleu = evaluate.load("bleu")
predictions = ["the cat sat on the mat"]
references = [["the cat is on the mat", "there is a cat on the mat"]]

results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU Score: {results['bleu']:.4f}")

# 2. Calculate ROUGE for Summarization
rouge = evaluate.load("rouge")
predictions = ["cat sat on mat"]
references = [["the cat sat on the mat"]]

results = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-1: {results['rouge1']:.4f}")
```

---

# 6. Failure Cases

BLEU and ROUGE are fast, but they are fundamentally flawed.
**They only look at exact string matching.**

If the reference is: *"The movie was fantastic."*
And the AI outputs: *"The film was wonderful."*

BLEU and ROUGE will give the AI a terrible score because "film" does not exactly match "movie". They do not understand synonyms or semantics. 
This is why modern evaluation is shifting towards **LLM-as-a-Judge**, where we ask GPT-4 to read the two sentences and grade them on a scale of 1-10 based on semantic similarity.

---

# 7. Interview Questions

### Beginner
**Q: Why can't we use Accuracy to evaluate Machine Translation?**
A: Because there are multiple valid ways to translate the same sentence. Accuracy expects a strict 1:1 positional match, which would incorrectly penalize valid translations with different word orders or synonyms.

### Intermediate
**Q: Explain the difference between BLEU and ROUGE.**
A: BLEU focuses on Precision (are the generated words present in the reference?), making it ideal for Translation where you don't want the AI to hallucinate extra details. ROUGE focuses on Recall (did the generated text capture all the words in the reference?), making it ideal for Summarization where capturing the core facts is paramount.

### Advanced
**Q: What does a Perplexity of 1 mean?**
A: A Perplexity of 1 means the Cross-Entropy loss is 0. The model predicts the next word with 100% confidence and is never wrong. It is mathematically the best possible score a language model can achieve.

---

# 8. Key Takeaways

* Sequence generation requires fuzzy evaluation metrics.
* **BLEU** evaluates Translation by measuring n-gram precision.
* **ROUGE** evaluates Summarization by measuring n-gram recall.
* **Perplexity** evaluates Language Models by measuring "surprise" (lower is better).
* All n-gram metrics fail to understand synonyms, leading to the rise of AI-assisted semantic grading.

---

# 9. Next Topic

We now know how to train, architect, and evaluate sequence models. But even the best Transformer with the best BLEU score will occasionally fail catastrophically in production. Let's examine why.

[← Time Series Forecasting](12-Time-Series-Forecasting.md) | [Back to Index](README.md) | [Next Topic: Common Failure Cases →](14-Common-Failure-Cases.md)
