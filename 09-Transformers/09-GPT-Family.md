# 09 - The GPT Family & Decoder Models

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Prerequisites**: 08-BERT | **Estimated Reading Time**: 25 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Autoregressive Generation](#2-autoregressive-generation)
3. [The Decoder-Only Architecture](#3-the-decoder-only-architecture)
4. [The Evolution of GPT (1 to 4)](#4-the-evolution-of-gpt-1-to-4)
5. [Scaling Laws & Emergent Abilities](#5-scaling-laws--emergent-abilities)
6. [Library Implementation (Hugging Face)](#6-library-implementation-hugging-face)
7. [Interview Questions](#7-interview-questions)
8. [Key Takeaways](#8-key-takeaways)
9. [Next Topic](#9-next-topic)

---

# 1. What Problem Does This Solve?

BERT (the Encoder) is incredible at reading. It can read a million documents and perfectly categorize them. But if you ask BERT to write a polite email to your boss, it will fail completely.

### 🟢 Beginner
To generate a story, you need an AI that can write one word, look at what it just wrote, and then guess the next word. It has to move strictly forward in time. You cannot use "bidirectional context" when writing, because the right side of the sentence literally hasn't been written yet!

### 🟡 Intermediate
To generate text, we need a **Causal Language Model**. We need an architecture that uses a strict triangular mask to prevent the Attention mechanism from looking into the future. OpenAI realized that the **Decoder** half of the Transformer was the perfect engine for this.

### 🔴 Advanced
**GPT (Generative Pre-trained Transformer)** proved that if you train a massive Decoder-only Transformer on a simple autoregressive objective (predicting the next token) across terabytes of internet data, the model implicitly learns a world model. It learns grammar, facts, reasoning, and logic purely as a byproduct of trying to guess the next word.

---

# 2. Autoregressive Generation

Autoregressive models generate text in a loop.

1.  **Input:** *"Once upon a"*
2.  **Model Process:** The model runs its matrix multiplications and outputs a probability distribution over the entire 50,000-word vocabulary.
3.  **Output:** It predicts *"time"* as the most likely next word.
4.  **The Loop:** We append *"time"* to the original input. The new input is *"Once upon a time"*.
5.  **Repeat:** The model processes the new input to predict the next word.

Because the model must process the *entire* growing sequence from scratch just to generate a single new word, text generation is incredibly computationally expensive and slow compared to classification. *(Modern systems use a technique called KV-Caching to speed this up, caching the past Key and Value vectors so they don't have to be recomputed).*

---

# 3. The Decoder-Only Architecture

OpenAI took the original 2017 Seq2Seq Transformer and made a radical simplification:
They deleted the Encoder entirely.
Because there is no Encoder, they also deleted the Cross-Attention layer in the Decoder.

The entire GPT architecture is just a stack of identical blocks containing:
1.  **Masked Multi-Head Self-Attention**
2.  **Feed Forward Network (FFN)**
3.  **Layer Normalization**

Because it uses **Masked** Attention, token $t$ is mathematically blocked from looking at token $t+1$. 

---

# 4. The Evolution of GPT (1 to 4)

The architecture of GPT hasn't actually changed much since 2018. What changed was the **Scale**.

### GPT-1 (2018)
*   **Parameters**: 117 Million
*   **Data**: 5 GB of books
*   **Achievement**: Proved that generative pre-training worked. It could generate somewhat coherent sentences.

### GPT-2 (2019)
*   **Parameters**: 1.5 Billion
*   **Data**: 40 GB of internet text (WebText)
*   **Achievement**: The model became so good at generating fake news that OpenAI initially refused to release the weights, sparking the modern AI safety debate.

### GPT-3 (2020)
*   **Parameters**: 175 Billion
*   **Data**: 570 GB of filtered internet text
*   **Achievement**: The breakthrough. At this scale, the model exhibited **Few-Shot Learning**. You didn't need to retrain the model to teach it a new task; you just gave it a few examples in the prompt, and it figured it out on the fly.

### GPT-4 (2023)
*   **Parameters**: Estimated ~1.8 Trillion (using a Mixture-of-Experts architecture)
*   **Data**: Petabytes of data, including images.
*   **Achievement**: Reached human-level performance on the Bar Exam, medical exams, and complex coding tasks.

---

# 5. Scaling Laws & Emergent Abilities

Why did OpenAI just keep making the same model bigger?

Because of **Scaling Laws**. Researchers discovered a smooth, mathematically predictable power-law relationship: as you increase the amount of compute, data, and parameters, the loss of the model drops predictably.

More surprisingly, as the models crossed certain parameter thresholds (e.g., 10 Billion, 100 Billion), they suddenly displayed **Emergent Abilities**—skills they were never explicitly trained to do.
- Small models could not do basic arithmetic.
- At 100B parameters, GPT-3 suddenly "learned" how to do 3-digit addition purely by reading text.

---

# 6. Library Implementation (Hugging Face)

Here is how to generate text using the GPT-2 architecture in Python.

```python
# pip install transformers torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Load the Tokenizer and the Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 2. Encode the prompt
prompt = "In a shocking turn of events, the AI decided to"
inputs = tokenizer(prompt, return_tensors="pt")

# 3. Generate Text
# max_length: Total length of output sequence
# do_sample=True: Uses probability sampling instead of just picking the most likely word (Greedy)
# temperature: Higher = more random/creative. Lower = more deterministic.
outputs = model.generate(
    inputs['input_ids'], 
    max_length=30, 
    do_sample=True, 
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

# 4. Decode the integer tokens back into text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated:", generated_text)
```

---

# 7. Interview Questions

### Beginner
**Q: What is the architectural difference between BERT and GPT?**
A: BERT is an Encoder-only model that uses bidirectional attention. GPT is a Decoder-only model that uses masked (causal) attention to prevent looking into the future.

### Intermediate
**Q: Explain Autoregressive Generation.**
A: Autoregressive generation is a process where the model predicts the next token, appends that token to the input sequence, and then feeds the new, longer sequence back into the model to predict the next token, looping until an End-of-Sequence (EOS) token is generated.

### Advanced
**Q: What are Scaling Laws in the context of LLMs?**
A: Scaling laws refer to the empirical observation that a language model's performance (cross-entropy loss) improves predictably as a power-law function of model size (parameters), dataset size, and training compute.

---

# 8. Key Takeaways

*   The **GPT family** utilizes a **Decoder-only** architecture.
*   Because it uses **Masked Attention**, it is a causal language model perfectly suited for **text generation**.
*   The fundamental breakthrough of GPT was proving that simply scaling up the size of a next-word prediction engine leads to **emergent reasoning** and few-shot learning capabilities.

---

# 9. Next Topic

We have talked extensively about models "predicting the next word". But neural networks don't actually read words like "dog" or "bank". They only read numbers. 

Before we can train an LLM, we have to chop the English language into mathematical pieces.

[← BERT & Encoder Models](08-BERT.md) | [Back to Index](README.md) | [Next Topic: Tokenization & Embeddings →](10-Tokenization-And_Embeddings.md)
