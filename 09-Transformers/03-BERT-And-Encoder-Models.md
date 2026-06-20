# 📖 BERT & Encoder-Only Models

> **Prerequisites**: Transformer Architecture | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Philosophy of Encoder-Only Models](#1-the-philosophy-of-encoder-only-models)
2. [Masked Language Modeling (MLM)](#2-masked-language-modeling-mlm)
3. [Next Sentence Prediction (NSP)](#3-next-sentence-prediction-nsp)
4. [The [CLS] Token & Fine-Tuning](#4-the-cls-token--fine-tuning)
5. [Code Example: Fine-Tuning with HuggingFace](#5-code-example-fine-tuning-with-huggingface)
6. [BERT Variants (RoBERTa, ALBERT, DistilBERT)](#6-bert-variants-roberta-albert-distilbert)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Philosophy of Encoder-Only Models

The original Transformer was built to translate text. But what if we just want a Neural Network to *understand* text deeply, so we can use it to classify emails as spam, extract names from a document, or answer multiple-choice questions?

In 2018, Google released **BERT** (Bidirectional Encoder Representations from Transformers). 

They took the original Transformer, **threw away the Decoder completely**, and stacked 12 Encoder layers on top of each other. 
Because there is no Decoder, there is no Masked Self-Attention. The model is **Bidirectional** — when processing a word, it can look at *both* the left context (previous words) and the right context (future words) simultaneously.

---

## 2. Masked Language Modeling (MLM)

How do you train a model to "understand" language without human labels? You use the text itself as the label (Self-Supervised Learning).

**The MLM Task**:
1. Take a sentence from Wikipedia: *"The chef cooked a delicious meal."*
2. Randomly replace 15% of the words with a special `[MASK]` token.
   - *"The `[MASK]` cooked a delicious `[MASK]`."*
3. Pass this masked sentence through the 12 Encoder layers.
4. The network outputs a probability distribution over the entire English vocabulary for those masked positions.
5. Calculate Cross-Entropy Loss against the original words (*"chef"*, *"meal"*).

Because the network must guess the missing words, it is forced to deeply learn grammar, syntax, and world facts.

---

## 3. Next Sentence Prediction (NSP)

To force BERT to understand relationships *between* sentences (useful for Question Answering tasks), Google trained it on a second task simultaneously: **NSP**.

1. Feed the network two sentences, separated by a special `[SEP]` token.
   - 50% of the time, the two sentences are actually sequential in the original text.
   - 50% of the time, the second sentence is a completely random sentence from another Wikipedia article.
2. The network must output a binary prediction: `IsNext` or `NotNext`.

*(Note: Later models like RoBERTa proved that NSP wasn't actually necessary, and removing it often improved performance).*

---

## 4. The [CLS] Token & Fine-Tuning

How do we actually use BERT for a downstream task like Spam Detection?

When feeding text into BERT, we always prepend a special Classification Token: `[CLS]`.
Because self-attention allows all tokens to communicate, the `[CLS]` token acts as an aggregator. By the time it reaches the 12th layer, the `[CLS]` embedding contains a compressed mathematical summary of the *entire* sentence.

**The Fine-Tuning Recipe**:
1. Download a pre-trained BERT model.
2. Freeze (or use a tiny learning rate for) the 12 Encoder layers.
3. Attach a randomly initialized Feed Forward Neural Network (a classification head) directly to the output of the `[CLS]` token.
4. Train the model on your small, labeled dataset (e.g., 500 spam emails and 500 safe emails).

---

## 5. Code Example: Fine-Tuning with HuggingFace

The HuggingFace `transformers` library makes fine-tuning BERT remarkably simple. Here's how you can fine-tune a model for Sequence Classification (e.g., Sentiment Analysis).

```python
# pip install transformers datasets torch
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Load Dataset & Tokenizer
dataset = load_dataset("imdb") # Movie reviews (positive/negative)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    # Padding and truncation ensure all sequences are the same length
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 2. Load Pre-trained Model with Classification Head
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 3. Setup Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5, # Small learning rate for fine-tuning
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 4. Initialize Trainer and Fine-Tune
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(1000)), # Subset for speed
    eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(500)),
)

trainer.train()

# 5. Inference
inputs = tokenizer("This movie was absolutely amazing!", return_tensors="pt")
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions) # Output probabilities for Negative and Positive
```

---

## 6. BERT Variants (RoBERTa, ALBERT, DistilBERT)

The original BERT was a massive breakthrough, but researchers quickly optimized it:

- **RoBERTa (Robustly Optimized BERT Approach)**: Meta (Facebook) found that BERT was heavily under-trained. RoBERTa removed the NSP task, used dynamic masking (changing the masks every epoch), and trained on 10x more data. It crushed original BERT on every benchmark.
- **ALBERT (A Lite BERT)**: Used cross-layer parameter sharing (forcing all 12 layers to share the exact same weights). This drastically reduced the memory footprint while maintaining accuracy.
- **DistilBERT**: Hugging Face used **Knowledge Distillation**. They trained a small 6-layer BERT to mimic the output probabilities of a massive 12-layer BERT. Result: 40% smaller, 60% faster, retains 97% of the original accuracy.

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Spam Classifier**: Go to Kaggle and download the SMS Spam Collection Dataset. Use the Hugging Face `transformers` library to load `distilbert-base-uncased`. Pass the SMS messages through the model, extract the `[CLS]` token embedding, and train a simple Scikit-Learn Logistic Regression model on top of it to classify spam!
- 🟡 **Named Entity Recognition**: Modify the HuggingFace script to use `AutoModelForTokenClassification` and fine-tune BERT on the CoNLL-2003 dataset to extract people, locations, and organizations.

### What's Next
| Next | Why |
|------|-----|
| [GPT & Decoder Models](./04-GPT-And-Decoder-Models.md) | BERT is great at reading, but terrible at writing. What happens if we throw away the Encoder and only keep the Decoder? |

---

[← Transformer Architecture](./02-Transformer-Architecture.md) | [Back to Index](../README.md) | [Next: GPT And Decoder Models →](./04-GPT-And-Decoder-Models.md)
