# 11 - Transformer Training & Fine-Tuning

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Prerequisites**: 09-GPT-Family | **Estimated Reading Time**: 25 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Phase 1: Pre-Training (The Foundation)](#2-phase-1-pre-training-the-foundation)
3. [Phase 2: Supervised Fine-Tuning (SFT)](#3-phase-2-supervised-fine-tuning-sft)
4. [Phase 3: RLHF (Alignment)](#4-phase-3-rlhf-alignment)
5. [Parameter-Efficient Fine-Tuning (LoRA)](#5-parameter-efficient-fine-tuning-lora)
6. [Library Implementation (PEFT)](#6-library-implementation-peft)
7. [Interview Questions](#7-interview-questions)
8. [Key Takeaways](#8-key-takeaways)
9. [Next Topic](#9-next-topic)

---

# 1. What Problem Does This Solve?

If you download the core GPT-3 foundation model from 2020 and type: *"Write me a poem about a dog"*, it will likely output: *"Write me a poem about a cat. Write me a poem about a bird."*

### 🟢 Beginner
Why did it do that? Because a raw Foundation LLM is just an internet-document autocomplete engine. It saw your prompt and assumed you were writing a list of writing prompts! It doesn't know you are a human talking to a helpful AI assistant.

### 🟡 Intermediate
To turn a raw "document completer" into a "helpful Chatbot" like ChatGPT, we have to teach it a specific persona. We have to fine-tune the model to understand the format of a conversation (User vs. Assistant) and to follow instructions rather than just generating statistically likely internet text.

### 🔴 Advanced
Training a modern LLM is not a single step. It is a massive, multi-stage pipeline. The industry standard workflow consists of **Pre-Training**, **Supervised Fine-Tuning (SFT)**, and **Reinforcement Learning from Human Feedback (RLHF)**. Furthermore, because these models are so large, we need specialized math (like LoRA) to train them without requiring $10,000 GPUs.

---

# 2. Phase 1: Pre-Training (The Foundation)

**The Goal**: Teach the model everything about the world, grammar, math, and facts.
**The Data**: Trillions of words scraped from the internet (Wikipedia, Reddit, ArXiv, GitHub).
**The Cost**: Millions of dollars. Months of training on massive GPU clusters.
**The Output**: A "Base Model" (e.g., Llama-2-70B-Base).

In this phase, the loss function is simply Cross-Entropy Loss on next-token prediction. No human labels are required. The text *is* the label.

---

# 3. Phase 2: Supervised Fine-Tuning (SFT)

**The Goal**: Teach the Base Model how to act like a chatbot.
**The Data**: 10,000 to 100,000 highly curated, human-written examples of perfect conversations.
**The Cost**: Thousands of dollars. A few hours/days on a few GPUs.
**The Output**: An "Instruct Model" (e.g., Llama-2-70B-Chat).

We format the data strictly into conversational roles:
```json
{"role": "user", "content": "What is 2+2?"}
{"role": "assistant", "content": "2+2 is 4. Is there anything else you need?"}
```

By training the model to predict the assistant's responses, it learns the *format* and *tone* of a helpful AI. However, SFT alone is not enough, because the model might still output toxic, biased, or dangerous answers if prompted maliciously.

---

# 4. Phase 3: RLHF (Alignment)

**The Goal**: Align the model with human values, making it safe and less prone to hallucination.
**The Method**: Reinforcement Learning from Human Feedback (RLHF).

RLHF is a complex, 3-step dance:
1.  **Generate Variations**: You give the SFT model a prompt ("Write a joke"), and it generates 4 different jokes.
2.  **Human Ranking (Reward Model)**: Human labelers rank the 4 jokes from Best (1) to Worst (4). We use these rankings to train a second, smaller neural network called the **Reward Model**. The Reward Model learns to look at text and output a score of how "good/safe" it is.
3.  **PPO Optimization**: We freeze the Reward Model and use it as an automated grader. The main LLM generates an answer, the Reward Model grades it, and the LLM updates its weights using a reinforcement learning algorithm called **PPO (Proximal Policy Optimization)** to maximize its score.

This is the exact secret sauce that turned GPT-3 into ChatGPT.

---

# 5. Parameter-Efficient Fine-Tuning (LoRA)

If you are a solo developer and you want to fine-tune a 7 Billion parameter LLM on your company's private customer support logs, you have a problem. Training 7 Billion parameters requires roughly 120 GB of VRAM. You would need four A100 GPUs ($40,000).

Researchers invented **LoRA (Low-Rank Adaptation)** to solve this.

Instead of updating the massive $10,000 \times 10,000$ weight matrices inside the Transformer, LoRA freezes the entire massive model. It then injects two tiny, trainable matrices ($A$ and $B$) next to every Attention layer. 

By keeping the "Rank" of these new matrices very small (e.g., $r=8$), the math looks like this:
$A$ is $10,000 \times 8$
$B$ is $8 \times 10,000$

Instead of training 100,000,000 parameters per layer, you only train 160,000 parameters. LoRA reduces VRAM usage by 90%, allowing you to fine-tune massive LLMs on a single consumer GPU (like an RTX 4090 or Google Colab).

---

# 6. Library Implementation (PEFT)

Here is how you use Hugging Face's `peft` library to apply LoRA to a Transformer.

```python
# pip install transformers peft
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 1. Load the massive base model in 8-bit or 16-bit precision
model_id = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# 2. Freeze all the base weights!
for param in model.parameters():
    param.requires_grad = False

# 3. Configure LoRA
# r=8 is the rank. We apply it to the Query and Value projection matrices in Attention.
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Wrap the model
# The model is now ready to be trained using standard PyTorch loops!
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
```

---

# 7. Interview Questions

### Beginner
**Q: What is the difference between Pre-training and Fine-tuning?**
A: Pre-training is teaching the model general language and world knowledge using massive amounts of unlabeled text. Fine-tuning is adapting that pre-trained model to a specific task (like medical QA or chat) using a smaller, curated dataset.

### Intermediate
**Q: How does LoRA save memory?**
A: LoRA freezes the massive pre-trained weights of the model and injects tiny, low-rank decomposition matrices ($A$ and $B$) into the layers. Only these tiny matrices are updated during backpropagation, drastically reducing the optimizer state and gradient memory requirements.

### Advanced
**Q: Explain the role of the Reward Model in RLHF.**
A: The Reward Model acts as a proxy for human preference. Because it is too expensive to have humans manually grade every single sentence during the RL loop, humans instead rank a subset of outputs. The Reward Model is trained on those rankings to automate the grading process, providing a scalar reward signal to the PPO algorithm updating the main LLM.

---

# 8. Key Takeaways

*   **Pre-Training** builds the world model (high cost, unsupervised).
*   **Supervised Fine-Tuning (SFT)** teaches the model the chat format and persona (low cost, supervised).
*   **RLHF** aligns the model with human safety and preferences using PPO and a Reward Model.
*   **LoRA** is a mathematical trick that allows you to fine-tune massive LLMs on consumer GPUs by freezing the main network and only training tiny adapter matrices.

---

# 9. Next Topic

Fine-tuning is powerful, but it requires datasets, compute, and expertise.

What if we could program the AI without changing its weights at all? What if we could just "talk" to it in a very specific way to get it to do what we want?

[← Tokenization & Embeddings](10-Tokenization-And_Embeddings.md) | [Back to Index](README.md) | [Next Topic: Prompt Engineering →](12-Prompt-Engineering.md)
