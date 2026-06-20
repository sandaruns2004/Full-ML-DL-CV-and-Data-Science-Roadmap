# 🛠️ LLM Fine-Tuning & RLHF

> **Prerequisites**: GPT & Decoder Models | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [The Base Model Problem](#1-the-base-model-problem)
2. [Stage 1: Supervised Fine-Tuning (SFT)](#2-stage-1-supervised-fine-tuning-sft)
3. [Parameter-Efficient Fine-Tuning (PEFT) & LoRA](#3-parameter-efficient-fine-tuning-peft--lora)
4. [Stage 2: Reinforcement Learning from Human Feedback (RLHF)](#4-stage-2-reinforcement-learning-from-human-feedback-rlhf)
5. [Direct Preference Optimization (DPO)](#5-direct-preference-optimization-dpo)
6. [From-Scratch LoRA Implementation (PyTorch)](#6-from-scratch-lora-implementation-pytorch)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Base Model Problem

If you train a massive GPT model (like GPT-3 or LLaMA) on 2 trillion tokens of internet text, you get a **Base Model**. 

A Base Model is an incredible completion engine, but it is **not an assistant**. 

**Example of interacting with a Base Model:**
> **Prompt**: "Write a python function to reverse a string."
> **Base Model Output**: "...and write a java function to reverse a string. How to reverse a string in C++?"

Because it learned from forums (like StackOverflow or Reddit), it assumes your prompt is the title of a forum post, so it just generates more titles! It doesn't know it's supposed to *answer* you.

To turn a Base Model into an Assistant (like ChatGPT), we need an alignment pipeline. The standard pipeline has two major stages: **SFT** and **RLHF**.

---

## 2. Stage 1: Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning (also called Instruction Tuning) teaches the model the *format* of a conversation.

We create a dataset of tens of thousands of high-quality `(Prompt, Response)` pairs written by humans.

```json
[
  {
    "instruction": "Write a python function to reverse a string.",
    "response": "Here is the python code:\n```python\ndef reverse_string(s):\n    return s[::-1]\n```"
  }
]
```

We format this into a single string using special tokens (e.g., `<|user|>` and `<|assistant|>`):
`<|user|> Write a python function to reverse a string. <|assistant|> Here is the python code... [EOS]`

We train the model using standard Autoregressive Cross-Entropy Loss, but we **only calculate loss on the Assistant's response tokens**. The model learns that when it sees the `<|assistant|>` token, it should stop generating internet garbage and start generating helpful answers.

---

## 3. Parameter-Efficient Fine-Tuning (PEFT) & LoRA

Fine-tuning a 70 Billion parameter model requires massive GPU clusters. If you update all 70B weights, you need terabytes of VRAM just to store the optimizer states (Adam momentum/variance).

To solve this, researchers use **PEFT (Parameter-Efficient Fine-Tuning)**. The most famous PEFT method is **LoRA (Low-Rank Adaptation)**.

### The Mathematics of LoRA
Let $W \in \mathbb{R}^{d \times k}$ be a pre-trained weight matrix in a Transformer (e.g., the Query projection matrix).
During standard fine-tuning, we update $W$ by adding a gradient matrix $\Delta W$:
$$ W_{new} = W + \Delta W $$

**The LoRA Hypothesis**: The "intrinsic rank" of the update $\Delta W$ is very low. You don't need a full-rank matrix to learn a new task.

Instead of learning the massive $d \times k$ matrix $\Delta W$, we decompose it into two tiny matrices, $A$ and $B$:
$$ \Delta W = B \times A $$

Where:
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- $r$ is the "rank" (a very small number, e.g., $r=8$).

**During Training**: 
1. We freeze the original $W$ completely.
2. We randomly initialize $A$, and initialize $B$ to all zeros (so $BA = 0$ at the start).
3. We only calculate gradients and update the tiny matrices $A$ and $B$.

**Memory Savings**:
If $d=4096, k=4096$, updating $\Delta W$ requires training $16,777,216$ parameters.
If we use LoRA with rank $r=8$, we train $B (4096 \times 8) + A (8 \times 4096) = 65,536$ parameters.
**That's a 99.6% reduction in trainable parameters!**

---

## 4. Stage 2: Reinforcement Learning from Human Feedback (RLHF)

SFT teaches the model how to talk, but it doesn't prevent the model from hallucinating, being toxic, or giving unsafe answers. For this, OpenAI introduced **RLHF**.

RLHF has 3 main steps:

### Step 1: Train the SFT Model
(As described in Section 2).

### Step 2: Train a Reward Model
1. Feed a prompt to the SFT model and ask it to generate 4 different responses.
2. Human labelers read the 4 responses and rank them from best to worst.
3. We train a separate **Reward Model** (usually a smaller Transformer) to predict a scalar reward score for a given `(Prompt, Response)` pair. The loss function forces the Reward Model to output a higher score for the response the human preferred.

### Step 3: PPO (Proximal Policy Optimization) Optimization
1. We freeze the Reward Model.
2. We take our SFT Model (now called the **Policy**). We give it a prompt and let it generate a response.
3. We feed that response into the Reward Model to get a score (e.g., +4.2).
4. We use the **PPO** Reinforcement Learning algorithm to update the Policy's weights to maximize that reward score.

*(To prevent the model from gaming the reward system by outputting gibberish that technically scores high, we add a Kullback-Leibler (KL) divergence penalty to ensure the Policy doesn't deviate too far from the original SFT model).*

---

## 5. Direct Preference Optimization (DPO)

RLHF is incredibly complex and unstable because PPO is notoriously difficult to tune.

In 2023, Stanford researchers published **DPO (Direct Preference Optimization)**, which effectively killed RLHF.

**The Math of DPO**:
They mathematically proved that you can re-arrange the RLHF objective to completely eliminate the Reward Model and the PPO loop!

Instead, you just need a dataset of human preferences: `(Prompt, Winning_Response, Losing_Response)`.

You calculate the probability of the Winning and Losing responses under both the model being trained ($\pi_\theta$) and a frozen reference model ($\pi_{ref}$). The loss function directly increases the probability of the Winning response while decreasing the probability of the Losing response, scaled by the reference model's logits.

$$ \mathcal{L}_{DPO} = - \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) $$

DPO achieves the same or better performance than RLHF but is as simple and stable to train as standard Supervised Fine-Tuning.

---

## 6. From-Scratch LoRA Implementation (PyTorch)

Let's implement a LoRA wrapper for a standard PyTorch Linear layer.

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16):
        super().__init__()
        # 1. The original frozen pre-trained weights
        self.pretrained_linear = nn.Linear(in_features, out_features, bias=False)
        self.pretrained_linear.weight.requires_grad = False # FREEZE!
        
        # 2. The LoRA Matrices
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Matrix A (r x in_features)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        # Matrix B (out_features x r)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # 3. Initialization
        # A is initialized normally. B is initialized to 0.
        # This ensures that at step 0, LoRA doesn't change the model output at all.
        nn.init.normal_(self.lora_A.weight, std=1 / r)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # 1. Standard forward pass through frozen weights
        pretrained_out = self.pretrained_linear(x)
        
        # 2. Forward pass through LoRA matrices
        # x -> A -> B
        lora_out = self.lora_B(self.lora_A(x))
        
        # 3. Scale and Add
        return pretrained_out + (lora_out * self.scaling)

# --- Demonstration ---
in_dim = 4096
out_dim = 4096
rank = 8

layer = LoRALinear(in_dim, out_dim, r=rank)

# Count parameters
frozen_params = sum(p.numel() for p in layer.pretrained_linear.parameters())
trainable_params = sum(p.numel() for p in layer.lora_A.parameters()) + \
                   sum(p.numel() for p in layer.lora_B.parameters())

print(f"Frozen Parameters (W): {frozen_params:,}")
print(f"Trainable Parameters (A + B): {trainable_params:,}")
print(f"Percentage of trainable params: {(trainable_params / frozen_params) * 100:.3f}%")
```

*Output:*
```text
Frozen Parameters (W): 16,777,216
Trainable Parameters (A + B): 65,536
Percentage of trainable params: 0.391%
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **QLoRA Fine-tuning**: Use the `Unsloth` library (or Hugging Face `PEFT`) to fine-tune Llama-3-8B on a Google Colab free tier using QLoRA (Quantized LoRA). Train it on a tiny medical dataset to turn it into a medical assistant.
- 🟡 **DPO Preference Tuning**: Download the `Anthropic/hh-rlhf` dataset from Hugging Face. Use the `TRL` (Transformer Reinforcement Learning) library to apply DPO to a pre-trained SFT model.

### What's Next
| Next | Why |
|------|-----|
| [Mixture Of Experts](./07-Mixture-Of-Experts.md) | Learn how massive models like GPT-4 route queries to specialized sub-networks to save compute. |

---

[← Vision Transformers ViT](./05-Vision-Transformers-ViT.md) | [Back to Index](../README.md) | [Next: Mixture Of Experts →](./07-Mixture-Of-Experts.md)
