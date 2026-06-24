# 12 - Prompt Engineering

> **Difficulty**: ⭐☆☆☆☆ Beginner | **Prerequisites**: 09-GPT-Family | **Estimated Reading Time**: 15 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Zero-Shot vs Few-Shot Prompting](#2-zero-shot-vs-few-shot-prompting)
3. [Chain-of-Thought (CoT)](#3-chain-of-thought-cot)
4. [Role Prompting & System Prompts](#4-role-prompting--system-prompts)
5. [Retrieval & Context Window Limits](#5-retrieval--context-window-limits)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

If an LLM is a massive, multi-billion parameter brain, how do you steer it?

### 🟢 Beginner
You can ask an AI to "write a story." But if it writes a story for children when you wanted a sci-fi thriller, the AI didn't fail—you failed to give it instructions. Prompt Engineering is the science of writing instructions so perfectly that the AI cannot possibly misunderstand what you want.

### 🟡 Intermediate
Before 2020, to make an AI do a specific task (like sentiment analysis), we had to write Python code to mathematically update the model's weights (Fine-Tuning). With modern LLMs, we don't need to change the weights. We just use natural language to "program" the model in real-time. This is called **In-Context Learning**.

### 🔴 Advanced
Because LLMs are autoregressive, next-token prediction engines, the mathematical probabilities of what word comes next are entirely dependent on the exact characters currently in the context window. By strategically placing examples and formatting structures into the prompt, we manipulate the Self-Attention mechanism, forcing the LLM to output text in our exact desired probability distribution.

---

# 2. Zero-Shot vs Few-Shot Prompting

**Zero-Shot Prompting**
You ask the model to do a task without showing it any examples.
*   **Prompt:** `Classify this review: "The battery life is terrible." Sentiment:`
*   **Output:** `Negative`
This works for simple tasks, but fails when you want a highly specific format.

**Few-Shot Prompting**
You provide 2-5 examples of the exact input/output format you want. By doing this, you mathematically bias the LLM's attention mechanism to copy the pattern.
*   **Prompt:**
    ```text
    Text: "The screen is bright." -> Label: POS
    Text: "The phone broke immediately." -> Label: NEG
    Text: "The camera is blurry." -> Label:
    ```
*   **Output:** `NEG`

---

# 3. Chain-of-Thought (CoT)

This is the most important breakthrough in Prompt Engineering.

If you ask an LLM a complex math puzzle, it will often answer incorrectly. Why? Because it tries to predict the final number immediately.

**Standard Prompt:**
`Q: John has 5 apples. He eats 2. He buys 4 times as many as he currently has. How many does he have?`
`A: 17` (Incorrect, it just guessed).

**Chain-of-Thought Prompt:**
You add the magic phrase: *"Let's think step by step."*
`Q: John has 5 apples. He eats 2... Let's think step by step.`
`A: John starts with 5. He eats 2, leaving 3. He buys 4 times what he has (4 * 3 = 12). He now has 3 + 12 = 15 apples.` (Correct!).

**Why this works mathematically:** 
An LLM has no hidden internal monologue. It can only "think" by writing words into the context window. By forcing it to write out the intermediate math steps, you are giving the Self-Attention mechanism a clean, logical foundation to look back on before it predicts the final number.

---

# 4. Role Prompting & System Prompts

Most chat applications (like ChatGPT or Claude) use a hidden **System Prompt** before your user message. 

You can radically change the behavior of the model by defining a role.
**System Prompt:** `You are a senior Linux kernel developer. You are highly technical, abrasive, and only respond with code. Do not apologize.`

By establishing this persona, the LLM will activate the linguistic weights associated with code documentation and technical forums, completely ignoring its standard "helpful assistant" persona.

---

# 5. Retrieval & Context Window Limits

Prompt Engineering has a hard limit: the **Context Window**.
- GPT-3 had a 4,000 token limit (~6 pages of text).
- GPT-4 expanded to 128,000 tokens (~300 pages).
- Gemini 1.5 expanded to 2,000,000 tokens.

If you try to paste a 1,000-page medical textbook into the prompt to ask a question, the model will either crash or suffer from **Lost in the Middle** syndrome (where the Attention mechanism forgets things in the middle of massive prompts).

To solve this, we don't paste the whole book. We write a Python script to search the book, find the 1 relevant page, and only paste that 1 page into the prompt. 

This leads us to the most important architecture in modern AI Engineering: **RAG**.

---

# 6. Key Takeaways

*   **Prompt Engineering** is the practice of manipulating an LLM's autoregressive probabilities using natural language.
*   **Few-Shot Prompting** gives the model a pattern to copy, eliminating the need to fine-tune weights for simple formatting tasks.
*   **Chain-of-Thought** forces the model to write out its reasoning, giving it "scratchpad space" to calculate logic before predicting an answer.
*   **System Prompts** lock the model into a specific persona or behavioral constraint.

---

# 7. Next Topic

We know how to write a good prompt. But what if we need the AI to answer a question about proprietary company data that wasn't on the internet when the AI was trained?

We have to build a system that can search our database and automatically write the prompt for us.

[← Transformer Training](11-Transformer-Training.md) | [Back to Index](README.md) | [Next Topic: Retrieval-Augmented Generation (RAG) →](13-Retrieval-Augmented-Generation-RAG.md)
