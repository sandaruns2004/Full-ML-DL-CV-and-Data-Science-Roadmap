# 12 - Prompt Engineering

> **Difficulty**: ⭐☆☆☆☆ Beginner | **Prerequisites**: 11-Large-Language-Models | **Estimated Reading Time**: 15 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Zero-Shot vs Few-Shot Prompting](#2-zero-shot-vs-few-shot-prompting)
3. [Chain of Thought (CoT)](#3-chain-of-thought-cot)
4. [Structured Prompting & Templates](#4-structured-prompting--templates)
5. [Key Takeaways](#5-key-takeaways)
6. [Next Topic](#6-next-topic)

---

# 1. What Problem Does This Solve?

We have a massive Foundation Model with 1 Trillion parameters. It is capable of writing code, drafting legal documents, and translating languages.

### 🟢 Beginner
If you treat an LLM like a Google Search bar and type *"Write me an app"*, the AI will fail. It will output a generic, useless script. The AI is a massive engine, but the **Prompt** is the steering wheel. If you don't know how to steer, you will crash.

### 🟡 Intermediate
**Prompt Engineering** is the practice of structuring text specifically to guide an LLM into producing highly accurate, formatted, and constrained outputs. It is not about "tricking" the AI. It is about providing maximum context so the Self-Attention mechanism can precisely locate the correct knowledge in its massive neural network.

### 🔴 Advanced
Because LLMs are autoregressive probability engines, every single word you put in the prompt alters the mathematical probability of the next generated word. Prompt Engineering is effectively "programming" the AI using natural language (In-Context Learning) instead of Python code.

---

# 2. Zero-Shot vs Few-Shot Prompting

**Zero-Shot Prompting**
You ask the model to perform a task without giving it any examples. This works for easy, well-defined tasks.
*   *Prompt:* `Translate to French: "The weather is nice."`
*   *Output:* `"Le temps est agréable."`

**Few-Shot Prompting**
If you want the AI to output data in a very specific, weird format (like a custom JSON schema), Zero-Shot will fail. You must use Few-Shot prompting, where you provide 2 to 5 examples of the exact input/output format you want.
*   *Prompt:*
    ```text
    Text: "I loved the movie!" -> JSON: {"sentiment": "positive"}
    Text: "Terrible service." -> JSON: {"sentiment": "negative"}
    Text: "The food was okay." -> JSON:
    ```
*   *Output:* `{"sentiment": "neutral"}`

By providing examples, you mathematically bias the LLM's autoregressive engine to copy your exact pattern.

---

# 3. Chain of Thought (CoT)

If you ask an LLM a complex math puzzle, it will often guess the wrong answer immediately.

LLMs do not have an internal monologue. They cannot think in silence. They can only "think" by writing words into the context window. If you force the LLM to write out the final answer immediately, it skips the logical reasoning steps and guesses blindly.

**Chain of Thought (CoT)** solves this. You simply append the phrase:
`"Let's think step by step."`

By forcing the LLM to write down step 1, step 2, and step 3 into the context window, its Self-Attention mechanism can physically look back at step 3 to calculate step 4. This simple trick improves the math and logic scores of LLMs by over 40%.

---

# 4. Structured Prompting & Templates

In production, you do not type prompts manually. You write Python code that builds prompts dynamically using **Templates**.

A professional prompt is highly structured using XML tags or Markdown to clearly separate instructions from data.

**Example Production Template:**
```xml
<system_instructions>
You are a senior data analyst. 
Your goal is to extract the names of companies mentioned in the text.
Output the result as a comma-separated list.
Do not write any introductory text.
</system_instructions>

<few_shot_examples>
Input: "Apple and Google are tech giants."
Output: Apple, Google
</few_shot_examples>

<user_text>
{USER_INPUT_GOES_HERE}
</user_text>
```

When building an AI app, the developer writes the `system_instructions` and the `few_shot_examples`. The user only provides the `user_text` through the app's UI.

---

# 5. Key Takeaways

*   **Prompt Engineering** is the primary way to interact with and control Foundation Models.
*   **Zero-Shot** works for simple tasks, but **Few-Shot** (providing examples) is required to guarantee strict output formats.
*   **Chain of Thought (CoT)** forces the LLM to write out its intermediate reasoning, drastically improving logic and math performance.
*   Production systems use **Structured Templates** (often with XML tags) to separate system instructions from user inputs safely.

---

# 6. Next Topic

Prompt Engineering is powerful, but what if the user asks a question about a private, highly confidential company document? The LLM has never seen the document, so no amount of prompt engineering will make it guess the right answer.

We need a way to combine the intelligence of the LLM with a private, searchable database.

[← Large Language Models](11-Large-Language-Models.md) | [Back to Index](README.md) | [Next Topic: RAG & Knowledge Augmentation →](13-RAG-And_Knowledge_Augmentation.md)
