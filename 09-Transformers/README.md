# Module 09: Transformers & Foundation Models

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Prerequisites**: Module 08 (Sequence Models) | **Estimated Learning Time**: 15 Hours

Welcome to **Module 09**, the most important module in modern Artificial Intelligence. 

Before 2017, Recurrent Neural Networks (RNNs) and LSTMs ruled sequence processing. But they were slow, sequential, and suffered from information bottlenecks. Then, researchers at Google published *"Attention Is All You Need"*, introducing the **Transformer**.

By dropping recurrence entirely and relying solely on **Self-Attention**, the Transformer unlocked infinite parallelization across GPUs. This architecture scaled so effectively that it birthed the modern era of **Large Language Models (LLMs)**, **Vision Transformers (ViTs)**, and **Multimodal Foundation Models** like ChatGPT, GPT-4, Llama, and Gemini.

---

## 🗺️ Study Roadmap

This module is designed to take you from a conceptual understanding of Attention all the way to building Retrieval-Augmented Generation (RAG) systems with modern LLMs.

### Part 1: Foundations & The Attention Revolution
*   [01 - Why Transformers?](01-Why-Transformers.md)
*   [02 - Attention Mechanism](02-Attention-Mechanism.md)
*   [03 - Self-Attention](03-Self-Attention.md)
*   [04 - Multi-Head Attention](04-Multi-Head-Attention.md)
*   [05 - Positional Encoding](05-Positional-Encoding.md)

### Part 2: The Core Architecture & The LLM Era
*   [06 - Transformer Architecture](06-Transformer-Architecture.md)
*   [07 - Building A Transformer Step-By-Step](07-Building-A-Transformer-Step-By-Step.md)
*   [08 - BERT & Encoder Models](08-BERT.md)
*   [09 - The GPT Family & Decoder Models](09-GPT-Family.md)
*   [10 - Tokenization & Embeddings](10-Tokenization-And_Embeddings.md)

### Part 3: Training & Production Engineering
*   [11 - Transformer Training & Fine-Tuning](11-Transformer-Training.md)
*   [12 - Prompt Engineering](12-Prompt-Engineering.md)
*   [13 - Retrieval-Augmented Generation (RAG)](13-Retrieval-Augmented-Generation-RAG.md)
*   [14 - Multimodal Transformers](14-Multimodal-Transformers.md)
*   [15 - Vision Transformers (ViT)](15-Vision-Transformers-ViT.md)
*   [16 - Modern Foundation Models](16-Modern-Foundation-Models.md)

---

## 💻 Educational Notebooks

Located in the `notebooks/` directory. These interactive labs bridge the theory with pure code:
1.  **Attention_Visualization_Lab.ipynb**: Visualize Attention matrices and token relationships.
2.  **Self-Attention_Playground.ipynb**: Build self-attention from scratch.
3.  **Transformer_Components_Lab.ipynb**: Implement Embeddings, Positional Encoding, and Attention step-by-step.
4.  **Transformer_From_Scratch.ipynb**: Educational NumPy/PyTorch implementation of the full architecture.
5.  **BERT_Fine_Tuning_Lab.ipynb**: Practical transfer learning for text classification.
6.  **GPT_Text_Generation_Lab.ipynb**: Token generation, sampling methods, and temperature effects.
7.  **RAG_Mini_Lab.ipynb**: Small Retrieval-Augmented Generation pipeline.

---

## 🛠️ Project Roadmap

Located in the `projects/` directory. Nine distinct, specialized projects to build your portfolio.

| Project | Description | Technology Focus |
| :--- | :--- | :--- |
| **01-Transformer-Text-Classifier** | Document classification system. | BERT Fine-Tuning |
| **02-Sentiment-Analysis-Using-BERT** | Analyze reviews and opinions. | Transfer Learning, Eval Pipelines |
| **03-GPT-Text-Generator** | CLI app for creative text generation. | GPT Architecture, Token Prediction |
| **04-Question-Answering-System** | Context-based QA. | BERT QA |
| **05-RAG-Knowledge-Assistant** | Interactive UI to query custom documents. | Embeddings, Vector Search, LLM |
| **06-Transformer-Machine-Translation**| Language translation engine. | Seq2Seq Transformer, Attention Viz |
| **07-News-Summarization-System** | Generate concise summaries from articles. | Transformer Summarizers |
| **08-Vision-Transformer-Classifier** | Image classification via image patches. | ViT vs CNN architectures |
| **09-Mini-Transformer-Framework** | Build core components from scratch. | PyTorch/NumPy internals |

---

## 🎯 Learning Outcomes & Skills Gained

By the end of this module, you will not just know how to call an API. You will understand:
*   **The Math of Attention**: How Query, Key, and Value matrices dynamically route information.
*   **Architectural Nuances**: Why BERT uses Encoders (Bi-directional) and GPT uses Decoders (Causal).
*   **Scaling Laws**: How and why LLMs exhibit emergent capabilities at scale.
*   **Production Workflows**: How to construct a RAG pipeline and fine-tune models on custom data.
*   **Beyond Text**: How Transformers are processing images (ViTs) and audio.

**Ready to begin? Start with [01 - Why Transformers?](01-Why-Transformers.md).**
