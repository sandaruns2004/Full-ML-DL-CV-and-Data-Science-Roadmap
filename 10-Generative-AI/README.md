# 🧠 Module 10: Generative AI

Welcome to **Module 10: Generative AI**. 

Generative AI represents a paradigm shift in machine learning. Instead of teaching models to analyze data (e.g., *"Is this a picture of a cat?"*), we are teaching models to create data (e.g., *"Draw me a picture of a cat riding a skateboard on Mars"*).

This module will take you from the foundational probability mathematics of how AI "creates", all the way up to building autonomous AI Agents powered by Large Language Models (LLMs).

---

## 📚 Curriculum Structure

This module follows a strict **Beginner $\to$ Intermediate $\to$ Advanced** progression. Do not skip the early chapters!

### Foundation & Math
1.  [**01 - Introduction To Generative AI**](01-Introduction-To-Generative-AI.md) - The difference between Discriminative and Generative models.
2.  [**02 - Probability & Generative Modeling**](02-Probability-And_Generative_Modeling.md) - Maximum Likelihood, Data Distributions, and the Curse of Dimensionality.

### The Compression Era
3.  [**03 - Autoencoders**](03-Autoencoders.md) - Compressing reality into Latent Spaces.
4.  [**04 - Variational Autoencoders (VAEs)**](04-Variational-Autoencoders-VAEs.md) - Organizing Latent Spaces using the Reparameterization Trick to generate new data.

### The Adversarial Era
5.  [**05 - Generative Adversarial Networks (GANs)**](05-Generative-Adversarial-Networks-GANs.md) - The Minimax game between Generator and Discriminator.
6.  [**06 - GAN Variants**](06-GAN-Variants.md) - DCGAN, cGAN, CycleGAN, and StyleGAN.
7.  [**07 - Image Generation With GANs**](07-Image-Generation-With-GANs.md) - Real-world applications like Data Augmentation and Style Transfer.

### The Diffusion Era
8.  [**08 - Diffusion Models**](08-Diffusion-Models.md) - Forward noise and Reverse denoising using U-Nets.
9.  [**09 - Stable Diffusion**](09-Stable-Diffusion.md) - Latent Diffusion and Cross-Attention Text Conditioning.

### The Transformer Era
10. [**10 - Transformers In Generative AI**](10-Transformers-In-Generative-AI.md) - Self-Attention and Autoregressive generation.
11. [**11 - Large Language Models**](11-Large-Language-Models.md) - Scaling Laws, Foundation Models, and In-Context Learning.
12. [**12 - Prompt Engineering**](12-Prompt-Engineering.md) - Zero-shot, Few-shot, and Chain of Thought.
13. [**13 - RAG & Knowledge Augmentation**](13-RAG-And_Knowledge_Augmentation.md) - Connecting LLMs to private Vector Databases to eliminate hallucinations.
14. [**14 - Multimodal Generative AI**](14-Multimodal-Generative-AI.md) - Fusing Vision, Audio, and Text into a single unified latent space.
15. [**15 - AI Agents & Tool Use**](15-AI-Agents-And_Tool_Use.md) - ReAct planning loops and autonomous tool execution.

### Ethics & Safety
16. [**16 - Responsible Generative AI**](16-Responsible-Generative-AI.md) - Bias, Deepfakes, Copyright, and RLHF alignment.

---

## 💻 Labs & Notebooks

Inside the `notebooks/` directory, you will find interactive Jupyter notebooks designed to let you visualize the mathematics discussed in the theory files:

*   `Latent_Space_Visualization_Lab.ipynb`
*   `GAN_Training_Playground.ipynb`
*   `Diffusion_Process_Visualization.ipynb`
*   `Stable_Diffusion_Exploration_Lab.ipynb`
*   `LLM_Token_Prediction_Lab.ipynb`
*   `Embeddings_And_Similarity_Lab.ipynb`
*   `RAG_Mini_Pipeline_Lab.ipynb`

---

## 🚀 Projects

Inside the `projects/` directory, you will build 10 end-to-end applications demonstrating these architectures in production environments:

1.  `01-Image-Generator-Using-GANs`
2.  `02-Face-Generation-Using-StyleGAN`
3.  `03-Autoencoder-Based-Anomaly-Detection`
4.  `04-Text-Generation-With-GPT-Models`
5.  `05-Document-Question-Answering-Assistant` (Streamlit + RAG)
6.  `06-AI-Research-Assistant`
7.  `07-Multimodal-Image-Captioning-System`
8.  `08-Stable-Diffusion-Prompt-Explorer`
9.  `09-Mini-GPT-From-Scratch` (PyTorch)
10. `10-AI-Agent-Workflow-Simulator`

Happy Generating!
