# 🧠 Advanced Projects

> **Target**: Master Transformers, Generative AI, Reinforcement Learning, and complex architectures.

These projects are difficult. They require understanding recent research papers, managing significant computational resources, and debugging complex tensor math. If you can complete these, you are ready for a Senior ML Engineering role.

---

## 1. Local RAG System with LLaMA-3

**Goal**: Build a fully private, offline Retrieval-Augmented Generation (RAG) system that can answer complex questions about a folder of massive PDF documents.

**The Tools**: Hugging Face `transformers`, `LangChain`, `ChromaDB` (or `FAISS`), and `Ollama` for local LLM inference.

**Requirements**:
1. **Document Ingestion**: Write a script that uses `PyPDF2` or `pdfplumber` to read 5 research papers.
2. **Chunking**: Use LangChain's `RecursiveCharacterTextSplitter` to chop the papers into 500-word chunks with a 50-word overlap.
3. **Embedding & Vector DB**: Use the `sentence-transformers/all-MiniLM-L6-v2` model to embed every chunk into a dense vector. Store these vectors in a local `ChromaDB` collection.
4. **Local LLM**: Install `Ollama` and pull a 8-Billion parameter LLM (like `llama3:8b`).
5. **The RAG Chain**: Write the retrieval logic. When a user asks a question, embed the question, query ChromaDB for the top 3 most relevant chunks, inject those chunks into a prompt template, and send the prompt to LLaMA-3 to generate the answer.
6. **UI**: Wrap the whole thing in a clean `Gradio` or `Streamlit` chat interface.

**Bonus Challenge**: Implement "Conversation Memory". Allow the user to ask follow-up questions (e.g., "Can you elaborate on your previous point?") by feeding the chat history back into the LLM along with the retrieved documents.

---

## 2. Proximal Policy Optimization (PPO) for Algorithmic Trading

**Goal**: Train a Deep Reinforcement Learning agent to autonomously trade stocks.

**The Tools**: `Gymnasium`, `Stable-Baselines3`, `pandas`.

**Requirements**:
1. **The Environment**: You cannot use a pre-built Gym environment. You must write a custom `gymnasium.Env` class. 
   - **State**: The last 30 days of `Open, High, Low, Close, Volume` (OHLCV) for AAPL, plus 3 technical indicators you calculate yourself (e.g., MACD, RSI, Bollinger Bands).
   - **Action Space**: Discrete (0: Sell, 1: Hold, 2: Buy).
   - **Reward**: The change in Portfolio Net Worth after the action is taken.
2. **Data**: Download 10 years of AAPL daily stock data using the `yfinance` library. Use the first 8 years for training and the last 2 years for testing.
3. **Training**: Instantiate an MLP PPO agent from `Stable-Baselines3` and train it for at least 500,000 timesteps.
4. **Evaluation**: Write an evaluation script that lets the trained agent "play" through the 2-year unseen test set. Plot the Agent's portfolio value over time against a simple "Buy and Hold" baseline.

**Bonus Challenge**: Add transaction fees to the environment (e.g., 0.1% fee per trade). Watch how the agent learns to stop high-frequency trading and holds longer to avoid going bankrupt from fees.

---

## 3. Image Generation with a From-Scratch Diffusion Model

**Goal**: Build the mathematical foundation of modern Generative AI (like Midjourney/DALL-E) by writing a Denoising Diffusion Probabilistic Model (DDPM) from scratch.

**The Dataset**: The CIFAR-10 dataset (or a custom dataset of 64x64 faces/anime characters).

**Requirements**:
1. **The Forward Process (Adding Noise)**: Write a PyTorch function that takes an image and iteratively adds Gaussian noise over $T=1000$ timesteps according to a strict cosine variance schedule.
2. **The U-Net Architecture**: Write a custom PyTorch U-Net model. It must take an image and the current timestep $t$ as inputs. It must output an image of the exact same size. Use Sinusoidal Positional Embeddings to inject the timestep $t$ into the convolutional layers.
3. **Training Loop**: Sample random images, sample a random timestep $t$, apply the noise, and pass it to the U-Net. Train the U-Net using Mean Squared Error to predict the *noise* that was added to the image.
4. **The Reverse Process (Sampling)**: Write the sampling loop. Start with pure random static noise. Iteratively use your trained U-Net to predict the noise, subtract a fraction of it, and repeat 1000 times until a crystal-clear generated image emerges.

**Bonus Challenge**: Implement Class-Conditioned Generation. Modify the U-Net to accept a class label (e.g., "Frog") so you can specifically command the model to generate a frog instead of a random image!

---

[← Intermediate Projects](02-Intermediate-Projects.md) | [Back to Index](../README.md) | [Next: Capstone Projects →](04-Capstone-Projects.md)
