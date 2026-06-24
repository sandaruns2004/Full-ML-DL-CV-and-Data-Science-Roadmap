# 13 - Retrieval-Augmented Generation (RAG)

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Prerequisites**: 12-Prompt-Engineering | **Estimated Reading Time**: 25 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [The RAG Architecture](#2-the-rag-architecture)
3. [Step 1: Ingestion & Chunking](#3-step-1-ingestion--chunking)
4. [Step 2: Vector Embeddings & Databases](#4-step-2-vector-embeddings--databases)
5. [Step 3: Semantic Search (Cosine Similarity)](#5-step-3-semantic-search-cosine-similarity)
6. [Step 4: Generation](#6-step-4-generation)
7. [Library Implementation (LangChain & ChromaDB)](#7-library-implementation-langchain--chromadb)
8. [Key Takeaways](#8-key-takeaways)
9. [Next Topic](#9-next-topic)

---

# 1. What Problem Does This Solve?

You work at a company. You have 10,000 PDF documents containing internal HR policies, technical manuals, and financial reports. You want an AI chatbot that can answer questions about these documents.

### 🟢 Beginner
You cannot just use ChatGPT directly. ChatGPT was trained on the public internet. It has never seen your company's private PDFs. If you ask it about your HR policy, it will **Hallucinate** (make up a fake, plausible-sounding answer).

### 🟡 Intermediate
Why not just paste the PDFs into the prompt? Because 10,000 PDFs contain hundreds of millions of words. This vastly exceeds the Context Window limit of any LLM. Even if an LLM had a massive context window, parsing 10,000 PDFs for every single chat message would cost hundreds of dollars per query and take 10 minutes to run.

### 🔴 Advanced
Why not Fine-Tune the LLM? Fine-tuning is great for teaching a model a *style* (like how to talk like Shakespeare). It is terrible for teaching a model *facts*. If the HR policy changes tomorrow, you would have to re-fine-tune the entire multi-billion parameter model. 

The industry standard solution is **Retrieval-Augmented Generation (RAG)**. We separate the "Memory" (a database) from the "Brain" (the LLM).

---

# 2. The RAG Architecture

RAG is a two-step pipeline.

```mermaid
flowchart TD
    subgraph The Database (Vector DB)
    Doc[10,000 PDFs] --> Chunk[Chunk into Paragraphs]
    Chunk --> Embed[Convert to Vectors]
    Embed --> DB[(Vector Database)]
    end
    
    subgraph The Runtime (Chatbot)
    User[User: "What is the refund policy?"] --> Q_Embed[Convert Query to Vector]
    Q_Embed --> Search{Search Database}
    Search -.->|Cosine Similarity| DB
    Search --> TopK[Retrieve Top 3 matching paragraphs]
    TopK --> Prompt[Combine Question + Top 3 Paragraphs into Prompt]
    Prompt --> LLM[LLM generates Final Answer]
    end
```

By retrieving only the 3 relevant paragraphs, we easily fit inside the Context Window, we drastically reduce hallucinations (because the LLM is explicitly instructed to only use the provided text), and if the policy changes, we just delete the old paragraph from the database.

---

# 3. Step 1: Ingestion & Chunking

We cannot embed entire 50-page PDFs as a single vector. The vector would become "diluted" and lose specific semantic meaning.
Instead, we **chunk** the text. 
We slice the documents into overlapping chunks of ~500 words. 
*   Chunk 1: Words 0-500
*   Chunk 2: Words 450-950 (overlapping prevents cutting a sentence in half).

---

# 4. Step 2: Vector Embeddings & Databases

We take each 500-word chunk and pass it through an **Embedding Model** (usually a small Encoder-only Transformer like BERT or OpenAI's `text-embedding-ada-002`). 

The model outputs a dense vector (e.g., a list of 1536 floating-point numbers) representing the semantic meaning of that paragraph.
We save the raw text and its mathematical vector into a **Vector Database** (like ChromaDB, Pinecone, or FAISS).

---

# 5. Step 3: Semantic Search (Cosine Similarity)

When the user asks: *"What is the refund policy?"*, we pass that exact question through the exact same Embedding Model. 

Now we have a Query Vector. 
We perform a mathematical operation called **Cosine Similarity** between the Query Vector and the 100,000 Chunk Vectors in our database.
Because they are in the same vector space, a question about refunds will mathematically cluster right next to the paragraph explaining the refund policy, *even if they don't share any exact keywords*.

This is why Vector Search is infinitely superior to traditional SQL keyword search.

---

# 6. Step 4: Generation

We take the top 3 most similar paragraphs returned by the database, and we inject them into a System Prompt.

**Final Prompt sent to LLM:**
```text
System: You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say "I don't know."

Context: 
[Paragraph 1: Refunds are allowed within 30 days of purchase...]
[Paragraph 2: Digital goods are non-refundable...]
[Paragraph 3: To process a refund, email support@company.com...]

User: What is the refund policy for digital goods?
```

The LLM (the Brain) reads the prompt and outputs: *"Digital goods are non-refundable according to the company policy."*

---

# 7. Library Implementation (LangChain & ChromaDB)

Here is a minimal RAG pipeline using Python.

```python
# pip install langchain chromadb sentence-transformers
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 1. Load and Chunk the Document
loader = TextLoader("company_policy.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. Create Embeddings and Store in Vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(chunks, embeddings)

# 3. The User Asks a Question
query = "What happens if I return an item after 40 days?"

# 4. Semantic Search (Retrieve top 2 chunks)
retrieved_docs = vector_db.similarity_search(query, k=2)

print("Most relevant text found:")
print(retrieved_docs[0].page_content)

# Step 5 (Not shown): You would take retrieved_docs[0].page_content 
# and paste it into an OpenAI/Llama API call alongside the user's query!
```

---

# 8. Key Takeaways

*   **RAG (Retrieval-Augmented Generation)** solves LLM hallucinations and context limits by connecting the LLM to an external knowledge database.
*   Documents are **chunked** and converted into **Vectors** using an Embedding model.
*   When a user asks a question, we use **Cosine Similarity** to find the most relevant chunks in the **Vector Database**.
*   The retrieved chunks are injected into the prompt, forcing the LLM to generate an answer based strictly on the provided factual context.

---

# 9. Next Topic

We have mastered text. But the world is not just text. We have images, audio, and video. 

How do we modify the Transformer architecture so it can understand a picture of a dog, or write code based on a UI wireframe?

[← Prompt Engineering](12-Prompt-Engineering.md) | [Back to Index](README.md) | [Next Topic: Multimodal Transformers →](14-Multimodal-Transformers.md)
