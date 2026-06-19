# 🗄️ RAG & Vector Databases

> **Prerequisites**: Transformer Architecture, Word Embeddings | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Limitations of Vanilla LLMs](#1-the-limitations-of-vanilla-llms)
2. [What is RAG?](#2-what-is-rag)
3. [The RAG Pipeline](#3-the-rag-pipeline)
4. [Vector Databases](#4-vector-databases)
5. [Implementation Example](#5-implementation-example)

---

## 1. The Limitations of Vanilla LLMs

Large Language Models (LLMs) like GPT-4 are incredibly powerful, but they suffer from three major limitations in production:
1. **Knowledge Cutoff**: They cannot access information generated after their training data was collected.
2. **Hallucination**: When they don't know an answer, they confidently invent one.
3. **Private Data**: They cannot answer questions about your company's proprietary internal documents.

To solve this without retraining or fine-tuning the massive model (which is too expensive), we use **RAG**.

---

## 2. What is RAG?

**RAG (Retrieval-Augmented Generation)** is an architecture that provides an LLM with relevant, up-to-date, or private information retrieved from an external database just before the LLM generates its response.

Instead of asking: *"What is our company's refund policy?"* (which the LLM doesn't know), you ask: 
*"Based on the following document: [insert retrieved document], what is our company's refund policy?"*

---

## 3. The RAG Pipeline

The standard RAG pipeline consists of two phases: Data Ingestion and Retrieval-Generation.

### Phase 1: Data Ingestion (Offline)
1. **Extract**: Load documents (PDFs, Confluence pages, HTML).
2. **Chunk**: Split the documents into smaller text chunks (e.g., 500 tokens each). LLMs have finite context windows.
3. **Embed**: Pass each chunk through an Embedding Model (e.g., `text-embedding-ada-002`) to convert the text into a dense vector (e.g., 1536 dimensions).
4. **Index**: Store the vectors and their corresponding text chunks in a **Vector Database**.

### Phase 2: Retrieval and Generation (Runtime)
1. **Query**: The user asks a question.
2. **Embed Query**: Convert the user's question into a vector using the exact same Embedding Model.
3. **Retrieve**: The Vector DB performs a similarity search (usually Cosine Similarity) to find the top-$K$ chunks closest to the user's query vector in the 1536-dimensional space.
4. **Generate**: The retrieved chunks are inserted into a prompt template alongside the user's original query. The LLM reads the context and generates an accurate answer.

---

## 4. Vector Databases

Standard relational databases (SQL) are built for exact keyword matches (`WHERE text = 'refund'`). Vector Databases (like ChromaDB, Pinecone, FAISS) are built for **semantic similarity search**. 

They search for meaning, not keywords. "Money back" will map very closely to "refund" in the embedding space.

To perform similarity search across millions of vectors quickly, Vector DBs use **Approximate Nearest Neighbors (ANN)** algorithms. The most popular is **HNSW (Hierarchical Navigable Small World)** graphs, which builds a multi-layered graph to jump quickly to the neighborhood of the query vector.

---

## 5. Implementation Example

Here is a minimal RAG pipeline using LangChain and ChromaDB.

```python
# pip install langchain chromadb sentence-transformers
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 1. Load Document
loader = TextLoader("company_policy.txt")
docs = loader.load()

# 2. Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 3. Embed and Index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(chunks, embeddings)

# 4. Retrieval
query = "What is the refund policy for defective items?"
retrieved_docs = vector_db.similarity_search(query, k=2)

print("Context found:")
for doc in retrieved_docs:
    print("-", doc.page_content)

# 5. Generation (Conceptual - requires LLM API)
# prompt = f"Based on this context: {retrieved_docs[0].page_content}, answer: {query}"
# response = llm(prompt)
```

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Write a script that uses FAISS and SentenceTransformers to find the most similar sentence in a text file to a user's input.
- 🟡 **Intermediate**: Build a full RAG Chatbot using Streamlit, LangChain, and the OpenAI API that can answer questions about a specific PDF.

### What's Next
| Next | Why |
|------|-----|
| [Autoencoders](../10-Generative/01-Autoencoders.md) | Shift focus to Generative AI architectures. |

---

[← Multi Modal Models](./08-Multi-Modal-Models.md) | [Back to Index](../README.md) | [Next: Autoencoders →](../10-Generative/01-Autoencoders.md)
