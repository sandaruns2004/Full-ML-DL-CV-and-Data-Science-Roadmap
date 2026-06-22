# 🗄️ RAG & Vector Databases

> **Prerequisites**: Transformer Architecture, Word Embeddings | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Limitations of Vanilla LLMs](#1-the-limitations-of-vanilla-llms)
2. [What is RAG?](#2-what-is-rag)
3. [The RAG Pipeline Deep Dive](#3-the-rag-pipeline-deep-dive)
4. [Vector Databases: ChromaDB & Pinecone](#4-vector-databases-chromadb--pinecone)
5. [LangChain & LlamaIndex Integrations](#5-langchain--llamaindex-integrations)
6. [Implementation Example: Full RAG Pipeline](#6-implementation-example-full-rag-pipeline)
7. [Advanced RAG Techniques](#7-advanced-rag-techniques)

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

## 3. The RAG Pipeline Deep Dive

The standard RAG pipeline consists of two phases: Data Ingestion and Retrieval-Generation.

### Phase 1: Data Ingestion (Offline)
1. **Extract**: Load documents (PDFs, Confluence pages, HTML) using Document Loaders.
2. **Chunk**: Split the documents into smaller text chunks (e.g., 500 tokens each) using a Text Splitter. LLMs have finite context windows.
3. **Embed**: Pass each chunk through an Embedding Model (e.g., `text-embedding-ada-002`, `all-MiniLM-L6-v2`) to convert the text into a dense vector (e.g., 1536 dimensions).
4. **Index**: Store the vectors and their corresponding text chunks in a **Vector Database**.

### Phase 2: Retrieval and Generation (Runtime)
1. **Query**: The user asks a question.
2. **Embed Query**: Convert the user's question into a vector using the exact same Embedding Model.
3. **Retrieve**: The Vector DB performs a similarity search (usually Cosine Similarity) to find the top-$K$ chunks closest to the user's query vector in the dimensional space.
4. **Generate**: The retrieved chunks are inserted into a prompt template alongside the user's original query. The LLM reads the context and generates an accurate answer.

---

## 4. Vector Databases: ChromaDB & Pinecone

Standard relational databases (SQL) are built for exact keyword matches. Vector Databases are built for **semantic similarity search**. 

To perform similarity search across millions of vectors quickly, Vector DBs use **Approximate Nearest Neighbors (ANN)** algorithms. The most popular is **HNSW (Hierarchical Navigable Small World)** graphs, which builds a multi-layered graph to jump quickly to the neighborhood of the query vector.

### ChromaDB
- **Type**: Open-source, local/embedded vector database.
- **Use Case**: Rapid prototyping, local AI applications, testing.

### Pinecone
- **Type**: Fully managed, cloud-native vector database.
- **Use Case**: Production-grade applications, massive scale, enterprise deployments.

---

## 5. LangChain & LlamaIndex Integrations

Building RAG from scratch requires a lot of boilerplate. We use frameworks to simplify it.

### LangChain
LangChain is a generic framework for developing applications powered by language models. It provides abstractions for Chains, Agents, Prompts, Memory, and Vector Stores.

### LlamaIndex
LlamaIndex (formerly GPT Index) is specifically hyper-focused on connecting LLMs to external data sources. It offers advanced indexing strategies (Tree Index, Keyword Index) tailored for complex RAG.

---

## 6. Implementation Example: Full RAG Pipeline

Here is a minimal RAG pipeline using **LangChain** and **ChromaDB**.

```python
# pip install langchain chromadb sentence-transformers
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# 1. Load Document
loader = TextLoader("company_policy.txt")
docs = loader.load()

# 2. Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 3. Embed and Index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(chunks, embeddings)

# 4. Retrieval & QA Setup
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 5. Generation Setup
# Note: You can replace HuggingFacePipeline with ChatOpenAI if using OpenAI API
llm = HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever
)

# 6. Execute RAG
query = "What is the refund policy for defective items?"
response = qa_chain.run(query)
print("Answer:", response)
```

---

## 7. Advanced RAG Techniques

Standard RAG ("Naive RAG") often fails on complex questions. Advanced techniques include:
- **Query Transformation**: Rewriting the user's query to improve retrieval (e.g., Multi-Query, HyDE).
- **Re-ranking**: Retrieving a large number of chunks (e.g., 20), then using a cross-encoder model to re-rank and pick the top 5 most relevant.
- **Parent-Child Retrieval**: Embedding small sentences for highly precise semantic matching, but retrieving the larger parent paragraph to give the LLM full context.

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Write a script that uses FAISS and SentenceTransformers to find the most similar sentence in a text file to a user's input.
- 🟡 **Intermediate**: Build a full RAG Chatbot using Streamlit, LangChain, and a local open-source model (like LLaMA-3 via Ollama) to answer questions about a PDF.
- 🔴 **Advanced**: Implement a RAG system using Pinecone with Re-ranking (Cohere) and Query Transformations to query complex financial reports.

### What's Next
| Next | Why |
|------|-----|
| [LangChain And Agents](./10-LangChain-And-Agents.md) | Learn how to give LLMs tools and autonomy. |

---

[← Multi-Modal Models](08-Multi-Modal-Models.md) | [Back to Index](../README.md) | [Next: LangChain & Agents →](10-LangChain-And-Agents.md)
