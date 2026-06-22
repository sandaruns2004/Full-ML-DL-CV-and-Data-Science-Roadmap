# 🛍️ Deep Recommender Systems

> **Prerequisites**: Matrix Math, Deep Learning, Embeddings | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Billion-Dollar Problem](#1-the-billion-dollar-problem)
2. [Collaborative Filtering & Matrix Factorization](#2-collaborative-filtering--matrix-factorization)
3. [Neural Collaborative Filtering (NCF)](#3-neural-collaborative-filtering-ncf)
4. [Two-Tower Architecture (The Industry Standard)](#4-two-tower-architecture-the-industry-standard)
5. [The Cold Start Problem](#5-the-cold-start-problem)
6. [From-Scratch Implementation (PyTorch)](#6-from-scratch-implementation-pytorch)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Billion-Dollar Problem

Recommender Systems are arguably the most commercially successful application of Machine Learning in history.
- 35% of Amazon's revenue comes from recommendations.
- 75% of what people watch on Netflix comes from recommendations.
- YouTube and TikTok algorithms are entirely driven by massive recommender pipelines.

**The Problem**: Given a User $U$ and an Item $I$, predict the rating the user will give the item (or predict the probability the user will click on the item).

---

## 2. Collaborative Filtering & Matrix Factorization

Before Deep Learning, the gold standard was **Matrix Factorization** (which famously won the $1M Netflix Prize in 2009).

Imagine a giant matrix $R$ where rows are Users and columns are Movies. The values are 1-5 star ratings. This matrix is 99% empty (sparse) because a single user only watches a fraction of all movies.

Matrix Factorization decomposes this giant matrix $R$ into two smaller, dense matrices:
- A User Embedding Matrix $P$ (size: Users $\times$ K)
- An Item Embedding Matrix $Q$ (size: Items $\times$ K)

Where K is the embedding dimension (e.g., K=50).
To predict the rating User $i$ gives Item $j$, you just take the dot product of their embeddings:
$$ \hat{r}_{i,j} = \mathbf{p}_i \cdot \mathbf{q}_j $$

By using Gradient Descent to minimize the Mean Squared Error between the dot products and the known ratings, the embeddings magically learn to represent latent concepts (e.g., Dimension 1 becomes "How much does this movie have explosions?", and User embedding Dimension 1 becomes "How much does this user like explosions?").

---

## 3. Neural Collaborative Filtering (NCF)

In 2017, researchers realized that a simple dot product $\mathbf{p}_i \cdot \mathbf{q}_j$ is a linear operation, heavily restricting the complex interactions the model can learn.

**Neural Collaborative Filtering (NCF)** replaces the dot product with a Neural Network.

**The Architecture:**
1. Pass the User ID through a User Embedding Layer to get $\mathbf{p}_i$.
2. Pass the Item ID through an Item Embedding Layer to get $\mathbf{q}_j$.
3. Concatenate the two vectors: $[\mathbf{p}_i, \mathbf{q}_j]$.
4. Pass the concatenated vector through a Multi-Layer Perceptron (MLP) with ReLU activations.
5. The final output layer is a single neuron (with Sigmoid for click-prediction, or linear for rating prediction).

The MLP can learn highly complex, non-linear interactions between users and items that a simple dot product misses entirely.

---

## 4. Two-Tower Architecture (The Industry Standard)

If YouTube has 2 Billion users and 1 Billion videos, running the NCF MLP 1 Billion times for every single user refresh is computationally impossible.

The modern industry standard for retrieval is the **Two-Tower Architecture**.

1. **User Tower**: A Neural Network that takes User ID, Age, Search History, Watch History, etc., and outputs a single 128-dimensional **User Vector**.
2. **Item Tower**: A Neural Network that takes Video ID, Title, Tags, Thumbnail CNN embeddings, etc., and outputs a single 128-dimensional **Item Vector**.

**The Trick**: The two towers NEVER interact during the neural network forward pass. The only interaction happens at the very end via a strict Dot Product (or Cosine Similarity).

**Why this is genius:**
- You can pre-compute the Item Vectors for all 1 Billion videos and store them in an Approximate Nearest Neighbor (ANN) database like FAISS or Pinecone.
- When a user logs in, you run the User Tower *once* to get their User Vector (128d).
- You query the FAISS database to find the top 100 Item Vectors that have the highest dot product with the User Vector. This search takes less than 10 milliseconds!

*(Note: Recommender systems use a 2-stage pipeline. The Two-Tower model retrieves 100 candidates. Then, a heavy NCF model "Ranks" those 100 candidates to pick the final top 10).*

---

## 5. The Cold Start Problem

The biggest issue in Recommender Systems is the **Cold Start Problem**.
- **New User**: A user creates an account today. They have no watch history. Their User Embedding is completely random.
- **New Item**: A video is uploaded today. It has zero views. Its Item Embedding is completely random.

**Solutions:**
1. **Content-Based Filtering**: Don't rely purely on User/Item IDs. Incorporate metadata into the Two-Tower model. If a new video has the tag "Minecraft", the Item Tower can look at the "Minecraft" tag embedding and recommend it to users who like gaming.
2. **Heuristics**: Ask users to pick 5 genres they like during onboarding. Recommend globally popular items until they generate enough personal click data.

---

## 6. From-Scratch Implementation (PyTorch)

Let's implement a PyTorch NCF model to predict MovieLens ratings.

```python
import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, hidden_layers=[64, 32, 16]):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Embedding Layers (The Lookup Tables)
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        # Neural Network Layers
        # Input size is embed_dim * 2 because we concatenate user and item
        self.fc1 = nn.Linear(embed_dim * 2, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        
        # Final prediction layer (outputs a single number: the rating)
        self.output = nn.Linear(hidden_layers[2], 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, user_indices, item_indices):
        # 1. Get Embeddings
        u_embed = self.user_embedding(user_indices) # (Batch, embed_dim)
        i_embed = self.item_embedding(item_indices) # (Batch, embed_dim)
        
        # 2. Concatenate
        x = torch.cat([u_embed, i_embed], dim=1)    # (Batch, embed_dim * 2)
        
        # 3. MLP Layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        # 4. Output (1-5 stars)
        rating = self.output(x)                     # (Batch, 1)
        
        # We can use Sigmoid and scale to [1, 5] to bound the outputs, 
        # but pure linear output with MSE loss works fine too!
        return rating.squeeze()

# --- Simulation ---
num_users = 10000
num_items = 5000
model = NeuralCollaborativeFiltering(num_users, num_items)

# Batch of 4 users watching 4 specific items
dummy_users = torch.tensor([42, 105, 888, 9999])
dummy_items = torch.tensor([5, 800, 1024, 4000])

predictions = model(dummy_users, dummy_items)
print("Predicted Ratings for the batch:")
print(predictions.detach().numpy())
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **MovieLens Matrix Factorization**: Download the `MovieLens 100k` dataset. Build a PyTorch model that implements pure Matrix Factorization (just dot products, no MLP). Train it using MSE loss and predict movies a specific user will rate 5 stars.
- 🟡 **Two-Tower Retrieval**: Build a Two-Tower model using the MovieLens dataset, incorporating metadata (Movie Genres and User Age/Occupation). Train the towers using a Contrastive Loss function (pulling clicked pairs together, pushing unclicked pairs apart). Extract the embeddings and load them into `FAISS` to perform ultra-fast neighbor retrieval.

### What's Next
| Next | Why |
|------|-----|
| [Time Series Deep Dive](./06-Time-Series-Deep-Dive.md) | How do we predict stock prices, weather, or server loads? We need to look at sequence modeling without NLP. We dive into ARIMA, Prophet, and Temporal Fusion Transformers. |

---

[← Meta-Learning](04-Meta-Learning.md) | [Back to Index](../README.md) | [Next: Time Series →](06-Time-Series-Deep-Dive.md)
