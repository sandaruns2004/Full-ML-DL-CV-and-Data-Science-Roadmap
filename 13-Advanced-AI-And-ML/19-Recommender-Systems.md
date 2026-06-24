# 19 - Recommender Systems

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Prerequisites**: Deep Learning Foundations | **Estimated Reading Time**: 30 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Content-Based Filtering](#2-content-based-filtering)
3. [Collaborative Filtering (Matrix Factorization)](#3-collaborative-filtering-matrix-factorization)
4. [Deep Learning in Recommender Systems (DLRM)](#4-deep-learning-in-recommender-systems-dlrm)
5. [The Two-Tower Model Architecture](#5-the-two-tower-model-architecture)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

The internet contains infinite choices. YouTube has 800 million videos. Amazon has 350 million products. Netflix has 17,000 titles.

### 🟢 Beginner
If a user logs into Netflix and is forced to scroll through 17,000 movies alphabetically to find something to watch, they will cancel their subscription. The platform must instantly predict what the user wants to see *before* they even search for it.

### 🟡 Intermediate
To do this, we build **Recommender Systems**. These algorithms analyze the user's past behavior (clicks, watch time, purchases) and compare it against the metadata of millions of items to rank the top 10 most relevant items for that specific user.

### 🔴 Advanced
Modern Recommender Systems are the most financially valuable AI models on the planet. They drive 70% of YouTube's views and 35% of Amazon's revenue. Solving this problem mathematically requires moving beyond simple content matching. We must map millions of discrete Users and millions of discrete Items into the exact same continuous Vector Latent Space using massive Deep Learning architectures like **Two-Tower Models** and **DLRMs (Deep Learning Recommendation Models)**.

---

# 2. Content-Based Filtering

The simplest form of recommendation is **Content-Based Filtering**.

**The Logic:** "If you liked Item A, we will recommend Item B because Item B has similar features."
*   *Example:* If you watch a 20-minute video tagged `[Sci-Fi, Space, Tutorial]`, the system searches the database for other videos tagged `[Sci-Fi, Space, Tutorial]`.

**The Flaw:**
It traps users in a "Filter Bubble". If you only watch Sci-Fi videos, the system will *only* recommend Sci-Fi videos. It will never recommend a Cooking video, even if a highly-rated Cooking video is going viral and you might love it. It cannot discover latent user interests.

---

# 3. Collaborative Filtering (Matrix Factorization)

To break the Filter Bubble, we use **Collaborative Filtering**.

**The Logic:** "If User A and User B both liked the same 5 movies, they have similar taste. Therefore, if User A watches Movie 6 and loves it, we will recommend Movie 6 to User B."

We don't need to know *anything* about the movie (no tags, no genres). We only look at user interactions.

### The Mathematics: Matrix Factorization
We build a massive User-Item Matrix.
*   Rows = 100 Million Users.
*   Columns = 10,000 Movies.
*   Cells = The 1-5 Star Rating the user gave the movie.

This matrix is 99% empty (most users haven't seen most movies). 
We use linear algebra (**Singular Value Decomposition** or Gradient Descent) to factor this massive, empty matrix into two much smaller, dense matrices:
1.  **A User Matrix** (Every user gets a vector of size 50).
2.  **An Item Matrix** (Every movie gets a vector of size 50).

If we take the Dot Product of User A's vector and Movie Z's vector, the result is the predicted Star Rating! 
This algorithm famously won the $1 Million Netflix Prize in 2009.

---

# 4. Deep Learning in Recommender Systems (DLRM)

Matrix Factorization is powerful, but it only looks at a single interaction (Rating). 

In reality, a user click is driven by hundreds of contextual features: The time of day, the user's age, the user's location, the device they are using (Mobile vs. Desktop), and their watch history over the last 10 minutes.

Linear Algebra cannot handle that level of complexity. We need Deep Learning.

Meta (Facebook) open-sourced the **DLRM (Deep Learning Recommendation Model)** architecture.
1.  **Dense Features:** Continuous numbers (e.g., User Age, Video Length) are fed directly into a Multi-Layer Perceptron (MLP).
2.  **Sparse Features:** Categorical IDs (e.g., User ID, Video ID, City ID) are passed through **Embedding Tables** to convert them into dense vectors.
3.  The Dense features and Sparse embeddings are concatenated together, and passed through a final massive Neural Network to output the probability that the user will click on the item.

---

# 5. The Two-Tower Model Architecture

Running a massive DLRM neural network on 800 million YouTube videos every time a user refreshes the page is computationally impossible. It would take an hour to load the page.

Production Recommender Systems use a two-stage pipeline: **Retrieval $\to$ Ranking**.

### Stage 1: Retrieval (The Two-Tower Model)
The goal is to filter 800 million videos down to 1,000 relevant candidates in 10 milliseconds.
We use a **Two-Tower Neural Network**:
*   **User Tower:** A neural network that processes User Features (History, Age) and outputs a single `User Vector`.
*   **Item Tower:** A separate neural network that processes Item Features (Tags, Title) and outputs a single `Item Vector`.

We pass all 800 million items through the Item Tower offline and store the vectors in an Approximate Nearest Neighbor (ANN) database like FAISS. 
At runtime, we pass the User through the User Tower once. We then instantly query the FAISS database to find the 1,000 closest Item Vectors.

### Stage 2: Ranking
Now that we only have 1,000 candidates, we pass them through a massive, heavy, highly-accurate DLRM (like we built in Section 4). The DLRM scores all 1,000 candidates, sorts them by probability, and returns the Top 10 to the user's screen.

---

# 6. Key Takeaways

*   **Recommender Systems** solve the problem of information overload on massive platforms.
*   **Content-Based Filtering** recommends items based on their metadata, but suffers from the Filter Bubble.
*   **Collaborative Filtering (Matrix Factorization)** recommends items based on the behavior of similar users, requiring no metadata.
*   **Deep Learning Recommendation Models (DLRMs)** combine dense contextual features with sparse embeddings to predict click-through rates with extreme accuracy.
*   Production systems use a two-stage pipeline: **Retrieval** (using fast Two-Tower models) followed by **Ranking** (using heavy DLRM models).

---

# 7. Next Topic

We have just seen how to analyze sequences of user behavior to predict their next click. 

But sequences of data are not just for recommending videos. They are the foundation of predicting the stock market, weather patterns, and heart rates. 

In the next lesson, we will explore advanced architectures for forecasting the future in **Time-Series Deep Dive**.

[← Meta-Learning](18-Meta-Learning.md) | [Back to Index](README.md) | [Next Topic: Time-Series Deep Dive →](20-Time-Series-Deep-Dive.md)
