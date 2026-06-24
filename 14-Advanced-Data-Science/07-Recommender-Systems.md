# Recommender Systems

## What Problem Does This Solve?

In the physical world, a bookstore can only display 10,000 books. In the digital world, Amazon offers over 30 million books. 

When humans are faced with infinite choice, they suffer from **Decision Paralysis** and often buy nothing. Recommender systems solve the problem of infinite inventory by filtering it down to the 5 items a specific user is most likely to buy right now.

According to McKinsey, recommender systems drive:
- 35% of what consumers purchase on Amazon.
- 75% of what they watch on Netflix.
- 80% of the content consumed on YouTube.

Recommender systems are the financial engines of the modern internet.

---

## Core Concepts & Methods

There are two primary ways to build a recommendation engine, and modern systems usually combine both into a **Hybrid System**.

### 1. Content-Based Filtering
Recommend items that are *similar to the items* the user has liked in the past.

- **How it works:** If you watch a Sci-Fi movie directed by Christopher Nolan starring Matthew McConaughey, the system looks at the metadata (Genre=Sci-Fi, Director=Nolan, Actor=McConaughey) and recommends *Inception*.
- **Mathematics:** It relies heavily on measuring the distance between item vectors. We calculate the **Cosine Similarity** between the TF-IDF vectors of item descriptions.
- **Pros:** Does not suffer from the "Cold Start" problem for new items. If a new movie is added to the database, it can be recommended immediately based on its metadata.
- **Cons:** Creates a "filter bubble." If you only watch Sci-Fi, the system will never recommend a Comedy, because it has no idea what other genres you might like.

### 2. Collaborative Filtering
Recommend items that *similar users* have liked. It completely ignores the metadata of the item itself.

- **How it works:** User A likes movies 1, 2, and 3. User B likes movies 1, 2, and 4. The system calculates that User A and User B are mathematically similar. Therefore, it recommends movie 4 to User A, and movie 3 to User B.
- **Pros:** Can recommend completely unexpected items (breaking the filter bubble). It doesn't need to know anything about the movie; it just relies on human behavior.
- **Cons:** Suffers from the **Cold Start** problem. If a new movie is added to the platform with 0 ratings, collaborative filtering cannot recommend it to anyone.

### 3. Matrix Factorization (The Netflix Prize Winner)
The most powerful form of collaborative filtering. Imagine a giant matrix where rows are Users, columns are Movies, and the values are ratings (1 to 5 stars). 

99% of this matrix is empty (sparse) because most users haven't seen most movies. 

**Matrix Factorization (e.g., SVD - Singular Value Decomposition)** decomposes this massive empty matrix into two smaller, dense matrices:
1. A User Matrix (users mapped to latent features).
2. An Item Matrix (movies mapped to latent features).

When you multiply these two smaller matrices back together, you recreate the original matrix, *but all the empty spaces are now filled with predicted ratings!*

---

## Workflow: Building a Hybrid Recommender

1. **Candidate Generation:** You have 30 million items. You cannot run a complex deep learning model on all of them in 50 milliseconds. First, use fast heuristics (like Collaborative Filtering or simple SQL queries) to filter 30 million items down to 1,000 candidates.
2. **Scoring / Ranking:** Run a heavy Machine Learning model (like XGBoost or a Neural Network) on those 1,000 candidates to predict the exact probability the user will click on them. Sort them by probability.
3. **Re-ranking / Business Logic:** Filter out items they have already bought, ensure diversity (don't show 5 red shirts in a row), and boost items that are currently on sale.
4. **Serving:** Return the top 5 items to the UI.

---

## From Scratch Implementation: Cosine Similarity (Content-Based)

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Our database of items
movies = pd.DataFrame({
    'title': ['The Matrix', 'Inception', 'The Notebook', 'Interstellar'],
    'description': [
        'A hacker discovers reality is a simulation.',
        'Thieves steal secrets from dreams within dreams.',
        'A poor man and rich woman fall in love.',
        'Astronauts travel through a wormhole to save humanity.'
    ]
})

# 2. Convert text descriptions into mathematical vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['description'])

# 3. Calculate similarity between all movies
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Create a function to recommend similar movies
def recommend(movie_title, sim_matrix, df):
    idx = df[df['title'] == movie_title].index[0]
    # Get similarity scores for this movie
    scores = list(enumerate(sim_matrix[idx]))
    # Sort by similarity
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # Get the top recommendation (excluding the movie itself)
    top_movie_idx = scores[1][0]
    return df['title'].iloc[top_movie_idx]

print("Because you watched 'Inception', you might like:")
print(recommend('Inception', similarity_matrix, movies))
# Output will correctly suggest The Matrix or Interstellar based on shared keywords.
```

---

## Common Failure Cases

1. **The Cold Start Problem:** As mentioned, new users have no history, and new items have no ratings. Systems usually solve this by asking new users to "Pick 3 genres you like" during onboarding, or falling back to "Top Trending in your Country."
2. **Feedback Loops and Echo Chambers:** If YouTube recommends conspiracy theories to a user, and the user watches them, the system assumes they want more. This creates a radicalization feedback loop.
3. **Implicit vs. Explicit Feedback:** A user might rate a documentary 5 stars (Explicit), but actually spend 4 hours a night watching trashy reality TV (Implicit). Recommender systems generally prioritize implicit behavioral data over what users *say* they like.

---

## Industry Applications

- **Spotify:** "Discover Weekly" uses advanced Matrix Factorization combined with Audio Analysis (Deep Learning CNNs listening to the actual audio waves) to find similar songs even if nobody has listened to them yet.
- **TikTok:** Uses a massive Deep Neural Network. Because videos are short, the system receives feedback every few seconds (watch time, swipe, like), allowing the recommender to adapt to user mood in real-time.

---

## Key Takeaways

1. Content-based filtering relies on item metadata.
2. Collaborative filtering relies on user behavior.
3. Matrix Factorization fills in the blanks of sparse user-item matrices.
4. Production systems use a two-stage pipeline: fast Candidate Generation followed by heavy ML Ranking.

## Next Topic

Recommender systems personalize the app. But how do we understand the users themselves? How much is a user worth to the company over their lifetime, and how do we prevent them from leaving? We step into the realm of Customer Analytics.

Navigation:

[← Previous Topic](./06-Anomaly-Detection.md) | [Back to Index](./README.md) | [Next Topic: Customer Analytics →](./08-Customer-Analytics.md)
