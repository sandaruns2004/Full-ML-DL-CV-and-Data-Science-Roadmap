# 🌱 Beginner Projects

> **Target**: Master the absolute basics of Scikit-Learn, Pandas, and EDA.

Welcome to the Projects Section! Doing tutorials will only get you so far. The only way to truly master Machine Learning is to build end-to-end projects without looking at the answer key.

Below are three beginner projects designed to solidify your understanding of Supervised and Unsupervised Learning.

---

## 1. The Real Estate Price Predictor

**Goal**: Build a complete end-to-end regression pipeline to predict house prices.

**The Dataset**: The Ames Housing Dataset (available on Kaggle). It has 79 explanatory variables covering almost every aspect of residential homes.

**Requirements**:
1. **EDA**: Create at least 5 different visualizations using Seaborn to find which variables correlate most strongly with Sale Price.
2. **Feature Engineering**: 
   - Handle missing values (e.g., impute missing `LotFrontage` with the median of the neighborhood).
   - Create a new feature: `TotalSquareFootage` by adding `1stFlrSF` + `2ndFlrSF` + `TotalBsmtSF`.
3. **Encoding**: Use `pd.get_dummies()` or `OneHotEncoder` to handle categorical variables like `Neighborhood` and `HouseStyle`.
4. **Modeling**: Train a Ridge Regression model and a Random Forest Regressor. 
5. **Evaluation**: Calculate the Root Mean Squared Error (RMSE) on a 20% test set.

**Bonus Challenge**: Use `GridSearchCV` to find the absolute best `max_depth` and `n_estimators` for your Random Forest.

---

## 2. Customer Segmentation Engine

**Goal**: Use Unsupervised Learning to group retail customers into distinct marketing personas based on their purchasing behavior.

**The Dataset**: The "Mall Customer Segmentation Data" (available on Kaggle). Features include Age, Annual Income, and Spending Score.

**Requirements**:
1. **Scaling**: Use `StandardScaler` to ensure Age and Annual Income are on the same scale, so the distance metrics work correctly.
2. **K-Means Clustering**: Train a K-Means model on the dataset.
3. **The Elbow Method**: Write a `for` loop that trains K-Means with $K=1$ through $K=10$, recording the Inertia (Within-Cluster-Sum-of-Squares). Plot the results and visually identify the "elbow" to pick the optimal number of clusters.
4. **Visualization**: Use `matplotlib` to create a 2D scatter plot (Annual Income vs Spending Score) where each dot is colored by its assigned Cluster ID.

**Bonus Challenge**: Use DBSCAN instead of K-Means. Try to tune the `eps` and `min_samples` parameters so it successfully finds the clusters without forcing outliers into them.

---

## 3. The Sentiment Analyzer

**Goal**: Build a classic NLP Text Classification model that predicts if a movie review is positive or negative.

**The Dataset**: The IMDB Movie Reviews dataset (50,000 reviews).

**Requirements**:
1. **Text Preprocessing**: Write a Python function that uses Regular Expressions (`re` library) to remove all HTML tags (`<br />`), punctuation, and converts the text to lowercase.
2. **Vectorization**: Use Scikit-Learn's `TfidfVectorizer` to convert the raw text strings into a massive sparse matrix of word frequencies. Restrict it to the top 5,000 most frequent words.
3. **Modeling**: Train a `MultinomialNB` (Naive Bayes) classifier and a `LogisticRegression` classifier.
4. **Evaluation**: Print the `classification_report` showing Precision, Recall, and F1-Score for both positive and negative classes.

**Bonus Challenge**: Write a custom script where you can type your own review into the terminal (e.g., `input("Enter review: ")`), and the script passes it through your saved TF-IDF vectorizer and Model to instantly output "Positive" or "Negative".

---

[← Data Versioning DVC](../15-ML-In-Production/08-Data-Versioning-DVC.md) | [Back to Index](../README.md) | [Next: Intermediate Projects →](./02-Intermediate-Projects.md)
