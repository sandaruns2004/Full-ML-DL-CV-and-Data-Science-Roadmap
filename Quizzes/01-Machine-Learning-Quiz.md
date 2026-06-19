# 🧠 Quiz: Machine Learning Fundamentals

> Test your knowledge of Supervised, Unsupervised, and Classical ML Algorithms! Click the `► Show Answer` toggle to check your work.

---

### Question 1: Linear Regression Assumptions
Which of the following is **NOT** a required assumption for standard Ordinary Least Squares (OLS) Linear Regression?

A) The relationship between X and y is linear.  
B) The features in X must be perfectly normally distributed.  
C) The residuals (errors) must have constant variance (Homoscedasticity).  
D) There must be little to no multicollinearity among the features.  

<details>
<summary><b>► Show Answer</b></summary>
<br>
<b>Correct Answer: B</b><br><br>
OLS does <i>not</i> require the predictor variables (X) to be normally distributed. It only assumes that the <i>residuals</i> (errors) are normally distributed, which is necessary for calculating p-values and confidence intervals, but not strictly required to fit the regression line.
</details>

---

### Question 2: The Bias-Variance Tradeoff
You train a Random Forest model on your data. The training accuracy is 99%, but the test accuracy is 65%. What is the state of your model, and how should you fix it?

A) High Bias (Underfitting) - Increase the maximum depth of the trees.  
B) High Variance (Overfitting) - Increase the maximum depth of the trees.  
C) High Variance (Overfitting) - Decrease the maximum depth of the trees.  
D) High Bias (Underfitting) - Decrease the maximum depth of the trees.  

<details>
<summary><b>► Show Answer</b></summary>
<br>
<b>Correct Answer: C</b><br><br>
A massive gap between training accuracy (99%) and test accuracy (65%) means the model has memorized the training data but fails to generalize. This is High Variance (Overfitting). To fix overfitting in a Decision Tree/Random Forest, you must <i>restrict</i> the model's complexity by decreasing the maximum depth, increasing the minimum samples per leaf, etc.
</details>

---

### Question 3: Dimensionality Reduction
You want to reduce a 100-dimensional dataset to 2 dimensions for visualization. You need to preserve the *global* linear structure and variance of the data as much as possible. Which algorithm should you use?

A) Principal Component Analysis (PCA)  
B) t-SNE  
C) K-Means Clustering  
D) UMAP  

<details>
<summary><b>► Show Answer</b></summary>
<br>
<b>Correct Answer: A</b><br><br>
PCA is a linear dimensionality reduction technique designed specifically to maximize the preserved global variance. While t-SNE and UMAP make much prettier 2D plots, they are manifold learning techniques that focus on preserving <i>local</i> neighbor structures, often severely distorting the global distances and relationships. K-Means is a clustering algorithm, not dimensionality reduction.
</details>

---

### Question 4: Dealing with Imbalanced Data
You are building a fraud detection system. 99% of transactions are legitimate, and 1% are fraud. You train a Logistic Regression model that simply predicts "Legitimate" every single time. What will be the Accuracy and Recall for the Fraud class?

A) Accuracy: 99%, Fraud Recall: 0%  
B) Accuracy: 1%, Fraud Recall: 99%  
C) Accuracy: 99%, Fraud Recall: 99%  
D) Accuracy: 50%, Fraud Recall: 0%  

<details>
<summary><b>► Show Answer</b></summary>
<br>
<b>Correct Answer: A</b><br><br>
Because the model always predicts "Legitimate", it will be right 99% of the time (Accuracy = 99%). However, Recall measures what percentage of the actual Fraud cases the model successfully found. Since it predicted 0 frauds, it found 0% of the actual frauds. This highlights why Accuracy is a terrible metric for imbalanced data!
</details>

---

### Question 5: Ensemble Methods
Which of the following statements about Gradient Boosting and Random Forests is TRUE?

A) Random Forests build trees sequentially, while Gradient Boosting builds trees in parallel.  
B) Random Forests use deep, complex trees, while Gradient Boosting usually uses shallow "weak learners" (stumps).  
C) Gradient Boosting averages the final output of all trees, while Random Forests add the outputs together.  
D) Random Forests are highly prone to overfitting, while Gradient Boosting cannot overfit.  

<details>
<summary><b>► Show Answer</b></summary>
<br>
<b>Correct Answer: B</b><br><br>
Random Forests train deep, overfitted trees independently in parallel, and average them to reduce variance. Gradient Boosting (like XGBoost) trains shallow, weak trees sequentially, where each new tree tries to predict and correct the residual errors of the previous trees.
</details>

---

[← Deep Learning Reference](../Cheat-Sheets/02-Deep-Learning-Reference.md) | [Back to Index](../README.md) | [Next: Deep Learning Quiz →](./02-Deep-Learning-Quiz.md)
