import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_blobs, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import missingno as msno

st.set_page_config(page_title="Machine Learning Intro", layout="wide")

st.title("Machine Learning Introduction Dashboard")
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a Section:", [
    "Supervised & Unsupervised Learning", 
    "Gradient Descent from Scratch", 
    "Breast Cancer ML Pipeline", 
    "Exploratory Data Analysis (EDA)"
])

if section == "Supervised & Unsupervised Learning":
    st.header("Supervised Learning: Iris Classification")
    st.write("Training a Decision Tree on the Iris dataset.")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    st.success(f"Decision Tree Accuracy: {acc:.2%}")
    
    st.markdown("---")
    
    st.header("Unsupervised Learning: K-Means Clustering")
    st.write("Clustering generated data into 7 clusters using K-Means.")
    
    X_blobs, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.8, random_state=42)
    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
    predictions_kmeans = kmeans.fit_predict(X_blobs)
    
    fig, ax = plt.subplots()
    ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=predictions_kmeans, cmap='viridis', alpha=0.5)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='X', s=200, label='Centroids')
    ax.set_title('K-Means Clustering')
    ax.legend()
    st.pyplot(fig)

elif section == "Gradient Descent from Scratch":
    st.header("Gradient Descent for Linear Regression")
    
    np.random.seed(42)
    X = 2 * np.random.rand(100)
    y = 4 + 3 * X + np.random.randn(100) * 0.5  
    
    epochs = st.slider("Number of Epochs", min_value=10, max_value=200, value=100, step=10)
    lr = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    
    if st.button("Run Gradient Descent"):
        w, b = 0.0, 0.0
        n = len(X)
        losses = []
        
        progress_bar = st.progress(0)
        
        for epoch in range(epochs):
            y_pred = w * X + b
            loss = np.mean((y - y_pred) ** 2)
            losses.append(loss)
            
            dw = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
            
            w -= lr * dw
            b -= lr * db
            
            progress_bar.progress((epoch + 1) / epochs)
            
        st.write(f"**Learned Equation:** `y = {w:.2f}x + {b:.2f}`")
        st.write("**True Equation:** `y = 3.00x + 4.00`")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(losses, color='orange')
        ax[0].set_title('Loss over Epochs')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Mean Squared Error')
        
        ax[1].scatter(X, y, label='Data', color='blue', alpha=0.6)
        x_vals = np.linspace(0, 2, 100)
        y_vals = w * x_vals + b
        ax[1].plot(x_vals, y_vals, color='red', label='Learned Line')
        ax[1].set_title('Data and Fitted Line')
        ax[1].legend()
        
        st.pyplot(fig)

elif section == "Breast Cancer ML Pipeline":
    st.header("Complete ML Pipeline: Breast Cancer Classification")
    
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    st.write(f"**Dataset shape:** {X.shape}")
    if st.checkbox("Show dataset head"):
        st.dataframe(X.head())
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    st.success(f"**Random Forest Accuracy:** {accuracy_score(y_test, y_pred):.2%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=data.target_names, yticklabels=data.target_names, ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Top 10 Important Features")
        importances = pd.Series(model.feature_importances_, index=X.columns)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        importances.nlargest(10).plot(kind='barh', ax=ax2, color='skyblue')
        st.pyplot(fig2)
        
elif section == "Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis (EDA)")
    
    @st.cache_data
    def load_eda_data():
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'age': np.random.normal(35, 12, n).clip(18, 80),
            'income': np.random.lognormal(10.5, 0.8, n),
            'education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], n, p=[0.4, 0.4, 0.15, 0.05]),
            'credit_score': np.random.normal(650, 80, n).clip(300, 850),
            'churned': np.random.binomial(1, 0.2, n)
        })
        df.loc[np.random.choice(n, 50), 'income'] = np.nan
        return df
        
    df = load_eda_data()
    
    st.subheader("1. Basic Structure")
    st.write(f"**Shape:** {df.shape}")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. Missing Values")
        missing = df.isnull().sum()
        st.write(missing[missing > 0])
    with col2:
        st.subheader("3. Numerical Summary")
        st.write(df.describe().round(2))
        
    st.subheader("4. Visualizations")
    
    viz_choice = st.selectbox("Select Visualization", 
                              ["Distributions (Income, Age, Education)", 
                               "Relationships (Age vs Credit Score, Credit by Edu)", 
                               "Missing Values Matrix",
                               "Correlation Heatmap"])
                               
    if viz_choice == "Distributions (Income, Age, Education)":
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(df['income'], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title('Income Distribution')
        axes[0].set_xscale('log')
        sns.boxplot(y=df['age'], ax=axes[1], color='lightgreen')
        axes[1].set_title('Age Boxplot')
        sns.countplot(data=df, x='education', ax=axes[2], palette='viridis', 
                      order=['High School', 'Bachelors', 'Masters', 'PhD'])
        axes[2].set_title('Education Level')
        axes[2].tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
    elif viz_choice == "Relationships (Age vs Credit Score, Credit by Edu)":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.scatterplot(data=df, x='age', y='credit_score', hue='churned', alpha=0.6, ax=axes[0])
        axes[0].set_title('Age vs Credit Score')
        sns.violinplot(data=df, x='education', y='credit_score', ax=axes[1], palette='Set2',
                       order=['High School', 'Bachelors', 'Masters', 'PhD'])
        axes[1].set_title('Credit Score by Education')
        axes[1].tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
    elif viz_choice == "Missing Values Matrix":
        fig, ax = plt.subplots(figsize=(10, 4))
        msno.matrix(df, ax=ax, sparkline=False)
        st.pyplot(fig)
        
    elif viz_choice == "Correlation Heatmap":
        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

    st.subheader("5. Outlier Detection (IQR on Credit Score)")
    Q1 = df['credit_score'].quantile(0.25)
    Q3 = df['credit_score'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['credit_score'] < lower_bound) | (df['credit_score'] > upper_bound)]
    st.write(f"Detected **{len(outliers)}** outliers in Credit Score.")
