# 📧 Spam Detection System

## Overview
This project uses Supervised Learning Natural Language Processing (NLP) techniques to classify text messages or emails as either 'Spam' or 'Ham' (Not Spam). 

## Architecture
```mermaid
graph TD
    A[Raw Text Data] --> B[Text Preprocessing \n Tokenization, Stopwords]
    B --> C[Vectorization \n TF-IDF / CountVectorizer]
    C --> D[Naive Bayes Classifier]
    C --> E[Support Vector Machine]
    D --> F[Model Evaluation \n Precision, Recall]
    E --> F
    F --> G[Streamlit Dashboard \n Text Classification]
    
    style A fill:#4c566a,color:#eceff4
    style G fill:#a3be8c,color:#2e3440
```

## Project Structure
*   `data/`: Contains the text datasets (e.g., SMS Spam Collection).
*   `notebooks/`: Jupyter notebooks with text EDA and model training.
*   `src/`: Python scripts for text cleaning and vectorization pipelines.
*   `app.py`: Streamlit dashboard for real-time message testing.

## How to Run
1. Install dependencies: `pip install streamlit scikit-learn pandas numpy nltk`
2. Navigate to the project directory.
3. Run the dashboard: `streamlit run app.py`
