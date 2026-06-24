# 05 - Spam Detection System

## 🎯 Objective
Use a Multi-Layer Perceptron (MLP) to classify raw text (emails/SMS) as Spam or Not Spam.

## 🧠 Concepts Covered
- **Text Vectorization**: Converting words to numbers using TF-IDF or Bag-of-Words.
- **Sparse Data**: Feeding highly dimensional, sparse arrays into a dense Feed-Forward Neural Network.
- **NLP Baselines**: Establishing the foundations of text processing before moving to Recurrent Neural Networks (RNNs).

## 🚀 Getting Started

### 1. Training the Model
We provide a Jupyter Notebook that will walk you through vectorizing text and training the MLP.
1. Navigate to the `notebooks/` directory.
2. Open and run `Spam_Detector.ipynb`. 
3. This will generate a `model.pth` (weights) and `vectorizer.pkl` (vocabulary) in the `src/` folder.

### 2. Testing the Web App
Once trained, you can interact with your model using the provided Streamlit dashboard. Paste an email or SMS message into the text box and the app will classify it instantly!
```bash
streamlit run app.py
```

## 📂 Project Structure
```
05-Spam-Detection-System/
│
├── notebooks/
│   └── Spam_Detector.ipynb       # NLP preprocessing and Training loop
├── src/
│   ├── model.pth                 # Saved weights (generated after training)
│   └── vectorizer.pkl            # Text vocabulary (generated after training)
├── app.py                        # Streamlit text input dashboard
└── README.md
```
