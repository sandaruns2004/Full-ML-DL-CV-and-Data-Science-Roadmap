# 03 - House Price Estimator

## 🎯 Objective
Use a Multi-Layer Perceptron (MLP) as a regression model to estimate the continuous value (price) of a house based on numerical features like square footage and age.

## 🧠 Concepts Covered
- **Regression with Neural Networks**: Changing the output node to a single linear value (no sigmoid/softmax).
- **Mean Squared Error (MSE)**: Using `MSELoss` for continuous predictions.
- **Interactive UI**: Utilizing Streamlit sliders to adjust inputs dynamically.

## 🚀 Getting Started

### 1. Training the Model
We provide a python script to generate synthetic house data and train the regression network.
```bash
python src/train.py
```
This outputs a `model.pth` weight file.

### 2. Running the Web App
Launch the Streamlit dashboard to interactively test the model. You can adjust the house's square footage, bedrooms, and age using sliders to see how the Neural Network adjusts the estimated price!
```bash
streamlit run app.py
```

## 📂 Project Structure
```
03-House-Price-Estimator/
│
├── src/
│   ├── train.py                  # PyTorch training script
│   └── model.pth                 # Saved weights (generated after training)
├── app.py                        # Interactive Streamlit Web App
└── README.md
```
