import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Neural Network Playground")
st.sidebar.header("Hyperparameters")
lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.03)
activation = st.sidebar.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid"])
hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)

st.write(f"Training a network with {hidden_layers} hidden layers using {activation} and LR={lr}...")
# Placeholder for visual
fig, ax = plt.subplots()
ax.scatter(np.random.randn(50), np.random.randn(50), c='blue')
ax.scatter(np.random.randn(50)+2, np.random.randn(50)+2, c='red')
st.pyplot(fig)
