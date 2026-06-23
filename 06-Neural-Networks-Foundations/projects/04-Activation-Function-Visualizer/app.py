import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Activation Function Visualizer")
func_name = st.selectbox("Choose Function", ["Sigmoid", "ReLU", "Tanh", "Leaky ReLU", "GELU"])

z = np.linspace(-5, 5, 100)
if func_name == "ReLU":
    y = np.maximum(0, z)
elif func_name == "Sigmoid":
    y = 1 / (1 + np.exp(-z))
else:
    y = np.tanh(z) # Simplified for placeholder

fig, ax = plt.subplots()
ax.plot(z, y, lw=3)
ax.grid(True)
st.pyplot(fig)
