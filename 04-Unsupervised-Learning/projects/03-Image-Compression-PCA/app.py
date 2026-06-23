import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

st.set_page_config(page_title="PCA Image Compression", layout="wide")

st.title("🖼️ PCA Image Compression")
st.markdown("Upload an image and see how Principal Component Analysis compresses it while retaining structural integrity.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    img = Image.open(uploaded_file).convert('L') # Convert to grayscale for simplicity
    img_array = np.array(img)
    
    st.sidebar.header("Compression Settings")
    max_components = min(img_array.shape)
    
    # Slider for number of components
    n_components = st.sidebar.slider(
        "Number of Principal Components", 
        min_value=1, 
        max_value=max_components, 
        value=int(max_components * 0.1),
        step=1
    )
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(img_array)
    reconstructed = pca.inverse_transform(compressed)
    
    # Calculate compression ratio
    original_size = img_array.shape[0] * img_array.shape[1]
    compressed_size = (compressed.shape[0] * compressed.shape[1]) + (pca.components_.shape[0] * pca.components_.shape[1])
    compression_ratio = original_size / compressed_size
    variance_retained = np.sum(pca.explained_variance_ratio_) * 100
    
    st.sidebar.metric("Variance Retained", f"{variance_retained:.2f}%")
    st.sidebar.metric("Compression Ratio", f"{compression_ratio:.2f}x")
    
    # Display Images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True, caption=f"Shape: {img_array.shape}")
        
    with col2:
        st.subheader(f"Compressed Image (K={n_components})")
        # Ensure pixel values are valid
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        st.image(reconstructed, use_container_width=True, caption=f"Reconstructed Shape: {img_array.shape}")

else:
    st.info("Please upload an image to begin.")
