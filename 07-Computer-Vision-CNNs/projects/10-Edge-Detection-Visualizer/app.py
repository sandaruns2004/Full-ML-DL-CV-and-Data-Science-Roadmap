import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Edge Detection Visualizer", page_icon="🔲", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #eceff4; }
    h1, h2, h3 { color: #88c0d0; }
</style>
""", unsafe_allow_html=True)

st.title("🔲 Edge Detection Visualizer")
st.markdown("Upload an image and interactively apply classic Computer Vision edge detection algorithms.")

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_arr = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        st.markdown("### ⚙️ Filter Controls")
        filter_type = st.selectbox("Select Edge Detection Algorithm", ["Canny", "Sobel", "Laplacian"])
        
        if filter_type == "Canny":
            st.info("Canny Edge Detector uses a multi-stage algorithm. Adjust the hysteresis thresholds.")
            threshold1 = st.slider("Min Threshold", 0, 255, 100)
            threshold2 = st.slider("Max Threshold", 0, 255, 200)
        elif filter_type == "Sobel":
            st.info("Sobel Operator calculates the gradient of image intensity. Adjust the kernel size.")
            ksize_sobel = st.selectbox("Kernel Size (Must be odd)", [1, 3, 5, 7], index=1)
            direction = st.radio("Direction", ["Both (Magnitude)", "Horizontal (X)", "Vertical (Y)"])
        elif filter_type == "Laplacian":
            st.info("Laplacian Operator calculates the second derivative of the image.")
            ksize_laplace = st.selectbox("Kernel Size (Must be odd)", [1, 3, 5, 7], index=1)

    with col2:
        st.subheader("Filtered Output")
        
        result_img = gray.copy()
        
        if filter_type == "Canny":
            result_img = cv2.Canny(gray, threshold1, threshold2)
            
        elif filter_type == "Sobel":
            if direction in ["Horizontal (X)", "Both (Magnitude)"]:
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize_sobel)
                sobelx = cv2.convertScaleAbs(sobelx)
            if direction in ["Vertical (Y)", "Both (Magnitude)"]:
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize_sobel)
                sobely = cv2.convertScaleAbs(sobely)
                
            if direction == "Both (Magnitude)":
                result_img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
            elif direction == "Horizontal (X)":
                result_img = sobelx
            else:
                result_img = sobely
                
        elif filter_type == "Laplacian":
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize_laplace)
            result_img = cv2.convertScaleAbs(laplacian)
            
        st.image(result_img, use_column_width=True, clamp=True, channels="GRAY")
