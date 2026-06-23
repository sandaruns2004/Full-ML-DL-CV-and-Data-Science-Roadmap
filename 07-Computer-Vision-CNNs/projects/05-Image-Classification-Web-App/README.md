# Project 05: Image Classification Web App

> [!NOTE]  
> The ultimate portfolio project. A machine learning model is useless if non-technical users cannot interact with it. In this project, you will wrap your CNN in a beautiful web interface.

## The Goal
Build a Streamlit web application where a user can upload an image, and the backend model will classify it and display the results.

## Requirements

1. **The Model**: You can use the Cat vs. Dog model from Project 2, or the Plant Disease model from Project 3. Export the trained model to an `.h5` or `.pt` file.
2. **The App Framework**: Use **Streamlit** (or Gradio/FastAPI if you prefer).
3. **Features**:
   - A title and description explaining what the app does.
   - A file uploader widget (`st.file_uploader`) accepting `.png` and `.jpg` files.
   - Once uploaded, display the image on the screen.
   - A "Predict" button.
   - When clicked, run the inference pipeline (remember to apply the exact same normalizations/resizing used during training!).
   - Display the predicted class and a bar chart of the confidence scores.

## Bonus Challenge
Deploy the app live to the internet using **Streamlit Community Cloud** or **HuggingFace Spaces** so you can share the link on your resume.
