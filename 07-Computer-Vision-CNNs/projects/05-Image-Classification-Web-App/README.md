# 05 - Image Classification Web App

## 🎯 Objective
Deploy a State-of-the-Art (SOTA) pre-trained Deep Learning model (`MobileNetV2`) using PyTorch and Streamlit to classify everyday images into 1000 different categories.

## 🧠 Concepts Covered
- **Pre-trained Models**: Utilizing `torchvision.models` to leverage weights trained on massive datasets (ImageNet) without needing to train from scratch.
- **Image Preprocessing**: Using the model's specific `transforms()` to ensure the input image matches the exact normalization and scaling the model expects.
- **Softmax & Top-K**: Converting raw logits into probabilities and extracting the top 3 most likely classifications using `torch.topk`.

## 🚀 Getting Started

### Running the Web App
Because this project uses a pre-trained model downloaded dynamically via PyTorch, there is no custom training script required!
Just launch the dashboard:
```bash
streamlit run app.py
```
*Note: The first time you run this, PyTorch will download the MobileNetV2 weights (around 14MB) over your internet connection.*

## 📂 Project Structure
```
05-Image-Classification-Web-App/
│
├── app.py                # Streamlit Universal Classifier Web App
└── README.md
```
