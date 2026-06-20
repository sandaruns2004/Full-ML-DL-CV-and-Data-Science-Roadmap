# 📱 Edge ML & Model Optimization

> **Prerequisites**: PyTorch, Model Deployment | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [Why Edge Computing?](#1-why-edge-computing)
2. [Model Quantization](#2-model-quantization)
3. [Network Pruning](#3-network-pruning)
4. [Knowledge Distillation](#4-knowledge-distillation)
5. [TensorRT & CoreML](#5-tensorrt--coreml)
6. [Library Implementation (PyTorch Quantization)](#6-library-implementation-pytorch-quantization)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. Why Edge Computing?

Cloud Deployment (FastAPI on AWS) is great, but it has three major flaws:
1. **Latency**: Sending a photo to the cloud and waiting for the prediction takes 200ms. If you are building a self-driving car, 200ms is the difference between stopping safely and a fatal crash.
2. **Connectivity**: If your IoT agriculture sensor loses 4G connection, it stops working entirely.
3. **Privacy**: Users do not want to send their personal FaceID biometrics or voice recordings to a server.

**Edge ML** is the practice of deploying Machine Learning models directly onto the user's "Edge" device (iPhone, Raspberry Pi, Apple Watch, Arduino).

The problem? An iPhone does not have an NVIDIA A100 GPU with 80GB of VRAM. It has a tiny mobile processor. We must compress massive neural networks to fit onto these devices without destroying their accuracy.

---

## 2. Model Quantization

Neural Network weights are typically stored as **32-bit Floating Point numbers (FP32)**.
- `0.123456789` takes up 32 bits (4 bytes) of RAM.
- A model with 1 Billion parameters takes 4 Gigabytes of RAM.

**Quantization** converts these highly precise weights into lower-precision formats, usually **8-bit Integers (INT8)**.
- `0.123456789` is rounded to simply `0.12`.
- The 1 Billion parameter model now takes exactly 1 Gigabyte of RAM (a 4x compression)!

Not only does this save memory, but integer math is significantly faster for CPUs to compute than floating-point math, leading to a massive speedup in inference time.

**Types of Quantization:**
1. **Post-Training Quantization (PTQ)**: Train the model normally in FP32. After training, brutally convert all weights to INT8. (Fast, but might drop accuracy by 2-5%).
2. **Quantization-Aware Training (QAT)**: During the training loop, fake the quantization process. The model learns to adjust its weights to compensate for the rounding errors. (Slower to train, but retains 99.9% of the original accuracy!).

---

## 3. Network Pruning

Do we actually need all 1 Billion parameters?
Research shows that in massive neural networks, up to 90% of the neurons are "dead" or redundant—they contribute almost nothing to the final prediction.

**Pruning** is the process of setting these redundant weights to exactly $0.0$.
- **Unstructured Pruning**: Find the weights closest to zero anywhere in the network and delete them. This makes the matrix sparse, but modern GPUs actually struggle to process sparse matrices efficiently.
- **Structured Pruning**: Delete entire Convolutional Filters or entire Attention Heads. This physically shrinks the architecture of the network, guaranteeing a massive speedup on any hardware.

*Iterative Pruning:* Train $\rightarrow$ Prune 20% $\rightarrow$ Retrain to recover accuracy $\rightarrow$ Prune 20% $\rightarrow$ Retrain...

---

## 4. Knowledge Distillation

If you have a massive, slow, highly accurate model (The **Teacher**), and you want a tiny, fast model (The **Student**), you use Knowledge Distillation.

Instead of training the Student on the hard labels (e.g., `Dog = 1, Cat = 0`), you train the Student to mimic the Teacher's *Softmax probabilities*.
- Hard Label: `[1.0, 0.0, 0.0]`
- Teacher Output: `[0.85, 0.10, 0.05]` (The Teacher suspects the Dog looks slightly like a Cat).

By training the Student to output `[0.85, 0.10, 0.05]`, you transfer the "dark knowledge" of the Teacher. The Student learns the nuanced relationships between classes, allowing a tiny ResNet-18 to achieve the accuracy of a massive ResNet-152!

---

## 5. TensorRT & CoreML

Once the model is optimized, you must export it to a format that the specific Edge hardware understands.

- **NVIDIA TensorRT**: If you are deploying to an NVIDIA Jetson (used in drones and robotics), passing your model through the TensorRT compiler will physically fuse neural network layers together, unlocking maximum hardware acceleration.
- **Apple CoreML**: If you are deploying to an iOS app, you use `coremltools` to convert your PyTorch model into a `.mlmodel` file. This allows the iPhone to automatically route the math to the specialized "Apple Neural Engine" chip, saving battery life.
- **TensorFlow Lite (TFLite)**: The standard for Android and Raspberry Pi deployments.

---

## 6. Library Implementation (PyTorch Quantization)

Let's do Post-Training Dynamic Quantization on a pre-trained BERT model using PyTorch.

```python
import torch
import time
from transformers import BertModel

# 1. Load a heavy, standard FP32 BERT model
print("Loading FP32 Model...")
model_fp32 = BertModel.from_pretrained('bert-base-uncased')
model_fp32.eval()

# 2. Check the size in MB
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    print(f"Size: {size_mb:.2f} MB")
    os.remove("temp.p")

print("Original Model:")
print_size_of_model(model_fp32)

# 3. Apply Dynamic Quantization
# We target the Linear layers and compress their weights from Float32 to qint8
print("\nQuantizing Model to INT8...")
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

print("Quantized Model:")
print_size_of_model(model_int8)

# 4. Measure Inference Speedup
dummy_input = torch.randint(0, 1000, (1, 128)) # Simulate a 128-token sentence

def measure_time(model, input_tensor, iterations=50):
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(input_tensor)
    return (time.time() - start) / iterations

time_fp32 = measure_time(model_fp32, dummy_input)
time_int8 = measure_time(model_int8, dummy_input)

print(f"\nInference Time (FP32): {time_fp32 * 1000:.2f} ms")
print(f"Inference Time (INT8): {time_int8 * 1000:.2f} ms")
print(f"Speedup: {time_fp32 / time_int8:.2f}x")
```

*(You will typically see the file size drop from ~440MB to ~180MB, and inference speed on a CPU double!)*

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Knowledge Distillation from Scratch**: Write a PyTorch script where the Teacher is a pre-trained ResNet-50 and the Student is a tiny CNN you write yourself. Define a custom loss function that is a combination of standard Cross-Entropy (against the true labels) and Kullback-Leibler (KL) Divergence (against the Teacher's softmax outputs with a temperature scaler).
- 🟡 **iOS App Deployment**: Use `coremltools` to convert an image classification model into CoreML. Open Xcode, create a blank iOS app, drag the `.mlmodel` file into the project, and write 20 lines of Swift code to predict the class of a photo taken by the iPhone camera.

### What's Next
| Next | Why |
|------|-----|
| [Experiment Tracking](./06-Experiment-Tracking.md) | Now that our model runs on edge, we need to track our experiments properly. |

---

[← Model Monitoring](./04-Model-Monitoring.md) | [Back to Index](../README.md) | [Next: Experiment Tracking →](./06-Experiment-Tracking.md)
