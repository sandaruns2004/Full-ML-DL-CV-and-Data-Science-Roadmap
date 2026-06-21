# 🏛️ LeNet and AlexNet

---

## 📋 Table of Contents
1. [Beginner: The Pioneers of Computer Vision](#1-beginner-the-pioneers-of-computer-vision)
2. [Intermediate: PyTorch Implementation of LeNet-5](#2-intermediate-pytorch-implementation-of-lenet-5)
3. [Advanced: AlexNet Innovations and Parallel GPU Topologies](#3-advanced-alexnet-innovations-and-parallel-gpu-topologies)

---

## 1. Beginner: The Pioneers of Computer Vision

### LeNet-5 (1998)
Designed by Yann LeCun, **LeNet-5** was the first commercially successful convolutional network. It was used by banks and post offices to automatically read handwritten zip codes and numbers on checks. 

### AlexNet (2012)
After LeNet-5, neural networks went out of favor because computers were too slow to train large models. In 2012, **AlexNet** won the ImageNet competition by a massive margin, sparking the modern deep learning boom. AlexNet proved that deep neural networks, combined with graphics cards (GPUs), could outperform all traditional hand-crafted computer vision methods.

---

## 2. Intermediate: PyTorch Implementation of LeNet-5

Let us write Yann LeCun's original LeNet-5 architecture in PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Input: 1 channel (Grayscale), 32x32 image
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Output shape after pools:
        # 32x32 -> conv1 -> 28x28 -> pool1 -> 14x14 -> conv2 -> 10x10 -> pool2 -> 5x5
        # Flattened shape: 16 channels * 5 * 5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LeNet-5 originally used tanh activation
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# Verify model output shape
model = LeNet5()
x_input = torch.randn(2, 1, 32, 32)
out = model(x_input)
print("LeNet-5 Input Shape:", x_input.shape)
print("LeNet-5 Output Shape:", out.shape) # Expected: (2, 10)
```

---

## 3. Advanced: AlexNet Innovations and Parallel GPU Topologies

AlexNet won the ImageNet LSVRC-2012 competition, achieving a top-5 error rate of $15.3\%$, compared to $26.2\%$ achieved by the second-best entry.

### Core Innovations:
1. **ReLU Activation**: AlexNet replaced the traditional Sigmoid/Tanh activations with Rectified Linear Units ($f(x) = \max(0, x)$). This sped up training convergence by a factor of 6 because the gradient was non-vanishing in the positive domain.
2. **Dropout Regularization**: To prevent severe overfitting in its 60 million parameters, AlexNet used Dropout (rate 0.5) in the fully connected layers, forcing the model to learn redundant feature representations.
3. **Overlapping Pooling**: Rather than using non-overlapping pooling (like kernel 2, stride 2), AlexNet used overlapping Max Pooling (kernel 3, stride 2). The authors noted that this slightly reduced overfitting.

### GPU Parallelization
In 2012, GPUs had very limited memory. To train the network, the authors split the model across two 3GB NVIDIA GTX 580 GPUs.
- The convolutional layers in GPU 1 and GPU 2 only communicated at certain layers (like Layer 3 and the Fully Connected layers).
- This parallel GPU architecture is represented as two parallel branches in the original AlexNet diagrams. Today, modern GPUs (with 24GB to 80GB VRAM) can easily fit AlexNet on a single card, but this split-architecture was critical to proving the viability of GPU-accelerated deep learning in 2012.

---

[← Previous: CNN Backpropagation](./08-CNN-Backpropagation.md) | [Back to Index](./README.md) | [Next: VGG-Net →](./10-VGG-Net.md)
