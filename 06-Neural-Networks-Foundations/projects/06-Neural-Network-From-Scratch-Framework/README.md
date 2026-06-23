# 🛠️ Project 06: Mini Deep Learning Framework

## 🎯 Goal
Build your own lightweight, modular Deep Learning framework using purely NumPy.

## 📝 Description
In Lesson 16, you built a neural network from scratch using hardcoded, procedural Python. While educational, that code was extremely rigid. You couldn't easily add a new layer, or change the optimizer without rewriting the entire Backpropagation calculus block.

In this project, you will apply Object-Oriented Programming (OOP) to build a mini-version of PyTorch/Keras. 

## ✅ Requirements
You must implement a modular framework where a user can instantiate a network like this:

```python
model = Sequential()
model.add(Dense(units=128, input_dim=784))
model.add(ReLU())
model.add(Dense(units=64))
model.add(ReLU())
model.add(Dense(units=10))
model.add(Softmax())

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

To achieve this, you need to create:
- A `Layer` base class that defines `forward()` and `backward()` methods.
- A `Dense` class that handles weights, biases, and gradients.
- Activation function classes (`ReLU`, `Sigmoid`).
- Loss classes (`MSE`, `CrossEntropy`).
- Optimizer classes (`SGD`, `Adam`).

## 📁 Files
- `minitorch/` (Your framework module)
  - `layers.py`
  - `activations.py`
  - `losses.py`
  - `optimizers.py`
  - `network.py`
- `test_framework.ipynb`
