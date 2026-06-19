# 🧠 Quiz: Deep Learning Architectures

> Test your knowledge of Neural Networks, CNNs, RNNs, and Transformers! Click the `► Show Answer` toggle to check your work.

---

### Question 1: Vanishing Gradients
Which activation function was specifically adopted by the deep learning community to solve the Vanishing Gradient problem in deep multi-layer perceptrons and CNNs?

A) Sigmoid  
B) Tanh  
C) Softmax  
D) ReLU (Rectified Linear Unit)  

<details>
<summary><b>► Show Answer</b></summary>
<br>
<b>Correct Answer: D</b><br><br>
The derivative of Sigmoid and Tanh is always less than 1. When chained together in backpropagation across dozens of layers, these small fractions multiply together and cause the gradient to shrink exponentially to zero. ReLU solves this because its derivative for any positive input is exactly 1.0, allowing gradients to flow perfectly backwards without shrinking.
</details>

---

### Question 2: Convolutional Neural Networks (CNNs)
You have an image of size $32 \times 32$. You apply a $3 \times 3$ Convolutional filter with `stride=1` and `padding=0`. What are the spatial dimensions of the resulting feature map?

A) $32 \times 32$  
B) $30 \times 30$  
C) $34 \times 34$  
D) $16 \times 16$  

<details>
<summary><b>► Show Answer</b></summary>
<br>
<b>Correct Answer: B</b><br><br>
The formula for convolution output dimension is: $O = \frac{W - K + 2P}{S} + 1$. <br>
Here: $W=32, K=3, P=0, S=1$. <br>
$\frac{32 - 3 + 0}{1} + 1 = 30$. Without padding, a 3x3 filter shaves off 1 pixel from every edge (2 pixels total from the width and height).
</details>

---

### Question 3: Recurrent Neural Networks (RNNs)
Why are Vanilla RNNs almost never used for long text sequences like paragraphs or books?

A) They process the whole text simultaneously, requiring too much RAM.  
B) They suffer from severe short-term memory loss due to vanishing gradients across time.  
C) They cannot share weights across different time steps.  
D) They can only output a single classification label, not a sequence of words.  

<details>
<summary><b>► Show Answer</b></summary>
<br>
<b>Correct Answer: B</b><br><br>
Vanilla RNNs multiply the hidden state by the exact same weight matrix at every time step. For a 100-word paragraph, this means multiplying by the same matrix 100 times. If the matrix values are small, the gradients vanish, and the network cannot update its weights based on the first words in the paragraph. LSTMs and Transformers were invented to solve this exact problem.
</details>

---

### Question 4: Transformers and Self-Attention
In the scaled dot-product attention mechanism of a Transformer, three vectors are generated for every token: Query ($Q$), Key ($K$), and Value ($V$). What mathematical operation is used between the $Q$ and $K$ vectors to determine how much "attention" word A should pay to word B?

A) Element-wise addition  
B) Cross product  
C) Dot product  
D) Euclidean distance  

<details>
<summary><b>► Show Answer</b></summary>
<br>
<b>Correct Answer: C</b><br><br>
Attention is calculated as $softmax(\frac{Q \cdot K^T}{\sqrt{d_k}}) \cdot V$. The dot product measures the similarity or "alignment" between the Query vector of the current word and the Key vector of another word. The higher the dot product, the more attention is paid.
</details>

---

### Question 5: Generative Adversarial Networks (GANs)
In a GAN, what is the specific goal of the **Generator** network during training?

A) To predict whether an input image is Real or Fake.  
B) To compress an image into a low-dimensional latent space and reconstruct it.  
C) To maximize the probability that the Discriminator classifies its fake images as "Real".  
D) To gradually remove Gaussian noise from a completely random image.  

<details>
<summary><b>► Show Answer</b></summary>
<br>
<b>Correct Answer: C</b><br><br>
The Generator takes random noise and generates fake images. Its entire loss function is driven by trying to fool the Discriminator. It wants the Discriminator to output $1.0$ (Real) when fed a fake image. (Option B describes an Autoencoder, and Option D describes a Diffusion model).
</details>

---

[← Machine Learning Quiz](./01-Machine-Learning-Quiz.md) | [Back to Index](../README.md)
