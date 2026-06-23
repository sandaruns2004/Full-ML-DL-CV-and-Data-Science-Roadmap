import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('assets/activations', exist_ok=True)
x = np.linspace(-5, 5, 400)

def plot_activation(name, f, df):
    plt.figure(figsize=(8, 4))
    plt.plot(x, f(x), label='f(x)', linewidth=2)
    plt.plot(x, df(x), label="f'(x) (Derivative)", linestyle='--', linewidth=2)
    plt.title(f'{name} Activation Function')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'assets/activations/{name.lower().replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()

plot_activation('Sigmoid', lambda x: 1/(1+np.exp(-x)), lambda x: (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x)))))
plot_activation('Tanh', lambda x: np.tanh(x), lambda x: 1 - np.tanh(x)**2)
plot_activation('ReLU', lambda x: np.maximum(0, x), lambda x: np.where(x > 0, 1, 0))
plot_activation('Leaky ReLU', lambda x: np.where(x > 0, x, 0.01*x), lambda x: np.where(x > 0, 1, 0.01))

def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
def d_gelu(x): return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) + (0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2))
plot_activation('GELU', gelu, d_gelu)

print('Plots generated in assets/activations/')
