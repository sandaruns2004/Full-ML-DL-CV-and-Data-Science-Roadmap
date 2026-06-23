import numpy as np
class Dense:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros((out_features, 1))
    def forward(self, X): pass
    def backward(self, dZ): pass
