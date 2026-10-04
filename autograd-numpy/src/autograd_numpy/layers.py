"""Layers with hand-written forward and backward passes.

Convention: every layer caches what it needs in forward(), and backward(dout)
receives dL/d(output) and returns dL/d(input) while storing dL/d(params).
"""
import numpy as np


class Linear:
    """Y = X @ W + b      shapes: X (N,in), W (in,out), b (out,), Y (N,out)"""

    def __init__(self, in_features, out_features, rng):
        # He initialisation: keeps activation variance stable through ReLU layers
        self.W = rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self._x.T @ dout          # dL/dW = X^T dY
        self.db = dout.sum(axis=0)          # dL/db = sum over batch of dY
        return dout @ self.W.T              # dL/dX = dY W^T

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """Y = max(0, X)"""

    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask            # gradient passes only where input > 0

    def parameters(self):
        return []
