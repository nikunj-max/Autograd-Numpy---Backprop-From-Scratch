import numpy as np
from .layers import Linear, ReLU


class MLP:
    """Linear -> ReLU -> Linear   (the '2-layer' MLP)."""

    def __init__(self, in_dim=784, hidden=128, out_dim=10, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = [Linear(in_dim, hidden, rng), ReLU(), Linear(hidden, out_dim, rng)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):   # chain rule: walk the graph backwards
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        return [pg for layer in self.layers for pg in layer.parameters()]

    def predict(self, x):
        return self.forward(x).argmax(axis=1)

    def save(self, path):
        np.savez(path, *[p for p, _ in self.parameters()])

    def load(self, path):
        data = np.load(path)
        for (p, _), key in zip(self.parameters(), data.files):
            p[...] = data[key]
