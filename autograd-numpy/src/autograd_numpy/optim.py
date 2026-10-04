import numpy as np


class SGD:
    """theta <- theta - lr * g   (optionally with momentum)."""

    def __init__(self, lr=0.1, momentum=0.0):
        self.lr, self.momentum = lr, momentum
        self._v = None

    def step(self, params_and_grads):
        if self._v is None:
            self._v = [np.zeros_like(p) for p, _ in params_and_grads]
        for v, (p, g) in zip(self._v, params_and_grads):
            v[...] = self.momentum * v + g
            p -= self.lr * v


class Adam:
    """Kingma & Ba (2014).

    m <- b1*m + (1-b1)*g            (1st moment: direction / momentum)
    v <- b2*v + (1-b2)*g^2          (2nd moment: per-parameter scale)
    m_hat = m/(1-b1^t), v_hat = v/(1-b2^t)      (bias correction: m,v start at 0)
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self._m = self._v = None

    def step(self, params_and_grads):
        if self._m is None:
            self._m = [np.zeros_like(p) for p, _ in params_and_grads]
            self._v = [np.zeros_like(p) for p, _ in params_and_grads]
        self.t += 1
        for m, v, (p, g) in zip(self._m, self._v, params_and_grads):
            m[...] = self.b1 * m + (1 - self.b1) * g
            v[...] = self.b2 * v + (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
