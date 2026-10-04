import numpy as np


def softmax_cross_entropy(logits, y):
    """Fused softmax + cross-entropy.

    logits: (N, C) raw scores, y: (N,) integer class labels.
    Returns (mean loss, dL/dlogits).

    Fusing gives the clean gradient (p - onehot)/N and lets us use the
    log-sum-exp trick, so we never compute log(softmax(x)) naively (which
    can underflow to log(0) = -inf).
    """
    n = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)           # stability shift
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(n), y].mean()

    grad = np.exp(log_probs)                                  # softmax probs p
    grad[np.arange(n), y] -= 1.0                              # p - onehot(y)
    grad /= n                                                 # because loss is a mean
    return loss, grad
