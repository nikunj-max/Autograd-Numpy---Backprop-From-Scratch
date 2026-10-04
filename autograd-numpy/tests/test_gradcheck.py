"""Numerical gradient checking: compare hand-derived gradients to
central finite differences  (f(x+h) - f(x-h)) / 2h."""
import numpy as np
from autograd_numpy import MLP, softmax_cross_entropy, SGD, Adam


def _loss(model, x, y):
    return softmax_cross_entropy(model.forward(x), y)[0]


def test_all_parameter_gradients_match_numerical():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 12))
    y = rng.integers(0, 4, size=8)
    model = MLP(12, 7, 4, seed=3)

    loss, dlogits = softmax_cross_entropy(model.forward(x), y)
    model.backward(dlogits)
    analytic = [g.copy() for _, g in model.parameters()]

    h = 1e-5
    for (p, _), g in zip(model.parameters(), analytic):
        num = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + h; lp = _loss(model, x, y)
            p[i] = old - h; lm = _loss(model, x, y)
            p[i] = old
            num[i] = (lp - lm) / (2 * h)
        rel = np.abs(num - g).max() / (np.abs(num).max() + np.abs(g).max() + 1e-12)
        assert rel < 1e-6, f"relative error {rel}"


def test_softmax_ce_stable_for_huge_logits():
    logits = np.array([[1000.0, 0.0, -1000.0]])
    loss, grad = softmax_cross_entropy(logits, np.array([0]))
    assert np.isfinite(loss) and np.isfinite(grad).all()


def test_softmax_ce_gradient_rows_sum_to_zero():
    rng = np.random.default_rng(0)
    _, grad = softmax_cross_entropy(rng.standard_normal((5, 6)), rng.integers(0, 6, 5))
    assert np.allclose(grad.sum(axis=1), 0)


def test_optimizers_reduce_loss_on_tiny_problem():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 10)); y = (x[:, 0] > 0).astype(int)
    for opt in (SGD(0.1), Adam(1e-2)):
        model = MLP(10, 16, 2, seed=0)
        start = _loss(model, x, y)
        for _ in range(100):
            _, d = softmax_cross_entropy(model.forward(x), y)
            model.backward(d); opt.step(model.parameters())
        assert _loss(model, x, y) < 0.5 * start
