"""Training loop + CLI.  Usage:
    python -m autograd_numpy.train --optimizer both --epochs 10
"""
import argparse
import json
import os
import time
import numpy as np

from .data import load_mnist, load_digits_small
from .losses import softmax_cross_entropy
from .model import MLP
from .optim import SGD, Adam


def accuracy(model, x, y):
    return float((model.predict(x) == y).mean())


def fit(model, opt, data, epochs=10, batch_size=64, seed=0, log=print):
    x_train, y_train, x_test, y_test = data
    rng = np.random.default_rng(seed)
    hist = {"iter_loss": [], "epoch_loss": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(x_train))
        losses = []
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            logits = model.forward(x_train[b])                 # 1. forward
            loss, dlogits = softmax_cross_entropy(logits, y_train[b])
            model.backward(dlogits)                            # 2. backward
            opt.step(model.parameters())                       # 3. update
            losses.append(loss)
        hist["iter_loss"].extend(losses)
        hist["epoch_loss"].append(float(np.mean(losses)))
        hist["test_acc"].append(accuracy(model, x_test, y_test))
        log(f"epoch {epoch:2d} | loss {hist['epoch_loss'][-1]:.4f} | test acc {hist['test_acc'][-1]:.4f}")
    return hist


def plot(histories, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        k = min(50, len(h["iter_loss"]))                        # smooth the noisy per-batch loss
        sm = np.convolve(h["iter_loss"], np.ones(k) / k, mode="valid")
        ax[0].plot(sm, label=name)
        ax[1].plot(range(1, len(h["test_acc"]) + 1), h["test_acc"], marker="o", label=name)
    ax[0].set(title="Training loss (moving avg)", xlabel="iteration", ylabel="cross-entropy", yscale="log")
    ax[1].set(title="Test accuracy", xlabel="epoch", ylabel="accuracy")
    for a in ax:
        a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "digits"], default="mnist")
    ap.add_argument("--optimizer", choices=["sgd", "adam", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-sgd", type=float, default=0.1)
    ap.add_argument("--lr-adam", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    data = load_mnist() if a.dataset == "mnist" else load_digits_small()
    in_dim = data[0].shape[1]
    os.makedirs(a.out, exist_ok=True)
    runs = {"sgd": lambda: SGD(a.lr_sgd), "adam": lambda: Adam(a.lr_adam)}
    names = ["sgd", "adam"] if a.optimizer == "both" else [a.optimizer]

    histories = {}
    for n in names:
        print(f"\n=== {n.upper()} ===")
        model = MLP(in_dim, a.hidden, 10, seed=a.seed)           # same init => fair comparison
        t0 = time.time()
        histories[n] = fit(model, runs[n](), data, a.epochs, a.batch_size, a.seed)
        histories[n]["seconds"] = time.time() - t0
        model.save(os.path.join(a.out, f"weights_{n}.npz"))
    plot(histories, os.path.join(a.out, "curves.png"))
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({k: {"final_acc": v["test_acc"][-1], "epoch_loss": v["epoch_loss"], "seconds": v["seconds"]}
                   for k, v in histories.items()}, f, indent=2)
    print(f"\nSaved weights, curves.png and metrics.json to {a.out}/")


if __name__ == "__main__":
    main()
