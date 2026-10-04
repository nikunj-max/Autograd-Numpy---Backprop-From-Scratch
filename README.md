# 🧠 autograd-numpy: Backprop From Scratch

**A 2-layer neural network that reads handwritten digits, with every gradient derived by hand and coded in raw NumPy. No PyTorch, no TensorFlow, no autograd.**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-only-informational)
![Test accuracy](https://img.shields.io/badge/MNIST%20test%20acc-97.7%25-brightgreen)
![Gradient check](https://img.shields.io/badge/gradient%20check-rel.%20error%20~1e--11-success)

---

## TL;DR

| | |
|---|---|
| **What** | MLP `784 → 128 → 10` (101,770 parameters) trained on MNIST |
| **How** | Forward pass, backward pass, softmax cross-entropy, SGD and Adam, all hand-written in NumPy |
| **Proof it's correct** | Central finite-difference gradient check: relative error ≈ **1e-11** on every parameter |
| **Result** | **97.65%** test accuracy (SGD) and **97.55%** (Adam) after 10 epochs |
| **Run it** | One notebook, one click on Kaggle or Colab. Nothing to install |

---

## Why this project exists

`loss.backward()` is a black box. This repo opens it.

The goal is to be able to answer, with code and numbers instead of hand-waving:

- What does backpropagation actually compute, and why do the shapes work out?
- Why are softmax and cross-entropy fused in practice?
- How do you *know* your gradients are right?
- What does Adam do differently from SGD, and why does it need bias correction?

---

## Quickstart

### Option A: Kaggle (recommended)
1. Create a new Kaggle notebook, then **File → Import Notebook** and upload `autograd_numpy_notebook.ipynb`.
2. Get MNIST in either way:
   - **Add Data**: search "MNIST" (or attach *Digit Recognizer*), or
   - **Settings → Internet → On**: the notebook downloads it automatically.
3. **Run All**. No GPU is needed.

Outputs (plots, weights, `metrics.json`) land in `/kaggle/working`.

### Option B: Google Colab
Upload the notebook, then **Runtime → Run all**.

### Option C: Local
```bash
git clone <your-repo-url> && cd autograd-numpy
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install numpy matplotlib
jupyter notebook autograd_numpy_notebook.ipynb
```
Or run the single-file version:
```bash
python autograd_numpy_kaggle.py
EPOCHS=3 python autograd_numpy_kaggle.py              # quick smoke run
```

> **Data loader:** tries Kaggle-attached MNIST (IDX or CSV), then a public mirror, then falls back to scikit-learn's tiny 8×8 `digits` set (accuracy ~90%) so the notebook never hard-fails. Check the `Source:` line printed at the top of the run.

---

## Results

Run: 10 epochs, batch size 64, hidden width 128, seed 0. SGD and Adam share **identical initial weights and batch order**, so the optimizer is the only variable.

| Optimizer | LR | Final test acc | Best epoch acc | Final train loss | First epoch ≥ 97% | Wall time* |
|---|---|---|---|---|---|---|
| SGD  | 0.1  | **97.65%** | 97.65% (ep 10) | 0.0552 | epoch 5 | 10.3 s |
| Adam | 1e-3 | **97.55%** | 97.99% (ep 9)  | 0.0196 | epoch 3 | 32.0 s |

\*Measured in my Kaggle/CPU run; expect different absolute timings on other hardware.

**What the numbers say**
- **Adam converges faster early.** It reaches 97% two epochs sooner than SGD (epoch 3 vs 5) and has a much lower training loss (0.020 vs 0.055).
- **Final test accuracy is a statistical tie.** The gap is 0.1 points, about 10 of the 10,000 test images. Adam's test accuracy also wobbles between epochs (97.99% at epoch 9, 97.55% at epoch 10), so a single seed cannot rank them.
- **Adam's much lower training loss with similar test accuracy** hints at it fitting the training set harder without a matching generalization gain. Regularization or early stopping would be the next experiment.
- **Adam cost ~3× more wall time here**: it does several extra array operations per parameter per step, and this network is too small for the optimizer's benefits to pay that back.

More plots (`samples.png`, `mistakes.png`, `lr_sweep.png`, `relu.png`) are generated.

---

## How it works

Every training step has four beats:

```
1. Forward   X ──Linear──ReLU──Linear──▶ logits
2. Loss      logits, y ──softmax-CE──▶ loss, ∂L/∂logits
3. Backward  ∂L/∂logits ──(reverse through layers)──▶ ∂L/∂W, ∂L/∂b for every layer
4. Update    optimizer nudges each weight against its gradient
```

### The hand-derived gradients

| Operation | Forward | Backward |
|---|---|---|
| Linear | `Y = XW + b` | `dW = Xᵀ·dY`, `db = Σ dY`, `dX = dY·Wᵀ` |
| ReLU | `Y = max(0, X)` | `dX = dY ⊙ 1[X > 0]` |
| Softmax + CE | `L = −log p_y` | `∂L/∂z = (p − onehot(y)) / N` |

Backpropagation itself is four lines. Each layer caches what it needs in `forward`, then:

```python
for layer in reversed(self.layers):
    dout = layer.backward(dout)
```

### Design decisions worth knowing
- **He initialization** (`√(2/fan_in)`) keeps activation variance stable through ReLU layers.
- **Fused softmax-CE with log-sum-exp.** Subtracting the row max prevents overflow: naive softmax returns `NaN` on logits `[1000, 0, −1000]`, while this implementation stays finite.
- **In-place optimizer updates** (`p -= ...`). Writing `p = p - ...` only rebinds a local name, and the model silently never learns.
- **One seeded RNG** shared by all layers, so a full run is reproducible from a single seed.
- **Adam with bias correction.** `m` and `v` start at zero, so early estimates are biased low; dividing by `1 − βᵗ` fixes that.

---

## Verification: how I know the gradients are right

The backward pass is checked against **central finite differences**, which need no calculus and so cannot share a derivation bug:

```
∂L/∂θ ≈ [ L(θ + h) − L(θ − h) ] / 2h        with h = 1e-5
```

| Parameter | Relative error |
|---|---|
| W1 | 2.54e-11 ✅ |
| b1 | 2.75e-11 ✅ |
| W2 | 2.16e-11 ✅ |
| b2 | 1.81e-11 ✅ |

**The detector is itself tested.** I simulated a classic silent bug (forgetting to divide the loss gradient by the batch size, so gradients come out 8× too large). The model would still train, just worse. The check reports a relative error of **0.78 ❌** and catches it immediately.

---

## What's in the notebook

| Part | Content |
|---|---|
| 1 | Config, Kaggle-aware data loader, sample digits |
| 2 | `Linear` and `ReLU`: derivations, shape checks, ReLU gradient plot |
| 3 | Softmax cross-entropy: worked example (loss 0.417 matches the hand calculation) and a numerical-stability demo |
| 4 | Assemble the MLP and run the gradient check |
| 5 | SGD and Adam from scratch |
| 6 | Training loop and a fair SGD vs Adam comparison |
| 7 | Confusion matrix, misclassified digits, learned features |
| 8 | Break-it experiments: gradient bug and learning-rate sweep |
| 9 | Interview cheat sheet |

### Configuration
Edit the first code cell (or set `EPOCHS` as an environment variable):

| Variable | Default | Meaning |
|---|---|---|
| `EPOCHS` | 10 | Full passes over the training set |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `HIDDEN` | 128 | Hidden-layer width |
| `LR_SGD` / `LR_ADAM` | 0.1 / 1e-3 | Learning rates |
| `SEED` | 0 | Controls init, shuffling, and splits |

---

## Repository layout

```
.
├── autograd_numpy_notebook.ipynb   # theory + code + experiments (start here)
├── autograd_numpy_kaggle.py        # same code as one self-contained script
├── results/                        # plots, metrics.json, saved weights
│   ├── curves.png
│   ├── confusion.png
│   ├── weights.png
│   └── ...
└── README.md
```

---

## Limitations (honest ones)

- **Single seed, single run.** Differences under ~0.3 points between configurations are within noise. A proper comparison would average over several seeds.
- **Test set used for monitoring.** Per-epoch test accuracy is reported for convenience. For rigorous hyperparameter tuning, carve a validation split out of the training data instead.
- **Fully connected only.** An MLP on flattened pixels ignores image structure; a small CNN would beat ~98% easily.
- **Educational speed.** NumPy on CPU, no fused kernels. It is not meant to compete with a framework.

## Ideas to extend

- [ ] SGD with momentum (`SGD(0.1, momentum=0.9)`) and a three-way optimizer comparison
- [ ] L2 regularization / weight decay in `Linear.backward`
- [ ] Dropout with a hand-derived backward pass
- [ ] Learning-rate schedules (step decay, cosine)
- [ ] A convolutional layer with a hand-written backward pass
- [ ] Average results over 5 seeds with error bars

---

## References

- Rumelhart, Hinton & Williams (1986). *Learning representations by back-propagating errors.*
- Kingma & Ba (2014). *Adam: A Method for Stochastic Optimization.* [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- He et al. (2015). *Delving Deep into Rectifiers.* [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
- Stanford CS231n: backpropagation notes.
- LeCun et al. MNIST database of handwritten digits.

## Author

**Nikunj Bhardwaj**:· [LinkedIn]([https://linkedin.com/in/your-handle](https://www.linkedin.com/in/nikunj-bhardwaj-003829326/)) · [Twitter/X]([https://twitter.com/your-handle](https://x.com/knee_kunzz))

Built as part of a from-scratch deep-learning portfolio. Feedback and corrections are welcome. Open an issue or a PR.
