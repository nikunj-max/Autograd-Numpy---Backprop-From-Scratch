# autograd-numpy — Backprop From Scratch

A 2-layer MLP (784 → 128 → 10) trained on MNIST where **every gradient is derived and coded by hand in NumPy**. No PyTorch, no autograd. Includes a from-scratch **Adam** vs **SGD** comparison and a numerical gradient-check test suite.

## Quickstart
```bash
pip install -e ".[dev]"
pytest -q                                   # gradient checks
python -m autograd_numpy.train --optimizer both --epochs 10
```
Outputs land in `results/`: `curves.png`, `metrics.json`, `weights_*.npz`.

Docker:
```bash
docker build -t autograd-numpy .
docker run --rm -v $PWD/results:/app/results -v $PWD/data:/app/data autograd-numpy
```

## Design
| File | Responsibility |
|---|---|
| `layers.py` | `Linear`, `ReLU` — forward caches inputs; backward returns dL/dX and stores dL/dW, dL/db |
| `losses.py` | Fused, numerically stable softmax + cross-entropy; gradient `(p - y)/N` |
| `model.py`  | `MLP` container; backward walks layers in reverse (chain rule) |
| `optim.py`  | `SGD` (+momentum), `Adam` (bias-corrected) |
| `train.py`  | Mini-batch loop, evaluation, plots, CLI |
| `tests/`    | Central finite-difference gradient check on every parameter |

## The math
- Linear: `dW = Xᵀ·dY`, `db = Σ dY`, `dX = dY·Wᵀ`
- ReLU: `dX = dY ⊙ 1[X>0]`
- Softmax+CE: `∂L/∂z = (softmax(z) − onehot(y)) / N`

## Results
_Fill in after running on MNIST (10 epochs, batch 64, seed 0):_

| Optimizer | LR | Final test acc | Epochs to 97% |
|---|---|---|---|
| SGD  | 0.1  | _TBD_ | _TBD_ |
| Adam | 1e-3 | _TBD_ | _TBD_ |

![curves](results/curves.png)

## What I learned / failure modes
_Write 3–5 honest bullets: LR sensitivity of SGD, why Adam's bias correction matters early, a bug you hit and how gradcheck caught it._
