import gzip
import os
import urllib.request
import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def _fetch(name, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(MIRROR + name, path)
    return path


def _read_idx(path, is_image):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    offset = 16 if is_image else 8                  # IDX header sizes
    arr = np.frombuffer(buf, dtype=np.uint8, offset=offset)
    return arr.reshape(-1, 784) if is_image else arr


def load_mnist(cache_dir="data"):
    xs = {}
    for key, fname in FILES.items():
        xs[key] = _read_idx(_fetch(fname, cache_dir), key.startswith("x"))
    x_train = xs["x_train"].astype(np.float64) / 255.0   # scale to [0,1]
    x_test = xs["x_test"].astype(np.float64) / 255.0
    return x_train, xs["y_train"].astype(np.int64), x_test, xs["y_test"].astype(np.int64)


def load_digits_small():
    """Offline smoke-test dataset (8x8 digits, needs scikit-learn)."""
    from sklearn.datasets import load_digits
    d = load_digits()
    x, y = d.data / 16.0, d.target.astype(np.int64)
    split = int(0.8 * len(x))
    return x[:split], y[:split], x[split:], y[split:]
