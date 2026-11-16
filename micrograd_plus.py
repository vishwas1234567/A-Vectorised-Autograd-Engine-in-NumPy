#!/usr/bin/env python3
"""
micrograd_plus.py  --  a vectorized micrograd, trained on a small image dataset.

Single file, only needs numpy (scikit-learn is used just to load the 8x8 digits
dataset, which ships inside the package, so no download is needed).

What changed compared with the scalar `Value` engine
----------------------------------------------------
* `Value` (one number per node)  ->  `Tensor` (a whole numpy array per node),
  with broadcasting-aware gradients and matmul.  Thousands of times faster.
* Operations: + - * / ** matmul relu exp log sum mean reshape dropout and a
  fused, numerically stable softmax cross-entropy (with label smoothing).
* Iterative topological sort (no Python recursion limit).
* He initialisation, AdamW optimiser, cosine learning-rate schedule with
  warm-up, gradient clipping, mini-batches.
* Data augmentation (random +-1 pixel shifts and small noise).
* Stratified train / val / test split, best-validation checkpointing.
* Ensemble of several independently trained networks (averaged probabilities).
* Numerical gradient check to prove the backward pass is correct.
* The trained model is saved to one pickle file (weights + preprocessing).

Usage
-----
    python micrograd_plus.py                  # train (5-model ensemble) and save digits_model.pkl
    python micrograd_plus.py --check          # run the gradient check only
    python micrograd_plus.py --load digits_model.pkl   # evaluate a saved model
    python micrograd_plus.py --ensemble 1 --epochs 60  # quicker run

Use the saved model from your own code:
    from micrograd_plus import DigitClassifier
    clf = DigitClassifier.load("digits_model.pkl")
    labels = clf.predict(images)      # images: (N, 8, 8) or (N, 64), values 0..16
"""

import argparse
import pickle
import time

import numpy as np


# =============================================================================
# 1. Autograd engine
# =============================================================================
def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape` (undo numpy broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """A numpy array plus its gradient and the recipe to back-propagate through it."""

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ---- basic properties ----------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}')"

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    # ---- arithmetic ----------------------------------------------------------
    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only int/float powers are supported"
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += p * self.data ** (p - 1) * out.grad

        out._backward = _backward
        return out

    def matmul(self, other):
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    __matmul__ = matmul

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    def __rtruediv__(self, other):
        return self._wrap(other) * self ** -1

    # ---- non-linearities / elementwise --------------------------------------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "ReLU")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    # ---- reductions / shape --------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def dropout(self, p, rng, training=True):
        if not training or p <= 0.0:
            return self
        mask = (rng.random(self.shape) >= p) / (1.0 - p)
        return self * Tensor(mask)

    # ---- backprop ------------------------------------------------------------
    def backward(self):
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:  # iterative DFS -> no recursion limit
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


def cross_entropy(logits, y, smoothing=0.0):
    """Fused softmax + cross-entropy (stable, with optional label smoothing).

    logits: Tensor (N, C);  y: int array (N,)
    """
    n, c = logits.shape
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    logp = z - np.log(exp.sum(axis=1, keepdims=True))

    target = np.full((n, c), smoothing / c)
    target[np.arange(n), y] += 1.0 - smoothing

    out = Tensor(-(target * logp).sum() / n, (logits,), "xent")

    def _backward():
        logits.grad += (probs - target) / n * out.grad

    out._backward = _backward
    return out


# =============================================================================
# 2. Neural-network building blocks
# =============================================================================
class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    def __init__(self, nin, nout, rng):
        # He initialisation (good for ReLU)
        self.W = Tensor(rng.standard_normal((nin, nout)) * np.sqrt(2.0 / nin))
        self.b = Tensor(np.zeros((1, nout)))

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP(Module):
    """Fully-connected net: Linear -> ReLU -> Dropout ... -> Linear (logits)."""

    def __init__(self, nin, hidden, nout, dropout=0.0, seed=0):
        self.rng = np.random.default_rng(seed)
        sizes = [nin] + list(hidden) + [nout]
        self.layers = [Linear(sizes[i], sizes[i + 1], self.rng) for i in range(len(sizes) - 1)]
        self.dropout = dropout
        self.training = True

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        for layer in self.layers[:-1]:
            x = layer(x).relu().dropout(self.dropout, self.rng, self.training)
        return self.layers[-1](x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def state_dict(self):
        return [p.data.copy() for p in self.parameters()]

    def load_state_dict(self, state):
        for p, s in zip(self.parameters(), state):
            p.data = s.copy()
            p.grad = np.zeros_like(p.data)


# =============================================================================
# 3. Optimiser
# =============================================================================
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, clip=5.0):
        self.params, self.lr = params, lr
        self.b1, self.b2 = betas
        self.eps, self.wd, self.clip = eps, weight_decay, clip
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        # global gradient-norm clipping
        norm = np.sqrt(sum((p.grad ** 2).sum() for p in self.params))
        scale = min(1.0, self.clip / (norm + 1e-12))
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            if p.data.ndim > 1:  # decoupled weight decay on weights only, not biases
                p.data -= self.lr * self.wd * p.data
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def lr_at(step, total, base, warmup):
    """Linear warm-up followed by cosine decay."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + np.cos(np.pi * progress))


# =============================================================================
# 4. Data
# =============================================================================
def load_digits_split(seed=0):
    """8x8 handwritten digits (1797 images), stratified 70/15/15 split."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    X, y = d.images.astype(np.float64) / 16.0, d.target
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def augment(batch, rng, noise=0.03):
    """Random shift of up to 1 pixel in each direction + slight Gaussian noise. batch: (N,8,8)."""
    n = batch.shape[0]
    padded = np.pad(batch, ((0, 0), (1, 1), (1, 1)))
    dy = rng.integers(0, 3, size=n)
    dx = rng.integers(0, 3, size=n)
    out = np.empty_like(batch)
    for i in range(n):
        out[i] = padded[i, dy[i]:dy[i] + 8, dx[i]:dx[i] + 8]
    out += rng.normal(0.0, noise, out.shape)
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# 5. Training + model wrapper
# =============================================================================
class Standardizer:
    """Global (single mean / std) standardisation.

    Per-pixel statistics are avoided on purpose: border pixels of the 8x8 digits are
    almost always 0, so their std is ~0 and shifted/augmented images would explode.
    """

    def fit(self, X):
        self.mean = np.float64(X.mean())
        self.std = np.float64(X.std() + 1e-8)
        return self

    def __call__(self, X):
        return (X.reshape(len(X), -1) - self.mean) / self.std


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_one(train, val, scaler, cfg, seed, verbose=True):
    (X_tr, y_tr), (X_va, y_va) = train, val
    rng = np.random.default_rng(seed)
    model = MLP(64, cfg["hidden"], 10, dropout=cfg["dropout"], seed=seed)
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    n = len(X_tr)
    steps_per_epoch = int(np.ceil(n / cfg["batch"]))
    total = cfg["epochs"] * steps_per_epoch
    warmup = steps_per_epoch * 3

    best_acc, best_state, step = -1.0, None, 0
    Xv = scaler(X_va)
    for epoch in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(n)
        loss_sum = 0.0
        for s in range(steps_per_epoch):
            idx = order[s * cfg["batch"]:(s + 1) * cfg["batch"]]
            xb = scaler(augment(X_tr[idx], rng))
            yb = y_tr[idx]

            opt.lr = lr_at(step, total, cfg["lr"], warmup)
            model.zero_grad()
            loss = cross_entropy(model(xb), yb, cfg["smoothing"])
            loss.backward()
            opt.step()
            loss_sum += float(loss.data) * len(idx)
            step += 1

        model.eval()
        val_acc = float((model(Xv).data.argmax(1) == y_va).mean())
        if val_acc >= best_acc:  # >= prefers later (lower-LR) epochs on ties
            best_acc, best_state = val_acc, model.state_dict()
        if verbose and (epoch % 20 == 0 or epoch == cfg["epochs"] - 1):
            print(f"  seed {seed} | epoch {epoch + 1:3d}/{cfg['epochs']} | "
                  f"loss {loss_sum / n:.4f} | val acc {val_acc:.4f} | lr {opt.lr:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


class DigitClassifier:
    """Ensemble of MLPs + preprocessing. Pickle-friendly (stores plain numpy arrays)."""

    def __init__(self, cfg, scaler, states):
        self.cfg, self.scaler, self.states = cfg, scaler, states
        self.models = []
        for i, st in enumerate(states):
            m = MLP(64, cfg["hidden"], 10, dropout=0.0, seed=i)
            m.load_state_dict(st)
            m.eval()
            self.models.append(m)

    def predict_proba(self, images):
        x = self.scaler(np.asarray(images, dtype=np.float64) / 1.0)
        return np.mean([softmax(m(x).data) for m in self.models], axis=0)

    def predict(self, images):
        """images: (N,8,8) or (N,64) with pixel values in 0..16 (sklearn digits scale)."""
        images = np.asarray(images, dtype=np.float64) / 16.0
        return self.predict_proba(images).argmax(1)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"cfg": self.cfg, "mean": self.scaler.mean, "std": self.scaler.std,
                         "states": self.states}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        scaler = Standardizer()
        scaler.mean, scaler.std = blob["mean"], blob["std"]
        return cls(blob["cfg"], scaler, blob["states"])


def confusion(y_true, y_pred, k=10):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# =============================================================================
# 6. Gradient check
# =============================================================================
def gradient_check(eps=1e-6):
    """Compare autograd gradients with central finite differences."""
    rng = np.random.default_rng(0)
    model = MLP(6, [5, 4], 3, dropout=0.0, seed=1)
    x = rng.standard_normal((7, 6))
    y = rng.integers(0, 3, 7)

    def loss_fn():
        return cross_entropy(model(x), y, smoothing=0.1)

    model.zero_grad()
    loss_fn().backward()
    worst = 0.0
    for p in model.parameters():
        analytic = p.grad.copy()
        numeric = np.zeros_like(p.data)
        it = np.nditer(p.data, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p.data[i]
            p.data[i] = old + eps
            up = float(loss_fn().data)
            p.data[i] = old - eps
            down = float(loss_fn().data)
            p.data[i] = old
            numeric[i] = (up - down) / (2 * eps)
        denom = np.maximum(1e-8, np.abs(analytic) + np.abs(numeric))
        worst = max(worst, float((np.abs(analytic - numeric) / denom).max()))
    print(f"gradient check: max relative error = {worst:.2e}  ->  {'PASS' if worst < 1e-5 else 'FAIL'}")
    return worst < 1e-5


# =============================================================================
# 7. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ensemble", type=int, default=5, help="number of networks to train")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="digits_model.pkl")
    ap.add_argument("--check", action="store_true", help="only run the gradient check")
    ap.add_argument("--load", default=None, help="evaluate a saved model instead of training")
    args, _ = ap.parse_known_args()  # ignores extra args such as Jupyter/Colab's "-f kernel.json"

    if args.check:
        gradient_check()
        return

    train, val, test = load_digits_split(args.seed)
    print(f"data: train {len(train[0])} | val {len(val[0])} | test {len(test[0])} images (8x8, 10 classes)")

    if args.load:
        clf = DigitClassifier.load(args.load)
    else:
        assert gradient_check(), "backward pass is wrong - aborting"
        cfg = dict(hidden=args.hidden, lr=args.lr, wd=args.wd, dropout=args.dropout,
                   smoothing=args.smoothing, batch=args.batch, epochs=args.epochs)
        scaler = Standardizer().fit(train[0])
        states, t0 = [], time.time()
        for k in range(args.ensemble):
            model, val_acc = train_one(train, val, scaler, cfg, seed=args.seed * 100 + k)
            print(f"model {k + 1}/{args.ensemble}: best val acc {val_acc:.4f}")
            states.append(model.state_dict())
        print(f"training time: {time.time() - t0:.1f}s")
        clf = DigitClassifier(cfg, scaler, states)
        clf.save(args.out)
        print(f"saved -> {args.out}")

    # ---- evaluation (test set was never used for any decision) ---------------
    X_te, y_te = test
    X_te16 = X_te * 16.0  # predict() expects the original 0..16 scale
    singles = [float((m(clf.scaler(X_te)).data.argmax(1) == y_te).mean()) for m in clf.models]
    pred = clf.predict(X_te16)
    acc = float((pred == y_te).mean())
    print(f"single-model test accuracy: {', '.join(f'{a:.4f}' for a in singles)}")
    print(f"ENSEMBLE test accuracy:     {acc:.4f}  ({int((pred == y_te).sum())}/{len(y_te)})")
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion(y_te, pred))


if __name__ == "__main__":
    main()
