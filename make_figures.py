#!/usr/bin/env python3
"""Generates every figure and table used in report.tex (run from /home/claude/report)."""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/claude")
from micrograd_plus import (MLP, AdamW, DigitClassifier, Standardizer, augment, confusion,
                            cross_entropy, lr_at, load_digits_split, softmax)

os.makedirs("figs", exist_ok=True)
os.makedirs("tables", exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight",
})
C1, C2, C3, C4 = "#1f5fa8", "#d1495b", "#2a9d8f", "#8d6e63"

train, val, test = load_digits_split(0)
(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = train, val, test


class PerPixelScaler:
    """The first (flawed) normalisation: one mean/std per pixel."""
    def fit(self, X):
        flat = X.reshape(len(X), -1)
        self.mean = flat.mean(0)
        std = flat.std(0)
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def __call__(self, X):
        return (X.reshape(len(X), -1) - self.mean) / self.std


BASE = dict(hidden=[128, 64], lr=3e-3, wd=1e-2, dropout=0.2, smoothing=0.05, batch=64, epochs=120,
            aug=True, sched="cosine", norm="global")


def run(cfg_over, seed):
    cfg = {**BASE, **cfg_over}
    rng = np.random.default_rng(seed)
    scaler = (PerPixelScaler() if cfg["norm"] == "pixel" else Standardizer()).fit(X_tr)
    model = MLP(64, cfg["hidden"], 10, dropout=cfg["dropout"], seed=seed)
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n = len(X_tr)
    spe = int(np.ceil(n / cfg["batch"]))
    total, warm = cfg["epochs"] * spe, spe * 3
    Xv, Xt = scaler(X_va), scaler(X_te)
    hist = dict(loss=[], val=[], test=[], lr=[])
    best, best_state, step = -1, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(n)
        ls = 0.0
        for s in range(spe):
            idx = order[s * cfg["batch"]:(s + 1) * cfg["batch"]]
            xb = X_tr[idx]
            if cfg["aug"]:
                xb = augment(xb, rng)
            xb = scaler(xb)
            opt.lr = lr_at(step, total, cfg["lr"], warm) if cfg["sched"] == "cosine" else cfg["lr"]
            model.zero_grad()
            loss = cross_entropy(model(xb), y_tr[idx], cfg["smoothing"])
            loss.backward()
            opt.step()
            ls += float(loss.data) * len(idx)
            step += 1
        model.eval()
        va = float((model(Xv).data.argmax(1) == y_va).mean())
        te = float((model(Xt).data.argmax(1) == y_te).mean())
        hist["loss"].append(ls / n); hist["val"].append(va); hist["test"].append(te); hist["lr"].append(opt.lr)
        if va >= best:
            best, best_state = va, model.state_dict()
    model.load_state_dict(best_state)
    model.eval()
    probs = softmax(model(Xt).data)
    return dict(hist=hist, val=best, test=float((probs.argmax(1) == y_te).mean()), probs=probs)


t0 = time.time()
results = {}

# ------------------------------------------------------------------ 1. reference run (curves)
ref = run({}, seed=0)
h = ref["hist"]
ep = np.arange(1, len(h["loss"]) + 1)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ax[0].plot(ep, h["loss"], color=C1); ax[0].set_yscale("log")
ax[0].set(title="Training loss (log scale)", xlabel="epoch", ylabel="smoothed cross-entropy")
ax[1].plot(ep, h["val"], color=C2, label="validation"); ax[1].plot(ep, h["test"], color=C3, alpha=0.6, label="test (monitor only)")
ax[1].set(title="Accuracy per epoch", xlabel="epoch", ylim=(0.6, 1.0)); ax[1].legend(frameon=False, loc="lower right")
ax[2].plot(ep, h["lr"], color=C4)
ax[2].set(title="Learning rate (warm-up + cosine)", xlabel="epoch", ylabel="lr")
fig.tight_layout(); fig.savefig("figs/curves.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 2. ablations (3 seeds each)
ABL = [
    ("Full recipe (global norm)", {}),
    ("Per-pixel normalisation", dict(norm="pixel")),
    ("No augmentation", dict(aug=False)),
    ("No dropout", dict(dropout=0.0)),
    ("No label smoothing", dict(smoothing=0.0)),
    ("No weight decay", dict(wd=0.0)),
    ("Constant learning rate", dict(sched="const")),
]
SEEDS = [0, 1, 2]
abl = {}
for name, over in ABL:
    rs = [run(over, s) for s in SEEDS]
    abl[name] = dict(val=[r["val"] for r in rs], test=[r["test"] for r in rs])
    print(f"{name:32s} val {np.mean(abl[name]['val']):.4f}  test {np.mean(abl[name]['test']):.4f} "
          f"+- {np.std(abl[name]['test']):.4f}", flush=True)
results["ablation"] = abl

with open("tables/ablation.tex", "w") as f:
    f.write("\\begin{tabular}{lccc}\n\\toprule\nVariant & Val.\\ acc.\\ (\\%) & Test acc.\\ (\\%) & $\\Delta$ test (pts)\\\\\n\\midrule\n")
    base = np.mean(abl[ABL[0][0]]["test"]) * 100
    for name, _ in ABL:
        v = np.array(abl[name]["val"]) * 100
        t = np.array(abl[name]["test"]) * 100
        d = t.mean() - base
        f.write(f"{name} & {v.mean():.2f} $\\pm$ {v.std():.2f} & {t.mean():.2f} $\\pm$ {t.std():.2f} & "
                f"{'--' if name == ABL[0][0] else f'{d:+.2f}'}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

fig, ax = plt.subplots(figsize=(6.2, 3.3))
names = [n for n, _ in ABL]
means = [np.mean(abl[n]["test"]) * 100 for n in names]
errs = [np.std(abl[n]["test"]) * 100 for n in names]
cols = [C1] + [C2 if m < means[0] - 0.05 else C4 for m in means[1:]]
ax.barh(range(len(names)), means, xerr=errs, color=cols, capsize=3)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlim(94, 100); ax.set_xlabel("test accuracy (%), mean $\\pm$ std over 3 seeds")
ax.grid(axis="y", alpha=0)
fig.tight_layout(); fig.savefig("figs/ablation.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 3. width / depth sweep
ARCH = [[32], [64], [64, 32], [128, 64], [256, 128]]
arch = {}
for hid in ARCH:
    rs = [run(dict(hidden=hid), s) for s in SEEDS]
    nparam = 64 * hid[0] + hid[0]
    for a, b in zip(hid[:-1], hid[1:]):
        nparam += a * b + b
    nparam += hid[-1] * 10 + 10
    arch[str(hid)] = dict(params=nparam, val=[r["val"] for r in rs], test=[r["test"] for r in rs])
    print(f"hidden {hid}: params {nparam} test {np.mean(arch[str(hid)]['test']):.4f}", flush=True)
results["arch"] = arch
with open("tables/arch.tex", "w") as f:
    f.write("\\begin{tabular}{lrcc}\n\\toprule\nHidden layers & Parameters & Val.\\ acc.\\ (\\%) & Test acc.\\ (\\%)\\\\\n\\midrule\n")
    for hid in ARCH:
        a = arch[str(hid)]
        v, t = np.array(a["val"]) * 100, np.array(a["test"]) * 100
        lab = " -- ".join(str(x) for x in hid)
        f.write(f"{lab} & {a['params']:,} & {v.mean():.2f} $\\pm$ {v.std():.2f} & {t.mean():.2f} $\\pm$ {t.std():.2f}\\\\\n".replace(",", "{,}"))
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ 4. ensemble size
K = 8
members = [run({}, s) for s in range(K)]
ens = []
for k in range(1, K + 1):
    p = np.mean([m["probs"] for m in members[:k]], axis=0)
    ens.append(float((p.argmax(1) == y_te).mean()) * 100)
single = [m["test"] * 100 for m in members]
results["ensemble_curve"] = ens
results["single_8"] = single
print("ensemble curve", [round(e, 2) for e in ens], flush=True)
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.plot(range(1, K + 1), ens, "o-", color=C1, label="ensemble of $k$ networks")
ax.scatter(range(1, K + 1), single, color=C4, s=22, label="individual network $k$", zorder=3)
ax.set(xlabel="number of networks $k$", ylabel="test accuracy (%)", ylim=(96, 100))
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("figs/ensemble.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 5. final saved model: confusion + errors
clf = DigitClassifier.load("/home/claude/digits_model.pkl")
proba = clf.predict_proba(X_te)
pred = proba.argmax(1)
cm = confusion(y_te, pred)
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(cm, cmap="Blues")
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black")
ax.set(xticks=range(10), yticks=range(10), xlabel="predicted digit", ylabel="true digit")
ax.grid(False)
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig("figs/confusion.pdf"); plt.close(fig)

wrong = np.where(pred != y_te)[0]
fig, axs = plt.subplots(1, max(1, len(wrong)), figsize=(1.9 * max(1, len(wrong)), 2.3))
axs = np.atleast_1d(axs)
for a, i in zip(axs, wrong):
    a.imshow(X_te[i], cmap="gray_r"); a.axis("off")
    a.set_title(f"true {y_te[i]}, pred {pred[i]}\nconf. {proba[i].max():.2f}", fontsize=9)
fig.tight_layout(); fig.savefig("figs/errors.pdf"); plt.close(fig)
results["errors"] = [dict(true=int(y_te[i]), pred=int(pred[i]), conf=float(proba[i].max())) for i in wrong]

# per-class table
with open("tables/perclass.tex", "w") as f:
    f.write("\\begin{tabular}{crrrr}\n\\toprule\nDigit & Test images & Correct & Precision (\\%) & Recall (\\%)\\\\\n\\midrule\n")
    for c in range(10):
        tp = cm[c, c]; sup = cm[c].sum(); pp = cm[:, c].sum()
        f.write(f"{c} & {sup} & {tp} & {100 * tp / max(pp, 1):.1f} & {100 * tp / max(sup, 1):.1f}\\\\\n")
    f.write(f"\\midrule\nAll & {cm.sum()} & {np.trace(cm)} & \\multicolumn{{2}}{{c}}{{accuracy {100 * np.trace(cm) / cm.sum():.2f}}}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ 6. dataset samples
from sklearn.datasets import load_digits
d = load_digits()
fig, axs = plt.subplots(4, 10, figsize=(8.6, 3.7))
for c in range(10):
    idx = np.where(d.target == c)[0][:4]
    for r, i in enumerate(idx):
        axs[r, c].imshow(d.images[i], cmap="gray_r"); axs[r, c].axis("off")
    axs[0, c].set_title(str(c), fontsize=10)
fig.tight_layout(); fig.savefig("figs/samples.pdf"); plt.close(fig)

# augmentation illustration
rng = np.random.default_rng(3)
src = X_tr[:6]
fig, axs = plt.subplots(3, 6, figsize=(7.2, 3.8))
for c in range(6):
    axs[0, c].imshow(src[c], cmap="gray_r", vmin=0, vmax=1)
    for r in (1, 2):
        axs[r, c].imshow(augment(src[c:c + 1], rng)[0], cmap="gray_r", vmin=0, vmax=1)
for a in axs.ravel():
    a.axis("off")
for r, lab in enumerate(["original", "augmented", "augmented"]):
    axs[r, 0].text(-0.15, 0.5, lab, transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=9)
fig.tight_layout(); fig.savefig("figs/augment.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 7. numbers for the text
fin_split = {}
for nm, (X, y) in dict(train=(X_tr, y_tr), val=(X_va, y_va), test=(X_te, y_te)).items():
    fin_split[nm] = float((clf.predict(X * 16) == y).mean())
results["final_split_acc"] = fin_split
results["final_test_correct"] = int(np.trace(cm))
results["ref_run"] = dict(best_val=ref["val"], test=ref["test"],
                          epoch_loss_first=h["loss"][0], epoch_loss_last=h["loss"][-1])
results["pixel_epoch1_loss"] = None
# first-epoch loss for the per-pixel variant (explains the loss ~ 20 observed earlier)
pp = run(dict(norm="pixel", epochs=2), 0)
results["pixel_epoch1_loss"] = pp["hist"]["loss"][0]
with open("tables/results.json", "w") as f:
    json.dump(results, f, indent=1, default=float)
print("done in %.0fs" % (time.time() - t0))
print(json.dumps({k: results[k] for k in ["final_split_acc", "errors", "ref_run", "pixel_epoch1_loss", "ensemble_curve"]}, indent=1, default=float))
