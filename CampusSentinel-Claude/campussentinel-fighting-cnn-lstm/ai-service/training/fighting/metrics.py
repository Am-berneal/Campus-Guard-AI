"""Binary metrics for Fight (positive = 1) vs NonFight (0), without extra dependencies.

All functions take PROBABILITIES (after sigmoid). ROC-AUC uses the probabilities
directly; the other metrics use predictions at the configured threshold.
"""

from __future__ import annotations

import numpy as np

from training.fighting.config import CLASS_NAMES


def roc_auc(labels: np.ndarray, probs: np.ndarray) -> float | None:
    """Rank-based (Mann-Whitney U) ROC-AUC with tied scores averaged. None if only one class."""
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(probs, dtype=np.float64).ravel()
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(labels, probs, threshold: float = 0.5) -> dict:
    y = np.asarray(labels).astype(int).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def div(a, b):
        return a / b if b else 0.0

    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "accuracy": div(tp + tn, len(y)),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
        "roc_auc": roc_auc(y, p),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": threshold,
        "n": int(len(y)),
    }


def confusion_matrix(m: dict) -> list[list[int]]:
    """Rows = true class, columns = predicted class, order [NonFight, Fight]."""
    return [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]


def classification_report(m: dict) -> str:
    """Per-class precision/recall/F1/support, in the familiar scikit-learn layout."""
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    rows = {CLASS_NAMES[0]: (*prf(tn, fn, fp), tn + fp), CLASS_NAMES[1]: (*prf(tp, fp, fn), tp + fn)}
    lines = [f"{'':>12}{'precision':>11}{'recall':>9}{'f1-score':>10}{'support':>9}"]
    for name, (p, r, f, s) in rows.items():
        lines.append(f"{name:>12}{p:>11.4f}{r:>9.4f}{f:>10.4f}{s:>9}")
    total = m["n"]
    macro = [sum(v[i] for v in rows.values()) / 2 for i in range(3)]
    weighted = [sum(v[i] * v[3] for v in rows.values()) / total if total else 0 for i in range(3)]
    lines += ["", f"{'accuracy':>12}{'':>11}{'':>9}{m['accuracy']:>10.4f}{total:>9}",
              f"{'macro avg':>12}{macro[0]:>11.4f}{macro[1]:>9.4f}{macro[2]:>10.4f}{total:>9}",
              f"{'weighted avg':>12}{weighted[0]:>11.4f}{weighted[1]:>9.4f}{weighted[2]:>10.4f}{total:>9}",
              f"(threshold {m['threshold']})"]
    return "\n".join(lines)


def plot_confusion_matrix(m: dict, path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(confusion_matrix(m))
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
