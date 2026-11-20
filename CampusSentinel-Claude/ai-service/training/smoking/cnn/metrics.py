"""Binary classification metrics (no scikit-learn dependency).

Positive class = 1 = smoking.
    precision = TP / (TP + FP)   of images flagged "smoking", how many really were
    recall    = TP / (TP + FN)   of real smoking images, how many were flagged
    F1        = harmonic mean of precision and recall
    ROC-AUC   = probability that a random smoking image gets a higher score than a
                random not-smoking image (threshold-independent; 0.5 = chance, 1.0 = perfect)
"""

from __future__ import annotations

import math

import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via the rank-sum (Mann-Whitney U) formula, with tied scores averaged."""
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < scores.size:  # assign average rank to each group of equal scores
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels).astype(int).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    def safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    auc = roc_auc(labels, probs)
    return {
        "accuracy": safe_div(tp + tn, labels.size),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": None if math.isnan(auc) else auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": threshold,
    }
