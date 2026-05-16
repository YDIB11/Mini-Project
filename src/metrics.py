"""Classification metrics + per-target-community breakdown."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    has_both_classes = len(set(labels.tolist())) > 1
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "auroc": roc_auc_score(labels, probs) if has_both_classes else float("nan"),
    }


def per_community_breakdown(
    labels: np.ndarray,
    preds: np.ndarray,
    communities: list[str | None],
) -> dict[str, dict[str, float]]:
    """Returns {community: {n, accuracy, macro_f1}} for each target community."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(communities):
        groups[c if c else "unknown"].append(i)
    out: dict[str, dict[str, float]] = {}
    for c, idx in groups.items():
        y_true = labels[idx]
        y_pred = preds[idx]
        out[c] = {
            "n": float(len(idx)),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
    return out
