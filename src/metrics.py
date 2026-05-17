"""Classification metrics + per-target-community breakdown."""

from __future__ import annotations

import ast
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def compute_metrics(labels, preds, probs):
    has_both_classes = len(set(labels.tolist())) > 1
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "auroc": roc_auc_score(labels, probs) if has_both_classes else float("nan"),
    }


def per_community_breakdown(labels, preds, communities):
    groups = defaultdict(list)
    for i, c in enumerate(communities):
        groups[_normalize_community(c)].append(i)
    out = {}
    for c, idx in groups.items():
        y_true = labels[idx]
        y_pred = preds[idx]
        out[c] = {
            "n": float(len(idx)),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
    return out


def _normalize_community(raw):
    if raw is None or raw == "":
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                items = sorted({str(x).strip() for x in parsed if str(x).strip()})
                return ",".join(items) if items else "unknown"
        except (ValueError, SyntaxError):
            pass
    if "," in s:
        items = sorted({x.strip() for x in s.split(",") if x.strip()})
        return ",".join(items) if items else "unknown"
    return s
