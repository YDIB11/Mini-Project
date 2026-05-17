"""Classification metrics + per-target-community breakdown + bootstrap CIs."""

from __future__ import annotations

import ast
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


# Each metric is computed as fn(labels, preds, probs). Probs is only used by AUROC.
_METRIC_FNS = {
    "accuracy":  lambda y, p, pb: accuracy_score(y, p),
    "macro_f1":  lambda y, p, pb: f1_score(y, p, average="macro", zero_division=0),
    "precision": lambda y, p, pb: precision_score(y, p, zero_division=0),
    "recall":    lambda y, p, pb: recall_score(y, p, zero_division=0),
    "auroc":     lambda y, p, pb: roc_auc_score(y, pb) if len(set(y.tolist())) > 1 else float("nan"),
}


def compute_metrics(labels, preds, probs):
    return {name: fn(labels, preds, probs) for name, fn in _METRIC_FNS.items()}


def bootstrap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None,
    metric_name: str,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """95% percentile bootstrap CI for a metric. Returns (ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    fn = _METRIC_FNS[metric_name]
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            v = fn(labels[idx], preds[idx], probs[idx] if probs is not None else None)
            if np.isfinite(v):
                values.append(float(v))
        except (ValueError, ZeroDivisionError):
            continue
    if not values:
        return float("nan"), float("nan")
    alpha = (1 - confidence) / 2
    return float(np.percentile(values, 100 * alpha)), float(np.percentile(values, 100 * (1 - alpha)))


def compute_metrics_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    return {name: bootstrap_ci(labels, preds, probs, name, n_resamples, seed=seed)
            for name in _METRIC_FNS}


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


def per_community_breakdown_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    communities: list[str | None],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Bootstrap CIs for accuracy and macro_f1 per community."""
    groups = defaultdict(list)
    for i, c in enumerate(communities):
        groups[_normalize_community(c)].append(i)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for c, idx in groups.items():
        idx_arr = np.array(idx)
        y_true = labels[idx_arr]
        y_pred = preds[idx_arr]
        out[c] = {
            "accuracy": bootstrap_ci(y_true, y_pred, None, "accuracy", n_resamples, seed=seed),
            "macro_f1": bootstrap_ci(y_true, y_pred, None, "macro_f1", n_resamples, seed=seed),
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
