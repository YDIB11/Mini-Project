"""Aggregate multi-seed result JSONs into mean +/- std summaries.

Reads every results/<modalities>_seed<N>.json and emits:
  1. Per modality configuration: overall test-set metrics, mean +/- std across seeds.
  2. Per target community (n >= threshold): per-modality accuracy and macro_f1,
     mean +/- std across seeds.

Outputs:
  - results/_multiseed_summary.json (overall, machine-readable)
  - results/_multiseed_per_community.json (per-community, machine-readable)
  - Pretty summary printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


FILENAME_RE = re.compile(r"^(?P<modalities>.+?)_seed(?P<seed>\d+)\.json$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-overall", type=Path,
                        default=Path("results/_multiseed_summary.json"))
    parser.add_argument("--out-community", type=Path,
                        default=Path("results/_multiseed_per_community.json"))
    parser.add_argument("--min-community-n", type=int, default=15,
                        help="only summarize communities with at least this many test samples")
    args = parser.parse_args()

    by_modalities: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for f in sorted(args.results_dir.glob("*_seed*.json")):
        m = FILENAME_RE.match(f.name)
        if not m:
            continue
        modalities = m.group("modalities")
        seed = int(m.group("seed"))
        data = json.loads(f.read_text(encoding="utf-8"))
        by_modalities[modalities].append((seed, data))

    if not by_modalities:
        print(f"No *_seed*.json files in {args.results_dir}")
        return

    overall_summary = _summarize_overall(by_modalities)
    args.out_overall.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

    community_summary = _summarize_per_community(by_modalities, args.min_community_n)
    args.out_community.write_text(json.dumps(community_summary, indent=2), encoding="utf-8")

    _print_overall(overall_summary)
    _print_per_community(community_summary)

    print(f"Overall      written to {args.out_overall}")
    print(f"Per-community written to {args.out_community}")


def _summarize_overall(by_modalities) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for modalities, runs in by_modalities.items():
        metrics_per_seed: dict[str, list[float]] = defaultdict(list)
        for _, data in runs:
            for k, v in data.get("overall", {}).items():
                point = v["point"] if isinstance(v, dict) else v
                if isinstance(point, (int, float)) and np.isfinite(point):
                    metrics_per_seed[k].append(float(point))
        summary[modalities] = {
            "n_seeds": len(runs),
            "seeds": sorted(s for s, _ in runs),
            "metrics": {k: _mean_std(vs) for k, vs in metrics_per_seed.items()},
        }
    return summary


def _summarize_per_community(by_modalities, min_n: int) -> dict[str, dict]:
    # community -> modality -> metric -> [values across seeds]
    per_comm: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    n_by_community: dict[str, int] = {}
    for modalities, runs in by_modalities.items():
        for _, data in runs:
            for c, m in data.get("per_community", {}).items():
                n_by_community.setdefault(c, int(m.get("n", 0)))
                for metric in ("accuracy", "macro_f1"):
                    if metric in m:
                        per_comm[c][modalities][metric].append(float(m[metric]))

    out: dict[str, dict] = {}
    for c in sorted(per_comm, key=lambda x: -n_by_community.get(x, 0)):
        if n_by_community.get(c, 0) < min_n:
            continue
        out[c] = {
            "n": n_by_community[c],
            "by_modality": {
                modalities: {metric: _mean_std(vs) for metric, vs in metrics.items()}
                for modalities, metrics in per_comm[c].items()
            },
        }
    return out


def _mean_std(vs: list[float]) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(vs)), 4),
        "std":  round(float(np.std(vs, ddof=1) if len(vs) > 1 else 0.0), 4),
        "n":    len(vs),
    }


def _print_overall(summary) -> None:
    print()
    print(f"=== Overall (n_modalities={len(summary)}) ===")
    print()
    for modalities, s in summary.items():
        print(f"--- {modalities}   (n_seeds={s['n_seeds']}, seeds={s['seeds']}) ---")
        for metric, ms in s["metrics"].items():
            print(f"  {metric:12s}  {ms['mean']:.4f} +/- {ms['std']:.4f}   (n={ms['n']})")
        print()


def _print_per_community(community_summary) -> None:
    print()
    print(f"=== Per-target-community (only n >= threshold shown) ===")
    print()
    for c, info in community_summary.items():
        print(f"--- {c}   (n={info['n']}) ---")
        for modalities, metrics in info["by_modality"].items():
            line = f"  {modalities:24s}"
            for metric_name, ms in metrics.items():
                line += f"  {metric_name}: {ms['mean']:.3f} +/- {ms['std']:.3f}"
            print(line)
        print()


if __name__ == "__main__":
    main()
