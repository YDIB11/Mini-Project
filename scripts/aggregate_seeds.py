"""Aggregate multi-seed result JSONs into a mean+/-std summary table.

Reads every results/<modalities>_seed<N>.json and emits, for each modality
configuration, a per-metric mean and standard deviation across seeds.

Output written to results/_multiseed_summary.json and printed to stdout.
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
    parser.add_argument("--out", type=Path, default=Path("results/_multiseed_summary.json"))
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
            "metrics": {
                k: {
                    "mean": round(float(np.mean(vs)), 4),
                    "std":  round(float(np.std(vs, ddof=1) if len(vs) > 1 else 0.0), 4),
                    "n":    len(vs),
                }
                for k, vs in metrics_per_seed.items()
            },
        }

    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"=== Multi-seed summary ({len(by_modalities)} modality configs) ===")
    print()
    for modalities, s in summary.items():
        print(f"--- {modalities}   (n_seeds={s['n_seeds']}, seeds={s['seeds']}) ---")
        for metric, ms in s["metrics"].items():
            print(f"  {metric:12s}  {ms['mean']:.4f} +/- {ms['std']:.4f}   (n={ms['n']})")
        print()
    print(f"Summary written to {args.out}")


if __name__ == "__main__":
    main()
