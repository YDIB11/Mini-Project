#!/usr/bin/env bash
# Train and evaluate all 7 modality configurations across multiple seeds.
# 7 modality subsets x 3 seeds = 21 training + eval pairs (about 4 hours on a V100).
#
# Usage on Noto with the course kernel:
#   PY=/home/gnoto_venvs/ee559_venv/bin/python3 bash scripts/multiseed.sh
#
# Usage with the local venv:
#   PY=python bash scripts/multiseed.sh

set -euo pipefail

PY=${PY:-python}
CONFIG=${CONFIG:-configs/default.yaml}
SEEDS=${SEEDS:-"42 1337 2025"}
MODALITIES=(
  "video audio text"
  "video audio"
  "video text"
  "audio text"
  "video"
  "audio"
  "text"
)

mkdir -p results checkpoints

for seed in $SEEDS; do
  for modalities in "${MODALITIES[@]}"; do
    name=$(echo "$modalities" | tr ' ' '+')
    run_name="${name}_seed${seed}"
    ckpt="checkpoints/${run_name}.pt"
    out="results/${run_name}.json"

    if [[ -f "$out" ]]; then
      echo "=== skip $run_name (results exist) ==="
      continue
    fi

    echo "=== train $run_name ==="
    $PY -m src.train --config "$CONFIG" --modalities $modalities --seed "$seed"

    echo "=== eval  $run_name ==="
    $PY -m src.eval --config "$CONFIG" --checkpoint "$ckpt" > "$out"
  done
done

echo
echo "All runs finished. Aggregate with:"
echo "  $PY scripts/aggregate_seeds.py"
