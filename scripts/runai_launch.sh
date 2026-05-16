#!/usr/bin/env bash
# Submit a job to EPFL RCP / Runai for the HateMM mini-project.
# Reference: EE-559 cluster access tutorial (runai/RCP).
#
# Usage:
#   JOB_NAME=hatemm-train ./scripts/runai_launch.sh configs/default.yaml
#
# TODO before first submit:
#   - confirm IMAGE matches the EE-559 group container
#   - confirm PVC name / mount path for the group directory
#   - confirm gpu/cpu/memory quotas for the EE-559 allocation

set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
JOB_NAME="${JOB_NAME:-hatemm-$(date +%Y%m%d-%H%M%S)}"

# --- Edit these to match the EE-559 cluster tutorial -----------------------
IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee-559/pytorch:latest}"
PVC="${PVC:-runai-ee-559-<user>-scratch:/scratch}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/Mini-Project}"
GPU="${GPU:-1}"
CPU="${CPU:-4}"
MEMORY="${MEMORY:-32G}"
# ---------------------------------------------------------------------------

runai submit "${JOB_NAME}" \
  --image "${IMAGE}" \
  --gpu "${GPU}" \
  --cpu "${CPU}" \
  --memory "${MEMORY}" \
  --pvc "${PVC}" \
  --command -- bash -lc "cd '${PROJECT_DIR}' && python -m src.train --config '${CONFIG}'"

echo "Submitted: ${JOB_NAME}"
echo "Follow logs with: runai logs ${JOB_NAME} -f"
