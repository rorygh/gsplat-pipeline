#!/bin/bash
# One-shot: photos -> SfM -> train -> eval.
# Usage: IMAGE_DIR=path/to/photos ./scripts/run_pipeline.sh
set -euo pipefail

IMAGE_DIR="${IMAGE_DIR:?set IMAGE_DIR to a folder of input photos}"
DATA_DIR="${DATA_DIR:-data/scene}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scene}"
MAX_STEPS="${MAX_STEPS:-30000}"

echo "=== SfM: ${IMAGE_DIR} -> ${DATA_DIR} ==="
gsplat-pipeline sfm --image-dir "${IMAGE_DIR}" --output-dir "${DATA_DIR}"
# COLMAP expects an images/ folder alongside sparse/ for the dataset loader;
# `sfm` reads directly from IMAGE_DIR, so point training at it via a symlink.
mkdir -p "${DATA_DIR}"
[ -e "${DATA_DIR}/images" ] || ln -s "$(realpath "${IMAGE_DIR}")" "${DATA_DIR}/images"

echo "=== Train: ${DATA_DIR} -> ${OUTPUT_DIR} (${MAX_STEPS} steps) ==="
gsplat-pipeline train --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" --max-steps "${MAX_STEPS}"

echo "=== Eval: held-out views ==="
gsplat-pipeline eval --data-dir "${DATA_DIR}" \
    --checkpoint "${OUTPUT_DIR}/checkpoints/final.pt" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Metrics:        ${OUTPUT_DIR}/metrics.json"
echo "Held-out views: ${OUTPUT_DIR}/renders/"
echo "To view interactively: gsplat-pipeline view --checkpoint ${OUTPUT_DIR}/checkpoints/final.pt"
