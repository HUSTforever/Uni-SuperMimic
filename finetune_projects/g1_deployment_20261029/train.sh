#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
new_run_name="${1:-run_001}"
if [[ ! "$new_run_name" =~ ^run_[A-Za-z0-9_-]+$ ]]; then
  echo 'Use a new run name such as run_002.' >&2
  exit 2
fi
exec env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 .venv/bin/python -m unisupermimic.finetune.train \
 --bundle deploy/artifacts/ji_ni_tai_mei/bundle_v3 \
 --scene deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
 --checkpoint dance_projects/BV1ct4y1n7t9/run_002/last.pt \
 --output "finetune_projects/g1_deployment_20261029/$new_run_name" \
 --iterations 1000 --num-envs 64 --wm-warmup 150 --batch 512 --wm-batches 4 --lr 0.00005
