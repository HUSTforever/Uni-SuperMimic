#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
new_run_name="${1:-run_002}"
if [[ ! "$new_run_name" =~ ^run_[A-Za-z0-9_-]+$ ]]; then
  echo 'Use a new run name such as run_003.' >&2
  exit 2
fi
new_run_dir="dance_projects/BV1ct4y1n7t9/$new_run_name"
if [[ -e "$new_run_dir" ]]; then
  echo "Refusing to overwrite existing run: $new_run_dir" >&2
  exit 2
fi
exec env -u ALL_PROXY -u all_proxy CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 \
 .venv/bin/unisupermimic-train \
 algo.num_envs=1024 algo.iterations=1000 algo.buffer_steps=600 \
 algo.wm_only_iters=50 algo.wm_batches=4 algo.wm_batch=1024 algo.pi_batch=1024 \
 algo.lr=0.0001 algo.ckpt_every=250 algo.log_every=50 algo.action_smoothing=0.35 \
 training.init_checkpoint=dance_projects/BV1ct4y1n7t9/run_001/ckpt_004000.pt \
 env.commands.motion.motion_files='[dance_projects/BV1ct4y1n7t9/motion_static/ji_ni_tai_mei_static.csv]' \
 "training.log_dir=$new_run_dir"
