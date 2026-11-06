# G1 Continuous Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver a smoother real G1 policy demo and a reproducible motion-to-policy pipeline.

**Architecture:** Keep reusable preprocessing and quality metrics in the motion-tracking package. Demo entrypoints orchestrate existing upstream environments and training/evaluation/export. Restore evaluation configuration from the recorded run and keep old runs intact.

**Tech Stack:** Existing Python/NumPy/PyTorch/MuJoCo/UniLab/Hydra and imageio environments; no new paid services.

**Spec:** `docs/superpowers/specs/2026-10-28-g1-continuous-pipeline-design.md` (approved).

## Global Constraints

- Preserve original CSV, runs, checkpoints and videos; use new names and output directories.
- Produce genuine physics rollout; never smooth recorded policy trajectories and present them as simulation.
- Reuse external GVHMR/GMR environments by configurable paths; do not move/delete large models.
- CPU/MuJoCo validation first, followed by mjwarp evaluation of the same checkpoint.
- Native Codex handles architecture and final verification; dsh handles localized changes/tests.
- Work in the requested directory, because the demo and installed assets are currently untracked/ignored; keep baseline diagnostics in `.tmp/smooth_audit`.

## Review Focus

- Quaternion sign flips/world-frame yaw: orientation stays normalized and equivalent through conversion.
- Non-30-fps input and slowdown: phase/duration/velocity reflect actual timestamps.
- Foot support and flight: support geometry must not penetrate; no forced landing during genuine flight.
- Custom motion moved or changed: evaluate the explicitly selected new run/reference and record identity.
- Matching but nonfinite/short exported rollouts: numerical agreement alone is not successful rollout quality.

### Task 1: Motion conversion and smoothing

**Files:** create `src/unisupermimic/tasks/motion_tracking/g1/supermimic/preprocess.py`; modify `demo/convert_gmr_to_csv.py`; create `tests/tasks/test_motion_preprocess.py`.

**Interfaces:** `prepare_motion(qpos, src_fps, *, model_path, output_fps=30., smooth_seconds=0., slowdown=1., face_yaw_deg=None, ground=False) -> tuple[np.ndarray, dict]`; returns CSV xyzw layout and metadata. `motion_metrics(qpos_wxyz, fps=50.) -> dict` for finite trajectories.

- [x] Add tests for world-frame yaw, sign-flipped quaternions, invalid arrays/fps, exact slowdown duration, bounded joint positions, support height, and reduced jitter of a smooth motion with added noise.
- [x] Run the new tests red.
- [x] Implement validation, named joint mapping, continuous rotation interpolation/smoothing, temporal filtering/retiming, foot-geometry support correction and metadata.
- [x] Turn converter into portable CLI with explicit optional GMR/model paths, actual input fps, smoothing/slowdown and metadata output.
- [x] Run converter regression tests green and generate `demo/motion/kick_rave_smooth.csv` without touching the original.

### Task 2: Evaluation and rollout evidence

**Files:** modify `evaluate.py`, `render.py`, `demo/render_policy_video.py`; create `demo/evaluate_motion.py`; extend `tests/tasks/test_supermimic_evaluate.py` and `tests/tasks/test_supermimic_render.py`.

**Interfaces:** `make_eval_env(run_dir, backend='mjwarp', *, motion_files=None, device=None)` restores recorded full config; explicit motion override is checked against the selected run. Rollout evidence stores actual states and reports acceleration/jerk and tracking RMSE.

- [x] Add failing tests for recorded custom motion restoration and explicit selected-run overrides.
- [x] Restore compatible recorded configuration before backend/device overrides, preserve strict structural checks.
- [x] Reuse the same environment builder in demo rendering and TorchScript verification; render at 50 fps when requested.
- [x] Add evaluation report containing backend, checkpoint identity, motion identity, physical completion, trajectory metrics and raw rollout NPZ.
- [x] Verify baseline 10000 checkpoint and optimized reference independently.

### Task 3: Export interface and failed validation

**Files:** modify `demo/export_policy.py`, `demo/verify_torchscript.py`; create `tests/demo/test_policy_export.py` and `tests/demo/test_verify_torchscript.py` (dsh localized task).

**Interfaces:** export `.ts.json` with raw-action semantics, joint order, defaults, scales, hist/goal dimensions and 0.02 control timestep. `validate_rollouts(eager, scripted, atol=1e-3) -> float` rejects incompatible/nonfinite/mismatched arrays.

- [x] Add and execute failing tests for action metadata and disagreement/nonfinite failure.
- [x] Correct the output documentation; save machine-readable interface metadata.
- [x] Make mismatch a nonzero command exit, keep matching float32 rollouts working.
- [x] Personally review diff and rerun tests/real closed-loop export check.

### Task 4: Unified staged pipeline and safe new runs

**Files:** create `demo/pipeline.py`, `tests/demo/test_pipeline.py`; modify training entrypoint only as required for explicit warm start into a fresh run.

**Interfaces:** pipeline accepts local video or NPZ, optional `--mocap-root`, new output directory, `--checkpoint`, and explicit training budget. Stage manifest records commands, hashes, config, status and outputs. Fresh fine-tune initializes from checkpoint network weights without reusing the original run directory/config.

- [x] Test stage ordering, input preflight, failure propagation and no overwrite of baseline run.
- [x] Implement portable paths and upstream argument propagation; allow cached NPZ start.
- [x] Add explicit fresh-run initialization if needed; test snapshot and selected motion remain consistent.
- [x] Run all CPU tests and a real NPZ→evaluation/export/render pipeline.

### Task 5: Improve the real policy and deliver comparison

**Files:** new `demo/motion/kick_rave_smooth*`, `runs/demo_kick_smooth_*`, `demo/artifacts/smooth_*`; update `demo/README.md`.

- [x] Measure several conservative preprocessing settings, choose one preserving the principal kick motion and contact.
- [x] Evaluate existing policy on corrected reference; if inadequate, train/fine-tune in a fresh run and evaluate numbered checkpoints to select actual quality.
- [x] Compare at equal physical sampling rate and normalized phase; target >=30% lower policy acceleration and jerk with complete no-failure rollout and retained principal motion.
- [x] Re-export selected policy, verify eager/TorchScript, and evaluate MuJoCo/mjwarp.
- [x] Render 50 fps policy/reference and old/new comparison with honest timing labels; inspect rendered frames.
- [x] Update documentation with actual metrics/commands, all current paths, limitations and external dependencies.
- [x] Final verification: run tests and targeted lint, inspect git diff/status and report measured results plus any unmet target.
