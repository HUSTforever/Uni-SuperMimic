#!/usr/bin/env python3
"""Closed-loop equivalence check: eager checkpoint policy vs exported TorchScript.

Runs the same MuJoCo rollout twice (once with the checkpoint network, once with the
TorchScript file) and compares the produced joint trajectories.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

ROLLOUT_DIM = 7 + 29  # free root pose + G1 joint positions


def validate_rollouts(
    eager: np.ndarray,
    scripted: np.ndarray,
    atol: float = 1e-3,
    *,
    expected_frames: int | None = None,
) -> float:
    """Validate two ``N x 36`` qpos rollouts and return their max absolute difference.

    Raises ``ValueError`` for invalid thresholds, non-finite values, mismatched or
    malformed shapes, fewer than two frames, or a difference exceeding ``atol``.
    """
    try:
        atol = float(atol)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"atol must be a finite positive number, got {atol!r}") from exc
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError(f"atol must be a finite positive number, got {atol!r}")

    eager = np.asarray(eager)
    scripted = np.asarray(scripted)
    if eager.ndim != 2 or eager.shape[1] != ROLLOUT_DIM:
        raise ValueError(f"eager rollout must have shape (N, {ROLLOUT_DIM}), got {eager.shape}")
    if scripted.ndim != 2 or scripted.shape[1] != ROLLOUT_DIM:
        raise ValueError(
            f"scripted rollout must have shape (N, {ROLLOUT_DIM}), got {scripted.shape}"
        )
    if eager.shape != scripted.shape:
        raise ValueError(f"rollout shape mismatch: eager {eager.shape}, scripted {scripted.shape}")
    if eager.shape[0] < 2:
        raise ValueError(f"rollout must contain at least 2 frames, got {eager.shape[0]}")
    if expected_frames is not None and len(eager) < expected_frames:
        raise ValueError(f"expected {expected_frames} rollout frames, got only {len(eager)}")
    if not np.all(np.isfinite(eager)):
        raise ValueError("eager rollout contains non-finite values")
    if not np.all(np.isfinite(scripted)):
        raise ValueError("scripted rollout contains non-finite values")

    diff = float(np.max(np.abs(eager - scripted)))
    if diff > atol:
        raise ValueError(f"rollout mismatch: max |qpos diff| {diff:.6g} exceeds atol {atol:.6g}")
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--ts", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument(
        "--full-clip",
        action="store_true",
        help="require successful completion of the entire reference",
    )
    parser.add_argument("--motion-files", default=None)
    args = parser.parse_args()

    from unisupermimic.tasks.motion_tracking.g1.supermimic import render
    from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import make_eval_env

    env = make_eval_env(
        args.ckpt.parent,
        "mujoco",
        motion_files=args.motion_files.split(",") if args.motion_files else None,
        device="cpu",
    )
    checkpoint = torch.load(args.ckpt, map_location=torch.device(env.device), weights_only=True)
    # The exported TorchScript runs in float32, while training-time eval uses bfloat16
    # autocast; disable it here so the two rollouts are numerically comparable.
    fp32_checkpoint = dict(checkpoint)
    fp32_checkpoint["args"] = {**checkpoint["args"], "amp": "none"}
    motion = env.supermimic_mdp.motion
    clip_index = motion.names.index(args.clip)
    clip_length = int(motion.clip_len[clip_index])
    seconds = clip_length / 50 if args.full_clip else args.seconds
    expected_frames = min(max(2, round(seconds * 50)), clip_length - 1)
    eager_info, scripted_info = {}, {}

    try:
        _, eager = render.rollout_qpos(
            env, fp32_checkpoint, args.clip, seconds, diagnostics=eager_info
        )
        original = render.load_policy
        scripted = torch.jit.load(str(args.ts), map_location="cpu")

        class ScriptedPolicy:
            def __call__(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
                return scripted(history.cpu(), goal.cpu()).to(history.device)

        render.load_policy = lambda env, ck: ScriptedPolicy()
        try:
            _, from_ts = render.rollout_qpos(
                env, fp32_checkpoint, args.clip, seconds, diagnostics=scripted_info
            )
        finally:
            render.load_policy = original
    finally:
        env.close()
        cast(Any, env.backend).close()

    if eager_info["terminated"] or scripted_info["terminated"]:
        raise ValueError(
            "closed-loop verification physically terminated before successful completion"
        )
    if args.full_clip and not (
        eager_info["reached_clip_end"] and scripted_info["reached_clip_end"]
    ):
        raise ValueError("closed-loop verification did not reach the end of the clip")
    diff = validate_rollouts(eager, from_ts, expected_frames=expected_frames)
    print(f"rollout frames : {eager.shape[0]}")
    print(f"max |qpos diff|: {diff:.3e}")
    print("closed-loop equivalence: OK")


if __name__ == "__main__":
    main()
