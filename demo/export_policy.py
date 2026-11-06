#!/usr/bin/env python3
"""Export the trained SuperMimic policy to TorchScript for deployment.

The exported module takes ``(obs_hist, goal)`` exactly like ``networks.Policy``:

    obs_hist : float32[batch, hist, 99]
    goal     : float32[batch, 730]        (goal_frames * 73)
    returns  : float32[batch, 29]         raw actions (not joint targets in rad)

Deployment converts the raw actions to joint position targets with:

    target_q = default_q + context_offset + action_scale * raw_action

The sidecar ``<output>.json`` records these semantics and the robot constants.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.projected_policy import (
    PROJECTED_KIND,
    ProjectedResidualPolicy,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import GOAL_FRAME_DIM, NUM_J, OBS_DIM
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
    G1_JOINT_NAMES,
    action_scale,
    default_joint_pos_list,
)

CONTROL_DT = 0.02
ACTION_TARGET_FORMULA = "default_q + context_offset + action_scale * raw_action"


def build_policy(checkpoint: dict) -> torch.nn.Module:
    args = checkpoint["args"]
    networks.AMP_DTYPE = None  # export in float32
    hidden = tuple(args.get("test_hidden_dims") or args["policy_hidden"])
    policy = networks.Policy(
        OBS_DIM,
        GOAL_FRAME_DIM * int(args["goal_frames"]),
        hist=int(args["hist"]),
        z_dim=int(args["z_dim"]),
        num_joints=NUM_J,
        hidden=hidden,
        action_smoothing=float(args.get("action_smoothing", 1.0)),
    )
    kind = args.get("policy_type", "standard")
    if kind == PROJECTED_KIND:
        try:
            policy = ProjectedResidualPolicy(
                policy,
                raw_lower=args["residual_raw_lower"],
                raw_upper=args["residual_raw_upper"],
                hidden=args["residual_hidden"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid projected residual policy configuration") from exc
        for key in ("raw_lower_out", "raw_upper_out"):
            expected = getattr(policy, key).detach().cpu()
            actual = checkpoint["pi"].get(key)
            if (
                not isinstance(actual, torch.Tensor)
                or actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or not torch.equal(actual.detach().cpu(), expected)
            ):
                raise ValueError("projection bounds buffers do not match policy configuration")
    elif kind != "standard":
        raise ValueError(f"unsupported policy type {kind!r}")
    policy.load_state_dict(checkpoint["pi"])
    policy.eval()
    return policy


def export_metadata(checkpoint: dict) -> dict:
    """Return the deployment metadata stored beside the TorchScript export."""
    args = checkpoint["args"]
    goal_frames = int(args["goal_frames"])
    return {
        "schema_version": 1,
        "output_semantics": "raw_action",
        "joint_names": list(G1_JOINT_NAMES),
        "default_joint_positions": default_joint_pos_list(),
        "action_scales": [action_scale(name) for name in G1_JOINT_NAMES],
        "control_dt": CONTROL_DT,
        "action_smoothing": float(args.get("action_smoothing", 1.0)),
        "obs_dim": OBS_DIM,
        "hist": int(args["hist"]),
        "goal_frames": goal_frames,
        "goal_dim": GOAL_FRAME_DIM * goal_frames,
        "action_dim": NUM_J,
        "action_target_formula": ACTION_TARGET_FORMULA,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    policy = build_policy(checkpoint)
    hist = int(checkpoint["args"]["hist"])
    goal_dim = GOAL_FRAME_DIM * int(checkpoint["args"]["goal_frames"])

    obs = torch.randn(args.batch, hist, OBS_DIM)
    goal = torch.randn(args.batch, goal_dim)
    with torch.inference_mode():
        expected = policy(obs, goal)
    if not bool(torch.isfinite(expected).all()):
        raise ValueError("eager policy output must be finite")

    scripted = torch.jit.trace(policy, (obs, goal), strict=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_suffix(args.output.suffix + ".tmp")
    scripted.save(str(staged))
    try:
        loaded = torch.jit.load(str(staged))
        with torch.inference_mode():
            actual = loaded(obs, goal)
        if not bool(torch.isfinite(actual).all()):
            raise ValueError("exported policy output must be finite")
        diff = (expected - actual).abs().max().item()
        if not math.isfinite(diff) or diff > 1e-5:
            raise ValueError("TorchScript export drifted from the eager policy or is nonfinite")
        staged.replace(args.output)
    finally:
        staged.unlink(missing_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(export_metadata(checkpoint), indent=2) + "\n", encoding="utf-8"
    )

    print(f"iteration      : {checkpoint.get('iteration')}")
    print(f"obs/goal shape : {tuple(obs.shape)} / {tuple(goal.shape)}")
    print(f"output shape   : {tuple(actual.shape)}")
    print(f"max |diff|     : {diff:.3e}")
    print(f"saved          : {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
