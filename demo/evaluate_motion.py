#!/usr/bin/env python3
"""Record genuine physics rollout, completion, tracking and temporal smoothness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import evaluate, make_eval_env
from unisupermimic.tasks.motion_tracking.g1.supermimic.preprocess import motion_metrics
from unisupermimic.tasks.motion_tracking.g1.supermimic.render import (
    DEFAULT_SCENE,
    render_comparison,
    rollout_qpos,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--motion-files", help="explicit comma-separated references; otherwise restore run config"
    )
    parser.add_argument("--backend", choices=["mujoco", "mjwarp", "motrix"], default="mujoco")
    parser.add_argument("--device", default=None)
    parser.add_argument("--clip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    files = args.motion_files.split(",") if args.motion_files else None
    env = make_eval_env(args.ckpt.parent, args.backend, motion_files=files, device=args.device)
    try:
        ckpt = torch.load(args.ckpt, map_location=env.device, weights_only=True)
        result = evaluate(env, ckpt)
        motion = env.supermimic_mdp.motion
        clip = args.clip or motion.names[0]
        clip_index = motion.names.index(clip)
        seconds = float(motion.clip_len[clip_index]) / 50
        ref, policy = rollout_qpos(env, ckpt, clip, seconds)
        stats = result["per_clip"][clip]
        result.update(
            checkpoint=str(args.ckpt.resolve()),
            checkpoint_iteration=int(ckpt["iteration"]),
            checkpoint_sha256=hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
            backend=args.backend,
            device=str(env.device),
            clip=clip,
            control_fps=50,
            completed=bool(
                not stats["failed"] and stats["steps"] == int(motion.clip_len[clip_index]) - 1
            ),
            reference=motion_metrics(ref),
            policy=motion_metrics(policy),
            joint_tracking_rmse=float(np.sqrt(np.mean((policy[:, 7:] - ref[:, 7:]) ** 2))),
        )
        if files is None:
            recorded = json.loads((args.ckpt.parent / "run_config.json").read_text())
            files = recorded["config"]["env"]["commands"]["motion"]["motion_files"]
        result["motions"] = [
            {
                "path": str(Path(f).resolve()),
                "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest(),
            }
            for f in files
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_suffix(".npz"), reference=ref, policy=policy, fps=50)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.video:
            render_comparison(ref, policy, args.video, DEFAULT_SCENE, clip_name=clip, fps=args.fps)
        print(json.dumps(result, indent=2))
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
