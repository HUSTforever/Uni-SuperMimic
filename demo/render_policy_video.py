#!/usr/bin/env python3
"""Render a real policy rollout, restoring its recorded motion and configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import make_eval_env
from unisupermimic.tasks.motion_tracking.g1.supermimic.render import (
    DEFAULT_SCENE,
    render_comparison,
    rollout_qpos,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--backend", choices=["mujoco", "mjwarp", "motrix"], default="mujoco")
    parser.add_argument("--device")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--motion-files", help="optional explicit comma-separated references")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--global-ghost", action="store_true", help="show world-position tracking error"
    )
    parser.add_argument(
        "--hide-reference",
        action="store_true",
        help="render the real robot without the reference ghost",
    )
    args = parser.parse_args()
    env = make_eval_env(
        args.ckpt.parent,
        args.backend,
        motion_files=args.motion_files.split(",") if args.motion_files else None,
        device=args.device,
    )
    try:
        checkpoint = torch.load(args.ckpt, map_location=env.device, weights_only=True)
        reference, policy = rollout_qpos(env, checkpoint, args.clip, args.seconds, start=args.start)
        written = render_comparison(
            reference,
            policy,
            args.output,
            args.scene,
            clip_name=args.clip,
            width=args.width,
            height=args.height,
            fps=args.fps,
            global_ghost=args.global_ghost,
            show_reference=not args.hide_reference,
        )
        print(f"wrote {written} genuine rollout frames -> {args.output}")
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
