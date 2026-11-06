#!/usr/bin/env python3
"""Live MuJoCo window playing the trained policy's rollout against the reference ghost.

The qpos sequence comes from stepping the real simulation environment with the policy
(same code path as ``render_policy_video.py``), then it is replayed frame by frame in an
interactive MuJoCo window: drag to orbit, scroll to zoom, Esc to close.

    uv run python demo/live_viewer.py --ckpt runs/demo_kick_001/ckpt_010000.pt \
        --clip kick_rave --motion-files .tmp/custom_motions/kick_rave.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--motion-files", required=True)
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    import mujoco
    import mujoco.viewer
    from hydra import compose, initialize_config_dir
    from unilab.base.config_adapter import create_env
    from unilab.base.process_device import configure_backend_process_device
    from unilab.training import ensure_registries
    from unilab.utils.device import get_default_device

    import unisupermimic
    from unisupermimic.scripts.train_supermimic import build_env_override
    from unisupermimic.sim2sim import resolve_sim2sim_config
    from unisupermimic.tasks.motion_tracking.g1.supermimic.render import (
        DEFAULT_SCENE,
        load_comparison_model,
        rollout_qpos,
    )

    print("[viewer] building eval environment ...", flush=True)
    config_dir = Path(unisupermimic.__file__).resolve().parent / "conf" / "supermimic"
    paths = ",".join(repr(part) for part in args.motion_files.split(","))
    overrides = [
        "training.play_only=true",
        f"task=g1_super_mimic/{args.backend}",
        f"env.commands.motion.motion_files=[{paths}]",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    ensure_registries()
    resolve_sim2sim_config(
        args.ckpt.parent, cfg, algo_name="supermimic", strict=bool(cfg.training.sim2sim_strict)
    )
    device = get_default_device()
    bound = configure_backend_process_device(str(cfg.training.sim_backend), device)
    if bound is not None:
        device = bound
    env = cast(
        Any,
        create_env(
            cfg,
            num_envs=len(cfg.env.commands.motion.motion_files),
            env_cfg_override=build_env_override(cfg, eval_mode=True, learner_device=device),
        ),
    )
    try:
        checkpoint = torch.load(args.ckpt, map_location=torch.device(env.device), weights_only=True)
        print("[viewer] rolling out the policy in the simulator ...", flush=True)
        reference, policy = rollout_qpos(env, checkpoint, args.clip, args.seconds)
    finally:
        env.close()
        cast(Any, env.backend).close()

    model = load_comparison_model(DEFAULT_SCENE)
    model.vis.global_.offwidth = max(args.width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(args.height, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    print(f"[viewer] {len(policy)} frames ({len(policy) / 50:.1f} s); window opening ...", flush=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.4
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -10
        frame = 0
        while viewer.is_running():
            data.qpos[:36] = policy[frame]
            data.qpos[36:] = reference[frame]
            data.qpos[36:38] = policy[frame, :2]  # keep the ghost beside the robot
            mujoco.mj_kinematics(model, data)
            viewer.cam.lookat[:] = [policy[frame, 0], policy[frame, 1], 0.85]
            viewer.sync()
            frame = (frame + 1) % len(policy) if args.loop else min(frame + 1, len(policy) - 1)
            time.sleep(0.02)
    print("[viewer] closed", flush=True)


if __name__ == "__main__":
    main()
