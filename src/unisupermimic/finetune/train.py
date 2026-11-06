"""Warm-start simulation RL using the actual deployment model/PD collector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch

from unisupermimic.deploy.bundle import DeploymentBundle, _robot_xml_path
from unisupermimic.deploy.run_sim2sim import write_json
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig

from .collector import DeploymentCollector
from .learner import DeploymentLearner


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_training(
    bundle: DeploymentBundle,
    scene: Path,
    checkpoint_path: Path,
    output: Path,
    *,
    iterations: int = 1000,
    num_envs: int = 64,
    wm_warmup: int = 150,
    batch: int = 512,
    wm_batches: int = 4,
    lr: float = 5e-5,
    device: str = "cuda",
    compile: bool = True,
    horizon: int | None = None,
    checkpoint_every: int = 250,
    seed: int = 0,
    action_rate_weight: float | None = None,
    balance_weight: float = 0.0,
    ground_slip_weight: float = 0.0,
    ankle_rate_weight: float = 0.0,
    wm_ankle_velocity_weight: float = 1.0,
    wm_root_velocity_weight: float = 1.0,
    focus_intervals=None,
) -> dict:
    output, scene, checkpoint_path = (
        Path(output),
        Path(scene).resolve(),
        Path(checkpoint_path).resolve(),
    )
    if output.exists():
        raise FileExistsError(output)
    for name, value in (
        ("iterations", iterations),
        ("num_envs", num_envs),
        ("batch", batch),
        ("wm_batches", wm_batches),
        ("checkpoint_every", checkpoint_every),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= wm_warmup < iterations or not math.isfinite(lr) or lr <= 0:
        raise ValueError("invalid WM warmup or learning rate")
    if action_rate_weight is not None and (
        not math.isfinite(action_rate_weight) or action_rate_weight > 0
    ):
        raise ValueError("action_rate_weight must be finite and nonpositive")
    if not math.isfinite(ground_slip_weight) or ground_slip_weight < 0:
        raise ValueError("ground_slip_weight must be finite and nonnegative")
    if not math.isfinite(ankle_rate_weight) or ankle_rate_weight < 0:
        raise ValueError("ankle_rate_weight must be finite and nonnegative")
    for name, value in (
        ("wm_ankle_velocity_weight", wm_ankle_velocity_weight),
        ("wm_root_velocity_weight", wm_root_velocity_weight),
    ):
        if not math.isfinite(value) or value < 1:
            raise ValueError(f"{name} must be finite and at least 1")
    if ground_slip_weight > 0 and not balance_weight > 0:
        raise ValueError("ground_slip_weight requires balance_weight > 0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but no CUDA device is available")
    checkpoint_digest = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    allowed = {field.name for field in fields(SuperMimicConfig)}
    raw = {key: value for key, value in checkpoint["args"].items() if key in allowed}
    for key in ("wm_hidden", "policy_hidden", "critic_hidden", "test_hidden_dims"):
        if raw.get(key) is not None:
            raw[key] = tuple(raw[key])
    cfg = SuperMimicConfig(**raw)
    if cfg.hist != 10 or cfg.goal_frames != 10 or cfg.action_smoothing != 0.35:
        raise ValueError("checkpoint does not match the deployment fine-tuning contract")
    cfg = replace(
        cfg,
        iterations=iterations,
        wm_only_iters=wm_warmup,
        lr=lr,
        wm_batch=batch,
        pi_batch=batch,
        wm_batches=wm_batches,
        buffer_steps=600,
        warmup_steps=64,
        ckpt_every=checkpoint_every,
        log_every=min(50, iterations),
        seed=seed,
        log_backend="no_print",
        compile=compile,
        amp="none" if device == "cpu" else cfg.amp,
        horizon=cfg.horizon if horizon is None else horizon,
        action_noise=0.05,
    )
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = dict(
        method="simulation_rl_warm_start",
        policy_normalization_frozen=True,
        source_checkpoint=str(checkpoint_path),
        source_checkpoint_sha256=checkpoint_digest,
        source_bundle=str(bundle.directory.resolve()),
        source_reference_sha256=sha256(bundle.directory / "reference.npz"),
        scene=str(scene),
        scene_sha256=sha256(scene),
        robot_xml_sha256=sha256(_robot_xml_path(scene)),
        device=device,
        num_envs=num_envs,
        learner=asdict(cfg),
        collector=dict(
            physics_hz=200,
            policy_hz=50,
            delay_substeps=[0, 1],
            zero_velocity_fraction=0.25,
            friction_range=[0.7, 1.1],
            first_frame_fraction=0.25,
            hard_interval_fraction=0.25,
            hard_interval_seconds=[14, 18],
            uniform_fraction=0.5,
        ),
        hardware_ready=False,
        action_rate_weight=action_rate_weight,
        balance_weight=balance_weight,
        ground_slip_weight=ground_slip_weight,
        ankle_rate_weight=ankle_rate_weight,
        wm_ankle_velocity_weight=wm_ankle_velocity_weight,
        wm_root_velocity_weight=wm_root_velocity_weight,
        wm_loss_profile=dict(
            ankle_velocity_indices=[45, 46, 51, 52],
            root_local_linear_velocity_indices=[6, 7, 8],
            reduction="original mean over 70 dimensions; masked sum / batch",
            last_error="unweighted",
        ),
        ankle_rate_cost=dict(
            command_velocity_scale_rad_s=2.0,
            measured_velocity_scale_rad_s=2.0,
            measured_weight=0.5,
        ),
        focus_intervals=focus_intervals,
        balance_stance_speed_norm="full_3d",
        balance_cost_coefficients=dict(
            gravity_error=4.0,
            angular_velocity_error=0.15,
            stance_slip=1.0,
            ground_slip=float(ground_slip_weight),
        ),
    )
    write_json(output / "finetune_config.json", config)
    if "sequence" in bundle.metadata:
        config["collector"]["sampling"] = "25%firstframe,25%entry/exit,50%uniform"
        config["collector"]["sequence_segments"] = bundle.metadata["sequence"]["segments"]
        write_json(output / "finetune_config.json", config)
    env = DeploymentCollector(
        bundle,
        scene,
        num_envs=num_envs,
        seed=seed,
        balance_weight=balance_weight,
        ground_slip_weight=ground_slip_weight,
        ankle_rate_weight=ankle_rate_weight,
        focus_intervals=focus_intervals,
    )
    if focus_intervals:
        config["collector"]["sampling"] = (
            "25%firstframe,12.5%entry/exit(or default hard),12.5%focus,50%uniform"
        )
        write_json(output / "finetune_config.json", config)
    if action_rate_weight is not None:
        env.supermimic_mdp.cfg = replace(env.supermimic_mdp.cfg, w_action_rate=action_rate_weight)
    started = time.perf_counter()
    try:
        learner = DeploymentLearner(
            env,
            cfg,
            output,
            device=device,
            backend="deployment_mujoco",
            wm_ankle_velocity_weight=wm_ankle_velocity_weight,
            wm_root_velocity_weight=wm_root_velocity_weight,
        )
        learner.initialize_weights(checkpoint_path)
        learner.save(output / "ckpt_000000.pt", include_optimizers=False)
        print(
            json.dumps(
                dict(
                    status="training",
                    output=str(output),
                    iterations=iterations,
                    wm_warmup=wm_warmup,
                    num_envs=num_envs,
                    lr=lr,
                )
            ),
            flush=True,
        )
        result = learner.train()
        result["total_wall_seconds"] = time.perf_counter() - started
        result["source_checkpoint_unchanged"] = sha256(checkpoint_path) == checkpoint_digest
        if not result["source_checkpoint_unchanged"]:
            raise RuntimeError("source checkpoint unexpectedly changed")
        result["hardware_ready"] = False
        if device.startswith("cuda"):
            result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(output / "training_result.json", result)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--wm-warmup", type=int, default=150)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--wm-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-rate-weight", type=float)
    parser.add_argument("--balance-weight", type=float, default=0.0)
    parser.add_argument("--ground-slip-weight", type=float, default=0.0)
    parser.add_argument("--ankle-rate-weight", type=float, default=0.0)
    parser.add_argument("--wm-ankle-velocity-weight", type=float, default=1.0)
    parser.add_argument("--wm-root-velocity-weight", type=float, default=1.0)
    parser.add_argument("--focus-interval", type=int, nargs=2, action="append")
    args = parser.parse_args()
    report = run_training(
        DeploymentBundle.load(args.bundle),
        args.scene,
        args.checkpoint,
        args.output,
        iterations=args.iterations,
        num_envs=args.num_envs,
        wm_warmup=args.wm_warmup,
        batch=args.batch,
        wm_batches=args.wm_batches,
        lr=args.lr,
        device=args.device,
        compile=not args.no_compile,
        horizon=args.horizon,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        action_rate_weight=args.action_rate_weight,
        balance_weight=args.balance_weight,
        ground_slip_weight=args.ground_slip_weight,
        ankle_rate_weight=args.ankle_rate_weight,
        wm_ankle_velocity_weight=args.wm_ankle_velocity_weight,
        wm_root_velocity_weight=args.wm_root_velocity_weight,
        focus_intervals=args.focus_interval,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
