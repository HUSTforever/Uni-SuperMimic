"""Finite physical PPO experiments with immutable inputs and per-iteration logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.run_sim2sim import write_json

from .physical_ppo import PhysicalPPO, PhysicalPPOConfig


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_physical_training(bundle, scene, checkpoint, output, config, *, device="cpu"):
    output, scene, checkpoint = Path(output), Path(scene).resolve(), Path(checkpoint).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source_hash = _sha(checkpoint)
    learner = PhysicalPPO(bundle, scene, checkpoint, config, device=device)
    started, completed, status = time.monotonic(), 0, "completed"
    try:
        output.mkdir(parents=True, exist_ok=False)
        write_json(
            output / "finetune_config.json",
            dict(
                method="physical_ppo_warm_start",
                config=asdict(config),
                device=str(device),
                source_checkpoint=str(checkpoint),
                source_checkpoint_sha256=source_hash,
                source_bundle=str(bundle.directory.resolve()),
                source_reference_sha256=_sha(bundle.directory / "reference.npz"),
                scene=str(scene),
                scene_sha256=_sha(scene),
                hardware_ready=False,
                world_model_updated=False,
                policy_normalization_frozen=True,
                physics_hz=200,
                policy_hz=50,
                physical_sim_steps_unit="20ms physical control transition",
            ),
        )
        learner.save(output / "ckpt_000000.pt", 0)
        with (output / "metrics.jsonl").open("x") as stream:
            try:
                for iteration in range(1, config.iterations + 1):
                    stats = learner.update(
                        learner.collect(), actor=iteration > config.critic_warmup
                    )
                    completed = iteration
                    row = dict(
                        iteration=iteration,
                        physical_sim_steps=learner.sim_steps,
                        elapsed_seconds=time.monotonic() - started,
                        **stats,
                    )
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    if iteration % config.checkpoint_every == 0:
                        learner.save(output / f"ckpt_{iteration:06d}.pt", iteration)
                    if iteration == 1 or iteration % 10 == 0 or iteration == config.iterations:
                        print(json.dumps(row, allow_nan=False), flush=True)
            except KeyboardInterrupt:
                status = "interrupted"
                learner.save(output / f"interrupt_{completed:06d}.pt", completed)
        if status == "completed":
            learner.save(output / "last.pt", completed)
        unchanged = _sha(checkpoint) == source_hash
        if not unchanged:
            raise RuntimeError("source checkpoint unexpectedly changed")
        result = dict(
            status=status,
            iterations=completed,
            physical_sim_steps=learner.sim_steps,
            total_wall_seconds=time.monotonic() - started,
            source_checkpoint_unchanged=unchanged,
            hardware_ready=False,
            world_model_updated=False,
        )
        write_json(output / "training_result.json", result)
        return result
    finally:
        learner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "scene", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    defaults = PhysicalPPOConfig()
    for field in fields(defaults):
        default = getattr(defaults, field.name)
        kwargs = dict(default=default)
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(default)
        parser.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    args = parser.parse_args()
    config = PhysicalPPOConfig(**{f.name: getattr(args, f.name) for f in fields(defaults)})
    result = run_physical_training(
        DeploymentBundle.load(args.bundle),
        args.scene,
        args.checkpoint,
        args.output,
        config,
        device=args.device,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
