"""Reproduce the retained checkpoint before adapting it to a slower reference."""

import subprocess
import sys
from pathlib import Path

import torch

from unisupermimic.deploy.run_sim2sim import write_json

PROJECT = Path("finetune_projects/g1_continuity_20261029")
SCENE = "deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"


def train(bundle, checkpoint, output, iterations, warmup, lr, save, focus):
    command = [
        sys.executable,
        "-m",
        "unisupermimic.finetune.train",
        "--bundle",
        str(bundle),
        "--scene",
        SCENE,
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(PROJECT / output),
        "--iterations",
        str(iterations),
        "--num-envs",
        "64",
        "--wm-warmup",
        str(warmup),
        "--lr",
        str(lr),
        "--checkpoint-every",
        str(save),
        "--action-rate-weight",
        "-0.3",
        "--balance-weight",
        "1",
        "--ground-slip-weight",
        "0.35",
        "--ankle-rate-weight",
        "0.05",
    ]
    for pair in focus:
        command.extend(("--focus-interval", *(str(x) for x in pair)))
    with (PROJECT / f"{output}.log").open("x") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def main():
    train(
        PROJECT / "data/damping_kd200",
        Path("finetune_projects/g1_continuous_20261029/run_009_reproduce/last.pt"),
        "run_003_reproduce",
        250,
        100,
        1e-5,
        250,
        ((100, 250), (300, 700), (1000, 1400)),
    )
    selected = torch.load(PROJECT / "run_002/ckpt_000250.pt", weights_only=True, map_location="cpu")
    reproduced = torch.load(
        PROJECT / "run_003_reproduce/last.pt", weights_only=True, map_location="cpu"
    )
    matches = {
        name: all(
            torch.equal(value, reproduced[name][key]) for key, value in selected[name].items()
        )
        for name in ("pi", "wm")
    }
    complete = all(key in reproduced for key in ("pi", "wm", "critic", "opt_pi", "opt_wm", "opt_v"))
    # Optimizer key spelling is validated against the real saved checkpoint,
    # rather than treating a policy-only checkpoint as a full training state.
    write_json(
        PROJECT / "retime_warm_start_verification.json",
        dict(
            selected="run_002/ckpt_000250.pt",
            reproduced="run_003_reproduce/last.pt",
            bit_identical=matches,
            checkpoint_keys=sorted(reproduced),
            complete_checkpoint=complete,
            hardware_ready=False,
        ),
    )
    if not all(matches.values()) or not complete:
        raise RuntimeError("retained policy/world model reproduction did not match")
    train(
        PROJECT / "data/retime_run002_250_s090",
        PROJECT / "run_003_reproduce/last.pt",
        "run_004_slow090",
        800,
        50,
        5e-6,
        100,
        ((100, 267), (323, 767), (1101, 1543)),
    )


if __name__ == "__main__":
    main()
