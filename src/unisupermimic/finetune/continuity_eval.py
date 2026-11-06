"""Native full-sequence evaluation with unmodified 200Hz physics recording."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.run_sim2sim import write_json

from .continuity_metrics import summarize_continuity
from .sequence_eval import evaluate_sequence


def evaluate_continuity(
    bundle,
    scene,
    output,
    *,
    delay_substeps=0,
    friction=1.0,
    mass_scale=1.0,
    noise=0.0,
    seed=0,
    video=False,
    limits=None,
):
    samples = []

    def record(time, qpos, joint_velocity, root_velocity):
        samples.append((time, qpos, joint_velocity, root_velocity))

    output = Path(output)
    report = evaluate_sequence(
        bundle,
        scene,
        output,
        delay_substeps=delay_substeps,
        friction=friction,
        mass_scale=mass_scale,
        noise=noise,
        seed=seed,
        video=video,
        step_observer=record,
    )
    arrays = dict(
        sample_time=np.array([s[0] for s in samples]),
        qpos=np.array([s[1] for s in samples]).reshape(-1, 36),
        joint_velocity=np.array([s[2] for s in samples]).reshape(-1, 29),
        root_velocity=np.array([s[3] for s in samples]).reshape(-1, 3),
    )
    path = output / "physics_trajectory.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    try:
        continuity = summarize_continuity(
            bundle.metadata,
            arrays["qpos"],
            arrays["joint_velocity"],
            arrays["root_velocity"],
            arrays["sample_time"],
            limits=limits,
        )
    except ValueError as exc:
        # Expected early physical failures can leave fewer samples than a
        # spectrum needs. Missing/invalid data must fail, never imply success.
        continuity = dict(complete=False, passed=False, diagnostic_error=str(exc))
    dance_error = report["sequence"]["phases"]["dance"]["joint_rmse_rad"]
    report["continuity"] = continuity
    report["continuity_fidelity_passed"] = dance_error is not None and dance_error < 0.13
    report["full_sequence_and_continuity_passed"] = bool(
        report["all_attempted_phases_passed"]
        and continuity["passed"]
        and report["continuity_fidelity_passed"]
    )
    report["physics_trajectory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["goal_complete_claimed"] = False
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--delay", type=int, choices=[0, 1], default=0)
    p.add_argument("--friction", type=float, default=1.0)
    p.add_argument("--mass-scale", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    report = evaluate_continuity(
        DeploymentBundle.load(a.bundle),
        a.scene,
        a.output,
        delay_substeps=a.delay,
        friction=a.friction,
        mass_scale=a.mass_scale,
        noise=a.noise,
        seed=a.seed,
        video=a.video,
    )
    print(
        dict(
            physical_complete=report["all_attempted_phases_passed"],
            continuity=report["continuity"]["passed"],
            full_pass=report["full_sequence_and_continuity_passed"],
        )
    )
    raise SystemExit(0 if report["full_sequence_and_continuity_passed"] else 1)


if __name__ == "__main__":
    main()
