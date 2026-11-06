"""Matched deterministic deployment-physics scenarios for candidate selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.controller import PolicyController
from unisupermimic.deploy.mujoco_sim import NativeSimulator
from unisupermimic.deploy.run_sim2sim import save_result, write_json


def candidate_rank(reports):
    return (
        sum(int(r["tracking"]["completed"]) for r in reports),
        min(r["tracking"]["steps"] / r["tracking"]["expected_steps"] for r in reports),
        -float(np.mean([r.get("joint_tracking_rmse_rad", 1e6) for r in reports])),
    )


def evaluate_case(bundle, scene, output, *, delay_substeps=0, zero_velocity=False, video=False):
    if delay_substeps not in (0, 1):
        raise ValueError("delay_substeps must be0or1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sim = NativeSimulator(bundle, scene)
    sim.reset("reference_zero_velocity" if zero_velocity else "reference")
    controller = PolicyController(bundle)
    expected = len(bundle.reference["q"]) - 1
    report = dict(
        transport="native_mujoco",
        initialization="reference_zero_velocity" if zero_velocity else "reference",
        standing_entry_verified=False,
        entry=dict(attempted=False, completed=False),
        exit=dict(attempted=False, completed=False),
        tracking=dict(expected_steps=expected, steps=0, completed=False, failure=None),
        physics_hz=200,
        policy_hz=50,
        delay_ms=delay_substeps * 5,
    )
    data = dict(qpos=[], reference_frame=[], target=[], raw_action=[], sample_time=[])
    previous = sim.sensor().q.copy()
    for frame in range(expected):
        try:
            sensor = sim.sensor()
            target = controller.step(sensor, frame)
            data["qpos"].append(sim.qpos())
            data["reference_frame"].append(frame)
            data["target"].append(target)
            data["raw_action"].append(controller.previous_action.copy())
            data["sample_time"].append(sensor.sample_time)
            if frame > 0 and delay_substeps:
                sim.advance(previous, duration=0.005)
                sim.advance(target, duration=0.015)
            else:
                sim.advance(target)
            previous = target.copy()
            report["tracking"]["steps"] += 1
            failure = sim.failure(frame + 1)
            if failure:
                raise ValueError(failure)
        except (ValueError, RuntimeError) as exc:
            report["tracking"]["failure"] = str(exc)
            break
    report["tracking"]["completed"] = (
        report["tracking"]["steps"] == expected and report["tracking"]["failure"] is None
    )
    if not data["qpos"]:
        data["qpos"] = np.empty((0, 36))
    report["clipped_target_values"] = controller.clipped_targets
    report["effort_saturation_fraction"] = sim.saturated / max(sim.torque_samples, 1)
    report["max_torque_limit_ratio"] = sim.max_torque_ratio
    return save_result(output, bundle, Path(scene), data, report, video=video)


def evaluate_matrix(bundle, scene, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scenarios = dict(
        reference=dict(delay_substeps=0, zero_velocity=False),
        delayed_5ms=dict(delay_substeps=1, zero_velocity=False),
        zero_velocity=dict(delay_substeps=0, zero_velocity=True),
    )
    cases = {
        name: evaluate_case(bundle, scene, output / name, **settings)
        for name, settings in scenarios.items()
    }
    result = dict(
        bundle=str(bundle.directory.resolve()),
        cases=cases,
        rank=candidate_rank(list(cases.values())),
        hardware_ready=False,
        ranking_rule="completed_count, worst_step_fraction, negative_mean_joint_rmse",
    )
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_matrix(DeploymentBundle.load(args.bundle), args.scene, args.output)
    print(result["rank"])


if __name__ == "__main__":
    main()
