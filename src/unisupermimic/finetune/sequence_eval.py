"""Continuous sequence verification: every physical phase and final stability."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.controller import PolicyController
from unisupermimic.deploy.mujoco_sim import NativeSimulator
from unisupermimic.deploy.run_sim2sim import save_result


def summarize_sequence(metadata, reference, qpos, joint_velocity, root_velocity, steps):
    segments = metadata["sequence"]["segments"]
    phases = {}
    for name, (a, b) in segments.items():
        upto = min(b + 1, len(qpos))
        error = None
        if upto > a:
            error = float(np.sqrt(np.mean((qpos[a:upto, 7:] - reference["q"][a:upto]) ** 2)))
        phases[name] = dict(
            completed=steps >= b and len(qpos) > b,
            passed=steps >= b and len(qpos) > b and error is not None and error < 0.2,
            start_frame=a,
            end_frame=b,
            joint_rmse_rad=error,
        )
    full = all(v["completed"] for v in phases.values())
    final = dict(passed=False)
    if full and len(qpos) >= 50:
        q = qpos[-50:]
        qd = joint_velocity[-50:]
        v = root_velocity[-50:]
        tilt = np.arccos(np.clip(1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2), -1, 1))
        final.update(
            joint_velocity_rms=float(np.sqrt(np.mean(qd**2))),
            root_speed_max=float(np.linalg.norm(v, axis=-1).max()),
            tilt_max_rad=float(tilt.max()),
            standing_joint_rmse=float(np.sqrt(np.mean((q[:, 7:] - reference["q"][-1]) ** 2))),
        )
        final["passed"] = (
            final["joint_velocity_rms"] < 0.3
            and final["root_speed_max"] < 0.15
            and final["tilt_max_rad"] < 0.3
            and final["standing_joint_rmse"] < 0.2
        )
    return dict(
        phases=phases,
        final_hold=final,
        complete=all(v["passed"] for v in phases.values()) and final["passed"],
    )


def evaluate_sequence(
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
    step_observer=None,
):
    if "sequence" not in bundle.metadata:
        raise ValueError("sequence metadata required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sim = NativeSimulator(
        bundle, scene, friction=friction, mass_scale=mass_scale, step_observer=step_observer
    )
    sim.reset("reference_zero_velocity")
    controller = PolicyController(bundle)
    rng = np.random.default_rng(seed)
    expected = len(bundle.reference["q"]) - 1
    data = dict(qpos=[], reference_frame=[], target=[], raw_action=[], sample_time=[])
    velocities = []
    root_velocities = []
    report = dict(
        transport="native_sequence",
        initialization="stationary_sequence_zero_velocity",
        initial_velocity_norm=float(np.linalg.norm(sim.data.qvel)),
        standing_entry_verified=False,
        entry=dict(attempted=True, completed=False),
        exit=dict(attempted=True, completed=False),
        tracking=dict(expected_steps=expected, steps=0, completed=False, failure=None),
        delay_ms=delay_substeps * 5,
        perturbation=dict(friction=friction, mass_scale=mass_scale, noise=noise, seed=seed),
    )
    previous = sim.sensor().q.copy()
    for frame in range(expected):
        try:
            sensor = sim.sensor()
            velocities.append(sensor.dq.copy())
            root_velocities.append(sim.data.qvel[:3].copy())
            if noise:
                sensor.q += rng.normal(0, noise, 29)
                sensor.dq += rng.normal(0, noise * 10, 29)
                sensor.gyro += rng.normal(0, noise * 2, 3)
            target = controller.step(sensor, frame)
            data["qpos"].append(sim.qpos())
            data["reference_frame"].append(frame)
            data["target"].append(target)
            data["raw_action"].append(controller.previous_action.copy())
            data["sample_time"].append(sensor.sample_time)
            if delay_substeps and frame:
                sim.advance(previous, duration=0.005)
                sim.advance(target, duration=0.015)
            else:
                sim.advance(target)
            previous = target.copy()
            failure = sim.failure(frame + 1)
            if failure:
                raise ValueError(failure)
            report["tracking"]["steps"] += 1
        except (ValueError, RuntimeError) as exc:
            report["tracking"]["failure"] = str(exc)
            break
    if data["sample_time"] and sim.data.time > data["sample_time"][-1] + 1e-6:
        terminal = sim.sensor()
        data["qpos"].append(sim.qpos())
        data["reference_frame"].append(min(round(sim.data.time / 0.02), expected))
        data["target"].append(previous.copy())
        data["raw_action"].append(controller.previous_action.copy())
        data["sample_time"].append(terminal.sample_time)
        velocities.append(terminal.dq.copy())
        root_velocities.append(sim.data.qvel[:3].copy())
    report["terminal_sample_recorded"] = bool(
        data["sample_time"] and abs(data["sample_time"][-1] - sim.data.time) < 1e-6
    )
    report["tracking"]["completed"] = (
        report["tracking"]["steps"] == expected and report["tracking"]["failure"] is None
    )
    if not data["qpos"]:
        data["qpos"] = np.empty((0, 36))
    summary = summarize_sequence(
        bundle.metadata,
        bundle.reference,
        np.asarray(data["qpos"]),
        np.asarray(velocities)[: len(data["qpos"])],
        np.asarray(root_velocities)[: len(data["qpos"])],
        report["tracking"]["steps"],
    )
    report["sequence"] = summary
    report["entry"]["completed"] = (
        summary["phases"]["entry"]["passed"] and summary["phases"]["standing"]["passed"]
    )
    report["standing_entry_verified"] = report["entry"]["completed"]
    report["exit"]["completed"] = summary["complete"]
    report["clipped_target_values"] = controller.clipped_targets
    return save_result(output, bundle, Path(scene), data, report, video=video)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--delay", type=int, choices=[0, 1], default=0)
    a = p.parse_args()
    r = evaluate_sequence(
        DeploymentBundle.load(a.bundle), a.scene, a.output, delay_substeps=a.delay
    )
    print(r["tracking"], r["sequence"])
    raise SystemExit(0 if r["all_attempted_phases_passed"] else 1)
