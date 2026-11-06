"""Run exported G1 policy in independent native or loopback DDS simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .bundle import DeploymentBundle
from .controller import PolicyController
from .mujoco_sim import NativeSimulator


def write_json(path: Path, value: dict) -> None:
    """Publish status atomically so another process never reads a truncated file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def render_trajectory(scene: Path, qpos: np.ndarray, output: Path) -> None:
    """Render recorded physics states at native 50fps; never filter the trajectory."""
    import imageio.v2 as imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.1, 135, -12
    with mujoco.Renderer(model, height=540, width=960) as renderer:
        with imageio.get_writer(
            output, fps=50, codec="libx264", quality=7, macro_block_size=1
        ) as writer:
            for row in qpos:
                data.qpos[:] = row
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[:3] + [0, 0, 0.1]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())


def save_result(
    output: Path,
    bundle: DeploymentBundle,
    scene: Path,
    trajectory: dict,
    report: dict,
    *,
    video: bool,
) -> dict:
    arrays = {key: np.asarray(value) for key, value in trajectory.items()}
    with (output / "trajectory.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    frames = arrays.get("reference_frame", np.array([], dtype=int)).astype(int)
    if len(frames):
        ref = bundle.reference["q"][frames]
        actual = arrays["qpos"][:, 7:]
        report["joint_tracking_rmse_rad"] = float(np.sqrt(np.mean((actual - ref) ** 2)))
        if len(actual) > 2:
            velocity = np.diff(actual, axis=0) / 0.02
            accel = np.diff(velocity, axis=0) / 0.02
            report["joint_velocity_peak_rad_s"] = float(np.abs(velocity).max())
            report["joint_acceleration_rms_rad_s2"] = float(np.sqrt(np.mean(accel**2)))
    report["policy_sha256"] = hashlib.sha256(bundle.policy_path.read_bytes()).hexdigest()
    report["scene"] = str(scene.resolve())
    report["scene_sha256"] = hashlib.sha256(scene.read_bytes()).hexdigest()
    report["hardware_ready"] = False
    report["runtime"] = "standalone NumPy/TorchScript; no training environment/MDP"
    report["all_attempted_phases_passed"] = (
        report["tracking"]["completed"]
        and (not report["entry"]["attempted"] or report["entry"]["completed"])
        and (not report["exit"]["attempted"] or report["exit"]["completed"])
    )
    write_json(output / "report.json", report)
    if video and len(arrays.get("qpos", [])):
        render_trajectory(scene, arrays["qpos"], output / "policy.mp4")
    return report


def run_native(
    bundle: DeploymentBundle,
    scene: Path,
    output: Path,
    *,
    init_mode: str = "reference",
    video: bool = True,
    friction: float = 1.0,
    mass_scale: float = 1.0,
    sensor_noise: float = 0.0,
    seed: int = 0,
    exit_seconds: float = 2.0,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    simulator = NativeSimulator(bundle, scene, friction=friction, mass_scale=mass_scale)
    controller = PolicyController(bundle)
    simulator.reset(init_mode)
    rng = np.random.default_rng(seed)
    trajectory = dict(qpos=[], reference_frame=[], target=[], raw_action=[], sample_time=[])
    expected = len(bundle.reference["q"]) - 1
    report = dict(
        transport="native_mujoco",
        initialization=init_mode,
        physics_hz=200,
        policy_hz=50,
        standing_entry_verified=False,
        entry=dict(attempted=init_mode == "standing", completed=False),
        tracking=dict(expected_steps=expected, steps=0, completed=False, failure=None),
        exit=dict(attempted=False, completed=False),
        perturbation=dict(
            friction=friction, mass_scale=mass_scale, sensor_noise=sensor_noise, seed=seed
        ),
    )
    target = simulator.sensor().q.copy()
    if init_mode == "standing":
        default = np.array(bundle.metadata["default_joint_positions"])
        for step in range(150):
            # One second standing hold, then two-second pose blend, with physical dynamics.
            alpha = np.clip((step - 49) / 100, 0, 1)
            target = (1 - alpha) * default + alpha * bundle.reference["q"][0]
            simulator.advance(target)
            failure = simulator.failure()
            if failure:
                report["entry"]["failure"] = failure
                break
        else:
            error = float(np.sqrt(np.mean((simulator.sensor().q - bundle.reference["q"][0]) ** 2)))
            report["entry"]["joint_rmse_rad"] = error
            report["entry"]["completed"] = error < 0.2
            report["standing_entry_verified"] = error < 0.2
        if not report["entry"]["completed"]:
            report["tracking"]["failure"] = "standing entry failed"
    start = time.perf_counter()
    if not report["tracking"]["failure"]:
        for frame in range(expected):
            try:
                state = simulator.sensor()
                if sensor_noise:
                    state.q += rng.normal(0, sensor_noise, 29)
                    state.dq += rng.normal(0, sensor_noise * 10, 29)
                    state.gyro += rng.normal(0, sensor_noise * 2, 3)
                target = controller.step(state, frame)
                trajectory["qpos"].append(simulator.qpos())
                trajectory["reference_frame"].append(frame)
                trajectory["target"].append(target.copy())
                trajectory["raw_action"].append(controller.previous_action.copy())
                trajectory["sample_time"].append(state.sample_time)
                simulator.advance(target)
                report["tracking"]["steps"] += 1
                failure = simulator.failure(frame + 1)
                if failure:
                    raise ValueError(failure)
            except (ValueError, RuntimeError) as exc:
                report["tracking"]["failure"] = str(exc)
                break
    report["tracking"]["completed"] = (
        report["tracking"]["steps"] == expected and report["tracking"]["failure"] is None
    )
    report["tracking"]["wall_seconds"] = time.perf_counter() - start
    if report["tracking"]["completed"] and exit_seconds > 0:
        report["exit"]["attempted"] = True
        initial = target.copy()
        default = np.array(bundle.metadata["default_joint_positions"])
        nblend = max(1, round(exit_seconds / 0.02))
        for step in range(nblend + 50):
            alpha = min((step + 1) / nblend, 1)
            simulator.advance((1 - alpha) * initial + alpha * default)
            failure = simulator.failure()
            if failure:
                report["exit"]["failure"] = failure
                break
        else:
            error = float(np.sqrt(np.mean((simulator.sensor().q - default) ** 2)))
            report["exit"].update(completed=error < 0.2, joint_rmse_rad=error)
    if not trajectory["qpos"]:
        trajectory["qpos"] = np.empty((0, 36))
    report["clipped_target_values"] = controller.clipped_targets
    report["effort_saturation_fraction"] = simulator.saturated / max(simulator.torque_samples, 1)
    report["max_torque_limit_ratio"] = simulator.max_torque_ratio
    if controller.inference_ms:
        report["inference_ms"] = dict(
            p50=float(np.median(controller.inference_ms)),
            p99=float(np.percentile(controller.inference_ms, 99)),
            maximum=max(controller.inference_ms),
        )
    return save_result(output, bundle, Path(scene), trajectory, report, video=video)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=["native", "dds"], default="native")
    parser.add_argument(
        "--init",
        choices=["reference", "reference_zero_velocity", "standing", "sequence"],
        default="reference",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--sensor-noise", type=float, default=0)
    parser.add_argument("--domain", type=int, default=17)
    parser.add_argument("--watchdog-probe", action="store_true")
    args = parser.parse_args()
    bundle = DeploymentBundle.load(args.bundle)
    if args.watchdog_probe:
        from .dds_sim import run_watchdog_probe

        result = run_watchdog_probe(bundle, args.scene, args.output, domain=args.domain)
        print(json.dumps(result, indent=2))
        if not result["watchdog_triggered"]:
            raise SystemExit(1)
        return
    if args.transport == "dds":
        from .dds_sim import run_dds

        result = run_dds(
            bundle,
            args.scene,
            args.output,
            init_mode=args.init,
            domain=args.domain,
            video=not args.no_video,
        )
    elif args.init == "sequence":
        from unisupermimic.finetune.sequence_eval import evaluate_sequence

        result = evaluate_sequence(
            bundle,
            args.scene,
            args.output,
            video=not args.no_video,
            friction=args.friction,
            mass_scale=args.mass_scale,
            noise=args.sensor_noise,
        )
    else:
        result = run_native(
            bundle,
            args.scene,
            args.output,
            init_mode=args.init,
            video=not args.no_video,
            friction=args.friction,
            mass_scale=args.mass_scale,
            sensor_noise=args.sensor_noise,
        )
    print(json.dumps(result, indent=2))
    if not result["all_attempted_phases_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
