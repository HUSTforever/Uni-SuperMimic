"""Two-process loopback HG simulation using the actual Unitree SDK serializer.

This is a project headless adapter, not the unmodified upstream simulator. It can
load pinned official Unitree G1 assets or the training model; reports name which.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .bundle import DeploymentBundle
from .controller import PolicyController
from .dds import (
    CommandWatchdog,
    DDSClient,
    MotorCommand,
    decode_command,
    initialize_dds,
    validate_sim_endpoint,
)
from .mujoco_sim import NativeSimulator
from .run_sim2sim import save_result, write_json


def tracking_failure(simulator: NativeSimulator, tick_ms: int) -> str | None:
    """Reference height is evaluator-only state, checked at the 50Hz boundary."""
    frame = tick_ms // 20
    if tick_ms % 20 == 0 and frame < len(simulator.bundle.reference["q"]):
        return simulator.failure(frame)
    return simulator.failure()


def serve(bundle: DeploymentBundle, scene: Path, output: Path, domain: int, init_mode: str) -> None:
    initialize_dds(domain)
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    simulator = NativeSimulator(bundle, scene)
    sequence_mode = init_mode == "sequence"
    if sequence_mode and "sequence" not in bundle.metadata:
        raise ValueError("sequence mode requires a sequence bundle")
    simulator.reset("reference_zero_velocity" if sequence_mode else init_mode)
    entry = dict(attempted=init_mode in ("standing", "sequence"), completed=False)
    if init_mode == "standing":
        default = np.array(bundle.metadata["default_joint_positions"])
        for step in range(150):
            alpha = np.clip((step - 49) / 100, 0, 1)
            simulator.advance((1 - alpha) * default + alpha * bundle.reference["q"][0])
            if simulator.failure():
                entry["failure"] = simulator.failure()
                break
        else:
            entry["joint_rmse_rad"] = float(
                np.sqrt(np.mean((simulator.sensor().q - bundle.reference["q"][0]) ** 2))
            )
            entry["completed"] = entry["joint_rmse_rad"] < 0.2
    lock = threading.Lock()
    watchdog = CommandWatchdog()
    state = dict(command=None, error=None, count=0, physical_fault=None)

    def receive(msg):
        with lock:
            try:
                command = decode_command(msg, bundle.metadata)
                watchdog.received(time.monotonic())
                if not watchdog.latched and not state["physical_fault"]:
                    state["command"] = command
                    state["count"] += 1
            except (ValueError, RuntimeError) as exc:
                state["error"] = str(exc)

    publisher = ChannelPublisher("rt/lowstate", LowState_)
    publisher.Init()
    subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    subscriber.Init(receive)
    metadata = dict(
        identity="project headless HG adapter; official SDK2 serialization",
        model=str(scene.resolve()),
        entry=entry,
        watchdog_triggered=False,
        physics_hz=200,
        state_hz=200,
        command_watchdog_ms=100,
        failure=None,
        failure_tick=None,
        damping_latched=False,
        received_commands=0,
        max_scheduler_lag_ms=0.0,
    )
    metadata["initial_velocity_norm"] = float(np.linalg.norm(simulator.data.qvel))
    metadata["sequence_mode"] = sequence_mode
    write_json(output / "simulator_ready.json", metadata)
    records, ticks, joint_velocities, root_velocities = [], [], [], []
    initial_time = simulator.data.time
    started = None
    physical_steps = 0
    last_record_tick = -1
    ready_deadline = time.monotonic() + 30
    try:
        while not (output / "stop_simulator").exists():
            now = time.monotonic()
            with lock:
                command = state["command"]
                expired = watchdog.expired(now)
                command_error = state["error"]
                physical_fault = state["physical_fault"]
                metadata["received_commands"] = state["count"]
            if command is not None and started is None:
                started = now
            if started is None and now > ready_deadline:
                metadata["failure"] = "no controller command before startup timeout"
                break
            sensor = simulator.sensor()
            tick = round((simulator.data.time - initial_time) * 1000) % 2**32
            message = unitree_hg_msg_dds__LowState_()
            message.mode_pr, message.mode_machine, message.tick = 0, 0, tick
            message.imu_state.quaternion = sensor.quaternion.tolist()
            message.imu_state.gyroscope = sensor.gyro.tolist()
            for i in range(29):
                message.motor_state[i].mode = 1
                message.motor_state[i].q = float(sensor.q[i])
                message.motor_state[i].dq = float(sensor.dq[i])
            message.crc = CRC().Crc(message)
            if not publisher.Write(message):
                raise ValueError("simulator DDS LowState write failed")
            if tick != last_record_tick:
                ticks.append(tick)
                records.append(simulator.qpos())
                joint_velocities.append(sensor.dq.copy())
                root_velocities.append(simulator.data.qvel[:3].copy())
                last_record_tick = tick
            if expired or command_error or physical_fault:
                metadata["watchdog_triggered"] = expired
                metadata["damping_latched"] = True
                if metadata["failure"] is None:
                    metadata["failure"] = (
                        command_error or physical_fault or "LowCmd timeout; damping latched"
                    )
                    metadata["failure_tick"] = tick
                command = MotorCommand(sensor.q, np.zeros(29), np.maximum(2, simulator.kd))
            if command is not None:
                simulator.advance(command.target, kp=command.kp, kd=command.kd, duration=0.005)
                physical_steps += 1
                failure = tracking_failure(simulator, physical_steps * 5)
                if failure and metadata["failure"] is None:
                    metadata["failure"] = failure
                    metadata["failure_tick"] = tick + 5
                # Continue damping after a physical fault so a dead process never
                # leaves the previous high-gain command held indefinitely.
                if failure:
                    with lock:
                        state["physical_fault"] = failure
                deadline = started + physical_steps * 0.005
                lag = time.monotonic() - deadline
                metadata["max_scheduler_lag_ms"] = max(metadata["max_scheduler_lag_ms"], lag * 1000)
                if lag < 0:
                    time.sleep(-lag)
            else:
                time.sleep(0.005)
            if metadata["failure"]:
                write_json(output / "simulator_status.json", metadata)
    finally:
        metadata["watchdog_triggered"] = metadata["watchdog_triggered"] or watchdog.latched
        metadata["effort_saturation_fraction"] = simulator.saturated / max(
            simulator.torque_samples, 1
        )
        metadata["max_torque_limit_ratio"] = simulator.max_torque_ratio
        np.savez_compressed(
            output / "simulator_trajectory.npz",
            tick=np.array(ticks),
            qpos=np.array(records),
            joint_velocity=np.array(joint_velocities),
            root_velocity=np.array(root_velocities),
        )
        write_json(output / "simulator_report.json", metadata)
        subscriber.Close()
        publisher.Close()


def run_dds(
    bundle: DeploymentBundle,
    scene: Path,
    output: Path,
    *,
    init_mode: str = "reference",
    domain: int = 17,
    video: bool = True,
    watchdog_probe: bool = False,
) -> dict:
    validate_sim_endpoint("lo", domain)
    sequence_mode = init_mode == "sequence"
    if sequence_mode and "sequence" not in bundle.metadata:
        raise ValueError("sequence mode requires a sequence bundle")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    controller = PolicyController(bundle)
    expected = len(bundle.reference["q"]) - 1
    trajectory = dict(qpos=[], reference_frame=[], target=[], raw_action=[], sample_time=[])
    report = dict(
        transport="unitree_sdk2_hg_dds_loopback",
        domain=domain,
        interface="lo",
        initialization=init_mode,
        standing_entry_verified=False,
        entry=dict(attempted=init_mode == "standing", completed=False),
        tracking=dict(expected_steps=expected, steps=0, completed=False, failure=None),
        exit=dict(attempted=False, completed=False),
    )
    command = [
        sys.executable,
        "-m",
        "unisupermimic.deploy.dds_sim",
        "--serve",
        "--bundle",
        str(bundle.directory.resolve()),
        "--scene",
        str(Path(scene).resolve()),
        "--output",
        str(output),
        "--domain",
        str(domain),
        "--init",
        init_mode,
    ]
    client = None
    sequence_last_state = None
    simulator_report = {}
    with (output / "simulator.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            while not (output / "simulator_ready.json").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise ValueError("DDS simulator startup failed; see simulator.log")
                time.sleep(0.02)
            ready = json.loads((output / "simulator_ready.json").read_text())
            if sequence_mode:
                report["initial_velocity_norm"] = ready["initial_velocity_norm"]
            report["entry"] = ready["entry"]
            report["standing_entry_verified"] = ready["entry"].get("completed", False)
            if init_mode == "standing" and not report["standing_entry_verified"]:
                raise ValueError("standing entry failed in simulator")
            client = DDSClient(bundle.metadata, domain)
            client.state_at(0, timeout=5)
            for frame in range(expected):
                state = client.state_at(frame * 0.02)
                if sequence_mode:
                    sequence_last_state = state
                if state.sample_time - frame * 0.02 > 0.02 + 1e-6:
                    raise ValueError("DDS sensor arrived more than one policy frame late")
                target = controller.step(state, frame)
                client.send(target)
                trajectory["sample_time"].append(state.sample_time)
                trajectory["reference_frame"].append(frame)
                trajectory["target"].append(target.copy())
                trajectory["raw_action"].append(controller.previous_action.copy())
                if watchdog_probe and frame == min(5, expected - 1):
                    report["dds"] = dict(
                        received_states=client.received_count, sent_commands=client.command_count
                    )
                    client.close()
                    client = None
                    time.sleep(0.2)
                    raise ValueError("intentional controller publication stop for watchdog probe")
                state = client.state_at((frame + 1) * 0.02)
                if sequence_mode:
                    sequence_last_state = state
                status = output / "simulator_status.json"
                if status.exists():
                    fault = json.loads(status.read_text())
                    if fault["failure_tick"] <= round((frame + 1) * 20):
                        raise ValueError(fault["failure"])
                report["tracking"]["steps"] += 1
            report["tracking"]["completed"] = True
            report["exit"]["attempted"] = True
            if not sequence_mode:
                initial, default = (
                    target.copy(),
                    np.array(bundle.metadata["default_joint_positions"]),
                )
                for step in range(150):
                    alpha = min((step + 1) / 100, 1)
                    target = (1 - alpha) * initial + alpha * default
                    client.send(target)
                    state = client.state_at((expected + step + 1) * 0.02)
                    status = output / "simulator_status.json"
                    if status.exists():
                        raise ValueError(
                            "exit physical fault: " + json.loads(status.read_text())["failure"]
                        )
                report["exit"]["joint_rmse_rad"] = float(np.sqrt(np.mean((state.q - default) ** 2)))
                report["exit"]["completed"] = report["exit"]["joint_rmse_rad"] < 0.2
        except (ValueError, RuntimeError) as exc:
            stage = "exit" if report["tracking"]["completed"] else "tracking"
            report[stage]["failure"] = str(exc)
        finally:
            if client is not None:
                if client.latest is not None:
                    from .finish import completion_command

                    completed = bool(
                        report["tracking"]["completed"]
                        and not report["tracking"].get("failure")
                        and not report["exit"].get("failure")
                        and (sequence_mode or report["exit"].get("completed"))
                    )
                    last_target = locals().get(
                        "target", np.asarray(bundle.metadata["default_joint_positions"])
                    )
                    final_target, damping, assessment = completion_command(
                        bundle.metadata, client.latest, last_target, completed=completed
                    )
                    report["terminal_control"] = assessment
                    if (
                        bundle.metadata.get("sequence", {}).get("fast_finish")
                        and completed
                        and damping
                    ):
                        report["exit"]["failure"] = (
                            "normal completion did not reach a stationary standing pose"
                        )
                    try:
                        client.send(final_target, damping=damping)
                    except (ValueError, RuntimeError):
                        pass
                report["dds"] = dict(
                    received_states=client.received_count, sent_commands=client.command_count
                )
                client.close()
            (output / "stop_simulator").touch()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
            path = output / "simulator_report.json"
            if path.exists():
                simulator_report = json.loads(path.read_text())
    if simulator_report.get("failure") and not report["tracking"]["completed"]:
        report["tracking"]["failure"] = simulator_report["failure"]
    report["simulator"] = simulator_report
    report.setdefault("dds", dict(received_states=0, sent_commands=0))
    report["clipped_target_values"] = controller.clipped_targets
    report["watchdog_probe"] = watchdog_probe
    if controller.inference_ms:
        report["inference_ms"] = dict(
            p50=float(np.median(controller.inference_ms)),
            p99=float(np.percentile(controller.inference_ms, 99)),
            maximum=max(controller.inference_ms),
        )
    if (
        sequence_mode
        and sequence_last_state is not None
        and trajectory["sample_time"]
        and sequence_last_state.sample_time > trajectory["sample_time"][-1] + 1e-6
    ):
        trajectory["sample_time"].append(sequence_last_state.sample_time)
        trajectory["reference_frame"].append(
            min(round(sequence_last_state.sample_time / 0.02), expected)
        )
        trajectory["target"].append(trajectory["target"][-1].copy())
        trajectory["raw_action"].append(trajectory["raw_action"][-1].copy())
    if sequence_mode:
        report["terminal_sample_recorded"] = (
            sequence_last_state is not None
            and bool(trajectory["sample_time"])
            and abs(trajectory["sample_time"][-1] - sequence_last_state.sample_time) < 1e-6
        )
    telemetry = output / "simulator_trajectory.npz"
    sequence_velocities = []
    sequence_root_velocities = []
    if telemetry.exists() and trajectory["sample_time"]:
        with np.load(telemetry, allow_pickle=False) as data:
            times = data["tick"] / 1000
            for sample_time in trajectory["sample_time"]:
                index = int(np.argmin(np.abs(times - sample_time)))
                if abs(times[index] - sample_time) > 0.001:
                    raise ValueError("missing simulator telemetry for a policy observation")
                trajectory["qpos"].append(data["qpos"][index])
                if sequence_mode:
                    sequence_velocities.append(data["joint_velocity"][index])
                    sequence_root_velocities.append(data["root_velocity"][index])
    if not trajectory["qpos"]:
        # Empty tracking is represented with zero rows, not an invented rollout.
        trajectory["qpos"] = np.empty((0, 36))
    if sequence_mode:
        from unisupermimic.finetune.sequence_eval import summarize_sequence

        summary = summarize_sequence(
            bundle.metadata,
            bundle.reference,
            np.asarray(trajectory["qpos"]),
            np.asarray(sequence_velocities),
            np.asarray(sequence_root_velocities),
            report["tracking"]["steps"],
        )
        report["sequence"] = summary
        report["entry"]["attempted"] = True
        report["entry"]["completed"] = (
            summary["phases"]["standing"]["passed"] and summary["phases"]["entry"]["passed"]
        )
        report["standing_entry_verified"] = report["entry"]["completed"]
        report["exit"]["completed"] = (
            summary["complete"]
            and report["tracking"]["completed"]
            and not report["exit"].get("failure")
        )
    return save_result(output, bundle, Path(scene), trajectory, report, video=video)


def run_watchdog_probe(
    bundle: DeploymentBundle, scene: Path, output: Path, *, domain: int = 19
) -> dict:
    """Abruptly exit an actual DDS publisher process, then observe receiver damping."""
    validate_sim_endpoint("lo", domain)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = [
        "--bundle",
        str(bundle.directory.resolve()),
        "--scene",
        str(Path(scene).resolve()),
        "--output",
        str(output),
        "--domain",
        str(domain),
    ]
    prefix = [sys.executable, "-m", "unisupermimic.deploy.dds_sim"]
    publisher = None
    with (
        (output / "simulator.log").open("x") as log,
        (output / "publisher.log").open("x") as publog,
    ):
        simulator = subprocess.Popen(
            prefix + ["--serve"] + common, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 20
            while not (output / "simulator_ready.json").exists():
                if simulator.poll() is not None or time.monotonic() > deadline:
                    raise ValueError("watchdog simulator failed to start")
                time.sleep(0.02)
            publisher = subprocess.Popen(
                prefix + ["--probe-publisher"] + common, stdout=publog, stderr=subprocess.STDOUT
            )
            exit_code = publisher.wait(timeout=15)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                status = output / "simulator_status.json"
                if status.exists() and json.loads(status.read_text())["watchdog_triggered"]:
                    break
                time.sleep(0.01)
        finally:
            if publisher is not None and publisher.poll() is None:
                publisher.terminate()
                publisher.wait(timeout=5)
            (output / "stop_simulator").touch()
            try:
                simulator.wait(timeout=5)
            except subprocess.TimeoutExpired:
                simulator.terminate()
                simulator.wait(timeout=5)
    result = json.loads((output / "simulator_report.json").read_text())
    result.update(
        publisher_exit_code=exit_code,
        fault_injection="abrupt publisher process exit via os._exit(0)",
        domain=domain,
        interface="lo",
        hardware_ready=False,
    )
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--probe-publisher", action="store_true")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", type=int, default=17)
    parser.add_argument("--init", default="reference")
    args = parser.parse_args()
    bundle = DeploymentBundle.load(args.bundle)
    if args.probe_publisher:
        client = DDSClient(bundle.metadata, args.domain)
        state = client.state_at(0, timeout=5)
        target = np.clip(state.q, bundle.metadata["joint_lower"], bundle.metadata["joint_upper"])
        for _ in range(3):
            client.send(target)
            time.sleep(0.02)
        os._exit(0)
    serve(bundle, args.scene, args.output, args.domain, args.init)


if __name__ == "__main__":
    main()
