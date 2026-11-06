"""Controlled simulation-only servo diagnostic. Does not alter deployed artifacts."""

import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.controller import PolicyController
from unisupermimic.deploy.mujoco_sim import NativeSimulator, pd_torque

PROJECT = Path("finetune_projects/g1_continuity_20261029")
SOURCE = Path("finetune_projects/g1_continuous_20261029/candidates/ground008_000200")
SCENE = Path("deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")


def high_frequency_rms(x, dt, cut=8):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    if n < 16:
        return np.full(x.shape[1], np.nan)
    tt = np.arange(n, dtype=float)
    tt -= tt.mean()
    centered = x - x.mean(0)
    centered -= tt[:, None] * (tt @ centered / (tt @ tt))[None, :]
    w = np.hanning(n)
    F = np.fft.rfft(centered * w[:, None], axis=0)
    f = np.fft.rfftfreq(n, dt)
    factor = np.full(len(f), 2.0)
    factor[0] = 1
    if n % 2 == 0:
        factor[-1] = 1
    return np.sqrt((np.abs(F[f > cut]) ** 2 * factor[f > cut, None]).sum(0) / (n * np.sum(w * w)))


def metrics(bundle, q, dq, v, t):
    result = {}
    ankle = [i for i, n in enumerate(bundle.metadata["joint_names"]) if "ankle" in n]
    for name, (a, b) in bundle.metadata["sequence"]["segments"].items():
        mask = (t >= a * 0.02 - 1e-7) & (t <= b * 0.02 + 1e-7)
        if mask.sum() < 16:
            continue
        acc = np.gradient(v, t, axis=0)
        qa = np.gradient(dq, t, axis=0)
        qj = np.gradient(qa, t, axis=0)
        hfq = high_frequency_rms(q[mask, 7:], 0.005)
        hfv = high_frequency_rms(dq[mask], 0.005)
        result[name] = dict(
            samples=int(mask.sum()),
            ankle_hf_position_rms_rad=hfq[ankle].tolist(),
            ankle_hf_velocity_rms_rad_s=hfv[ankle].tolist(),
            ankle_hf_position_max_rad=float(hfq[ankle].max()),
            ankle_hf_velocity_max_rad_s=float(hfv[ankle].max()),
            root_vertical_hf_position_rms_m=float(high_frequency_rms(q[mask, 2], 0.005)[0]),
            root_vertical_acceleration_rms=float(np.sqrt(np.mean(acc[mask, 2] ** 2))),
            root_vertical_acceleration_peak=float(np.abs(acc[mask, 2]).max()),
            joint_acceleration_rms=float(np.sqrt(np.mean(qa[mask] ** 2))),
            joint_jerk_rms=float(np.sqrt(np.mean(qj[mask] ** 2))),
        )
    frames = np.arange(0, len(q), 4)
    ref = bundle.reference["q"][: len(frames)]
    result["joint_rmse"] = float(np.sqrt(np.mean((q[frames, 7:] - ref) ** 2)))
    dance_frames = frames[(frames * 0.005 >= 5) & (frames * 0.005 <= 27.96)]
    if len(dance_frames):
        result["dance_joint_rmse"] = float(
            np.sqrt(np.mean((q[dance_frames, 7:] - bundle.reference["q"][dance_frames // 4]) ** 2))
        )
    return result


def run(bundle, name, kp_factor=1.0, kd_factor=1.0, ramp=False, command_alpha=1.0):
    output = PROJECT / "evaluation" / name
    output.mkdir(parents=True, exist_ok=False)
    sim = NativeSimulator(bundle, SCENE)
    sim.reset("reference_zero_velocity")
    pi = PolicyController(bundle)
    ankle = np.array(["ankle" in n for n in bundle.metadata["joint_names"]])
    kp = sim.kp.copy()
    kd = sim.kd.copy()
    kp[ankle] *= kp_factor
    kd[ankle] *= kd_factor
    if command_alpha != 1:

        class AnkleFilter(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                values = np.ones(29, dtype=np.float32)
                values[ankle] = command_alpha
                self.register_buffer("alpha", torch.from_numpy(values))

            def forward(self, history, goal):
                raw = self.inner(history, goal)
                previous = history[:, -1, 64:93]
                return previous + self.alpha * (raw - previous)

        pi.policy = AnkleFilter(pi.policy).eval()
    qs = [sim.qpos()]
    dqs = [sim.data.qvel[sim.vadr].copy()]
    vs = [sim.data.qvel[:3].copy()]
    ts = [sim.data.time]
    previous = sim.sensor().q.copy()
    failure = None
    start = time.monotonic()
    steps = 0
    for frame in range(len(bundle.reference["q"]) - 1):
        target = pi.step(sim.sensor(), frame)
        for j in range(4):
            applied = previous + (target - previous) * (j + 1) / 4 if ramp else target
            tau, _ = pd_torque(
                sim.data.qpos[sim.qadr], sim.data.qvel[sim.vadr], applied, kp, kd, sim.effort
            )
            sim.data.ctrl[sim.actuators] = tau
            mujoco.mj_step(sim.model, sim.data)
            qs.append(sim.qpos())
            dqs.append(sim.data.qvel[sim.vadr].copy())
            vs.append(sim.data.qvel[:3].copy())
            ts.append(sim.data.time)
        mujoco.mj_forward(sim.model, sim.data)
        previous = target.copy()
        failure = sim.failure(frame + 1)
        if failure:
            break
        steps += 1
    q, dq, v, t = map(np.asarray, (qs, dqs, vs, ts))
    np.savez_compressed(
        output / "trajectory_200hz.npz", qpos=q, joint_velocity=dq, root_velocity=v, sample_time=t
    )
    report = dict(
        name=name,
        kp_ankle_factor=kp_factor,
        kd_ankle_factor=kd_factor,
        ramp=ramp,
        command_alpha=command_alpha,
        effective_ankle_action_smoothing=0.35 * command_alpha,
        steps=steps,
        expected_steps=len(bundle.reference["q"]) - 1,
        completed=steps == len(bundle.reference["q"]) - 1 and failure is None,
        failure=failure,
        wall_seconds=time.monotonic() - start,
        metrics=metrics(bundle, q, dq, v, t),
        hardware_ready=False,
    )
    if name == "baseline":
        reference = np.load(
            "finetune_projects/g1_continuous_20261029/evaluation/ground008_000200/trajectory.npz"
        )["qpos"]
        error = float(np.max(np.abs(q[::4] - reference)))
        report["baseline_max_abs_state_difference"] = error
        assert error < 1e-10, error
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    torch.set_num_threads(1)
    bundle = DeploymentBundle.load(SOURCE)
    rows = []
    variants = [
        ("baseline", 1.0, 1.0, False),
        ("ankle_kd_125", 1.0, 1.25, False),
        ("ankle_kd_150", 1.0, 1.5, False),
        ("ankle_kd_075", 1.0, 0.75, False),
        ("ankle_kp_075", 0.75, 1.0, False),
        ("ankle_kp_050", 0.5, 1.0, False),
        ("ramp", 1.0, 1.0, True),
        ("ramp_kd_125", 1.0, 1.25, True),
    ]
    for args in variants:
        rows.append(run(bundle, *args))
        (PROJECT / "servo_audit_results.json").write_text(json.dumps(rows, indent=2) + "\n")
