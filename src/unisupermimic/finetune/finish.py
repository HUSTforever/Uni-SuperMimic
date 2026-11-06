"""Versioned fast reference finish and measured, high-rate standing checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.rl.algos.supermimic.geometry import quat_slerp


def _duration_steps(value, name, minimum):
    if (
        not math.isfinite(value)
        or value < minimum
        or abs(value / 0.02 - round(value / 0.02)) > 1e-6
    ):
        raise ValueError(f"{name} must be a finite multiple of20ms >= {minimum}")
    return round(value / 0.02)


def _sole_heights(poses, scene):
    model = mujoco.MjModel.from_xml_path(str(scene))
    if model.nq != 36:
        raise ValueError("G1 free-root 29-joint model required")
    data = mujoco.MjData(model)
    feet = {model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")}
    ids = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] in feet
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
        and model.geom_contype[i]
    ]
    if not ids:
        raise ValueError("standing finish requires spherical sole geometry")
    result = []
    for row in poses:
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        result.append(np.min(data.geom_xpos[ids, 2] - model.geom_size[ids, 0]))
    return np.asarray(result)


def build_fast_finish(source, scene, output, *, exit_seconds=1.0, hold_seconds=12.0):
    from .artifacts import publish_reference

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    exit_steps = _duration_steps(exit_seconds, "exit_seconds", 0.2)
    hold_steps = _duration_steps(hold_seconds, "hold_seconds", 1.0)
    seq = source.metadata.get("sequence", {})
    segments = seq.get("segments", {})
    if set(segments) != {"standing", "entry", "dance", "exit", "settle"}:
        raise ValueError("fast finish requires all five sequence phases")
    start = segments["dance"][1]
    old_end = segments["exit"][1]
    if segments["exit"][0] != start or segments["settle"][0] != old_end:
        raise ValueError("finish phases must be contiguous")
    ref = source.reference
    lower = np.asarray(source.metadata["joint_lower"], dtype=float)
    upper = np.asarray(source.metadata["joint_upper"], dtype=float)
    source_q = np.asarray(ref["q"], dtype=float)
    if (
        not np.isfinite(source_q).all()
        or (source_q < lower - 1e-5).any()
        or (source_q > upper + 1e-5).any()
    ):
        raise ValueError("source reference violates joint limits")
    stand = np.asarray(source.metadata["default_joint_positions"]) + np.asarray(
        source.metadata["context_offset"]
    )
    if not np.allclose(ref["q"][old_end:], stand, atol=2e-5, rtol=0):
        raise ValueError("source must end in a static default standing pose")
    if (
        not np.allclose(ref["root_p"][old_end:], ref["root_p"][-1], atol=2e-5, rtol=0)
        or not np.allclose(ref["root_R"][old_end:], ref["root_R"][-1], atol=2e-5, rtol=0)
        or ref["root_R"][-1, 2, 2] < math.cos(0.01)
    ):
        raise ValueError("source standing root must be static and upright")
    samples = np.linspace(start, old_end, exit_steps + 1)[1:]
    lower = np.minimum(np.floor(samples).astype(int), len(ref["q"]) - 2)
    alpha = samples - lower
    parts = {}
    for key in ("q", "root_p"):
        middle = ref[key][lower] * (1 - alpha[:, None]) + ref[key][lower + 1] * alpha[:, None]
        parts[key] = np.concatenate(
            (ref[key][: start + 1], middle, np.repeat(ref[key][-1:], [hold_steps], axis=0))
        )
    quat = torch.tensor(ref["root_quat"], dtype=torch.float64)
    middle = quat_slerp(quat[lower], quat[lower + 1], torch.tensor(alpha)).numpy()
    rotations = np.concatenate(
        (
            ref["root_quat"][: start + 1],
            middle,
            np.repeat(ref["root_quat"][-1:], hold_steps, axis=0),
        )
    )
    poses = np.c_[parts["root_p"], rotations, parts["q"]]
    heights = _sole_heights(poses[start : start + exit_steps + 1], scene)
    u = np.linspace(0, 1, exit_steps + 1)
    bump = 64 * u**3 * (1 - u) ** 3
    positive = bump > 1e-8
    lift = max(0.0, float(np.max((-heights[positive] - 0.002) / bump[positive])))
    if lift > 0.02:
        raise ValueError("exit requires more than2cm of ground correction")
    poses[start : start + exit_steps + 1, 2] += lift * bump
    corrected = _sole_heights(poses[start : start + exit_steps + 1], scene)
    if corrected.min() < -0.00202:
        raise ValueError("exit endpoints/trajectory penetrate the floor")
    metadata = copy.deepcopy(source.metadata)
    metadata["sequence"]["segments"]["exit"] = [start, start + exit_steps]
    metadata["sequence"]["segments"]["settle"] = [start + exit_steps, len(poses) - 1]
    metadata["sequence"]["fast_finish"] = dict(
        exit_seconds=float(exit_seconds),
        hold_seconds=float(hold_seconds),
        source_reference_sha256=hashlib.sha256(
            (source.directory / "reference.npz").read_bytes()
        ).hexdigest(),
        unchanged_pose_prefix_last_frame=int(start),
        floor_lift_peak_m=lift,
        exit_min_sole_height_m=float(corrected.min()),
        description="C2 source exit retimed; original feedback policy actively tracks standing tail",
    )
    return publish_reference(source, scene, output, poses, metadata)


def summarize_finish(
    metadata,
    qpos,
    joint_velocity,
    root_velocity,
    sample_time,
    *,
    max_recovery_seconds=1.5,
    required_hold_seconds=10.0,
):
    from .continuity_metrics import _clock

    t, dt = _clock(sample_time)
    if not math.isclose(dt, 0.005, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("finish verification requires200Hz physics samples")
    q, dq, v = (np.asarray(x, dtype=float) for x in (qpos, joint_velocity, root_velocity))
    if (
        q.shape != (len(t), 36)
        or dq.shape != (len(t), 29)
        or v.shape != (len(t), 3)
        or not all(np.isfinite(x).all() for x in (q, dq, v))
    ):
        raise ValueError("finite matching physical trajectory arrays required")
    for value in (max_recovery_seconds, required_hold_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("finish limits must be finite and positive")
    segments = metadata["sequence"]["segments"]
    dance_end = segments["dance"][1] * 0.02
    end = segments["settle"][1] * 0.02
    complete = len(t) == round(end / dt) + 1 and abs(t[0]) < 1e-7 and abs(t[-1] - end) < 1e-6
    stand = np.asarray(metadata["default_joint_positions"]) + np.asarray(metadata["context_offset"])
    error = np.sqrt(np.mean((q[:, 7:] - stand) ** 2, axis=1))
    speed = np.sqrt(np.mean(dq**2, axis=1))
    root_speed = np.linalg.norm(v, axis=1)
    tilt = np.arccos(np.clip(1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2), -1, 1))
    stable = (error < 0.1) & (speed < 0.3) & (root_speed < 0.1) & (tilt < 0.2) & (q[:, 2] >= 0.35)
    width = round(0.5 / dt) + 1
    recovery = None
    if len(t) >= width:
        # A brief quiet half-second followed by another wobble is not recovery.
        # Locate the final continuous stable run, then require confirmation and
        # the independent full-duration hold below.
        after_end = np.flatnonzero(t >= dance_end - 1e-7)
        if len(after_end):
            first = int(after_end[0])
            bad = np.flatnonzero(~stable[first:])
            first = first + int(bad[-1]) + 1 if len(bad) else first
            if len(t) - first >= width:
                recovery = float(t[first] - dance_end)
    final = t >= end - required_hold_seconds - 1e-7
    held = bool(
        complete
        and (segments["settle"][1] - segments["settle"][0]) * 0.02 >= required_hold_seconds
        and final.sum() >= round(required_hold_seconds / dt) + 1
        and stable[final].all()
    )
    return dict(
        complete=bool(complete),
        passed=bool(
            complete and recovery is not None and recovery <= max_recovery_seconds and held
        ),
        recovery_seconds=recovery,
        max_recovery_seconds=max_recovery_seconds,
        stable_detection_seconds=0.5,
        required_hold_seconds=required_hold_seconds,
        final_hold_passed=held,
        dance_end_seconds=dance_end,
        final_joint_error_max=float(error[final].max()) if final.any() else None,
        final_joint_speed_rms_max=float(speed[final].max()) if final.any() else None,
        final_root_speed_max=float(root_speed[final].max()) if final.any() else None,
        final_tilt_max=float(tilt[final].max()) if final.any() else None,
        limits=dict(joint_rmse_rad=0.1, joint_velocity_rms=0.3, root_speed=0.1, tilt_rad=0.2),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-bundle", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exit-seconds", type=float, default=1.0)
    p.add_argument("--hold-seconds", type=float, default=12.0)
    a = p.parse_args()
    print(
        build_fast_finish(
            DeploymentBundle.load(a.source_bundle),
            a.scene,
            a.output,
            exit_seconds=a.exit_seconds,
            hold_seconds=a.hold_seconds,
        )
    )
