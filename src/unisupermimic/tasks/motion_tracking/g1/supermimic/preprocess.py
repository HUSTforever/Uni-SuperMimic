"""Timestamp-aware G1 reference preparation; native qpos is always wxyz here."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .robot import G1_JOINT_NAMES

CSV_TO_QPOS = [0, 1, 2, 6, 3, 4, 5, *range(7, 36)]
QPOS_TO_CSV = [0, 1, 2, 4, 5, 6, 3, *range(7, 36)]


def smooth_values(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Symmetric Gaussian filtering, without adding a new numerical dependency."""
    if sigma_frames < 0.1:
        return values.copy()
    radius = min(math.ceil(3 * sigma_frames), len(values) - 1)
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma_frames) ** 2)
    weights /= weights.sum()
    pad = [(radius, radius), *[(0, 0)] * (values.ndim - 1)]
    padded = np.pad(values, pad, mode="edge")
    return sum(weight * padded[i : i + len(values)] for i, weight in enumerate(weights))


def continuous_quaternions(quats: np.ndarray) -> np.ndarray:
    quats = quats.copy()
    lengths = np.linalg.norm(quats, axis=1, keepdims=True)
    if (lengths < 1e-8).any():
        raise ValueError("zero quaternion in motion")
    quats /= lengths
    for index in range(1, len(quats)):
        if np.dot(quats[index - 1], quats[index]) < 0:
            quats[index] *= -1
    return quats


def interpolate_quaternions(quats: np.ndarray, samples: np.ndarray) -> np.ndarray:
    lo = np.minimum(samples.astype(int), len(quats) - 2)
    alpha = (samples - lo)[:, None]
    first, second = quats[lo], quats[lo + 1]
    dot = np.sum(first * second, axis=1, keepdims=True).clip(-1, 1)
    second = np.where(dot < 0, -second, second)
    theta = np.arccos(np.abs(dot))
    denominator = np.maximum(np.sin(theta), 1e-8)
    spherical = (np.sin((1 - alpha) * theta) * first + np.sin(alpha * theta) * second) / denominator
    linear = (1 - alpha) * first + alpha * second
    return continuous_quaternions(np.where(np.abs(dot) > 0.9995, linear, spherical))


def foot_heights(qpos_wxyz: np.ndarray, model_path: str | Path) -> np.ndarray:
    """Lowest point of each foot's four sphere collision geoms, in world metres."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    foot_geoms = []
    for side in ("left", "right"):
        ids = [model.geom(f"{side}_ankle_roll_link_collision_{i}").id for i in range(1, 5)]
        if any(model.geom_type[i] != mujoco.mjtGeom.mjGEOM_SPHERE for i in ids):
            raise ValueError("foot height calculation requires the G1 sphere collision model")
        foot_geoms.append(ids)
    heights = np.empty((len(qpos_wxyz), 2))
    for index, row in enumerate(qpos_wxyz):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        for side, ids in enumerate(foot_geoms):
            heights[index, side] = np.min(data.geom_xpos[ids, 2] - model.geom_size[ids, 0])
    return heights


def motion_metrics(qpos_wxyz: np.ndarray, fps: float = 50.0) -> dict:
    rows = np.asarray(qpos_wxyz, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 36 or len(rows) < 4:
        raise ValueError("metrics require at least four frames of 36-column native qpos")
    if not np.isfinite(rows).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("metrics require finite qpos and positive fps")
    result = {"frames": len(rows), "fps": fps, "duration_seconds": (len(rows) - 1) / fps}
    differences = rows[:, 7:]
    for name in ("velocity", "acceleration", "jerk"):
        differences = np.diff(differences, axis=0) * fps
        result[f"joint_{name}_rms"] = float(np.sqrt(np.mean(differences**2)))
        result[f"joint_{name}_peak"] = float(np.abs(differences).max())
    result["root_vertical_acceleration_peak"] = float(
        np.abs(np.diff(rows[:, 2], n=2) * fps**2).max()
    )
    return result


def prepare_motion(
    qpos: np.ndarray,
    src_fps: float,
    *,
    model_path: str | Path,
    output_fps: float = 30.0,
    smooth_seconds: float = 0.0,
    slowdown: float = 1.0,
    face_yaw_deg: float | None = None,
    ground: bool = False,
    preserve_flight: bool = False,
) -> tuple[np.ndarray, dict]:
    rows = np.asarray(qpos, dtype=np.float64).copy()
    if rows.ndim != 2 or rows.shape[1] != 36 or len(rows) < 2:
        raise ValueError("G1 qpos must have at least two rows and 36 columns")
    if not np.isfinite(rows).all():
        raise ValueError("qpos must be finite")
    for name, value in (("src_fps", src_fps), ("output_fps", output_fps), ("slowdown", slowdown)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(smooth_seconds) or smooth_seconds < 0:
        raise ValueError("smooth_seconds must be finite and nonnegative")
    if face_yaw_deg is not None and not np.isfinite(face_yaw_deg):
        raise ValueError("face_yaw_deg must be finite")
    rows[:, 3:7] = continuous_quaternions(rows[:, 3:7])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    names = [model.joint(i).name for i in range(1, model.njnt)]
    if names != G1_JOINT_NAMES or model.nq != 36:
        raise ValueError("model must use G1 natural joint order")
    if face_yaw_deg is not None:
        w, x, y, z = rows[0, 3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        angle = np.radians(face_yaw_deg) - yaw
        c, s = np.cos(angle), np.sin(angle)
        rows[:, :2] = rows[:, :2] @ np.array([[c, s], [-s, c]])
        half_c, half_s = np.cos(angle / 2), np.sin(angle / 2)
        w, x, y, z = rows[:, 3:7].T.copy()
        rows[:, 3:7] = np.stack(
            (
                half_c * w - half_s * z,
                half_c * x - half_s * y,
                half_c * y + half_s * x,
                half_c * z + half_s * w,
            ),
            axis=1,
        )
    sigma = smooth_seconds * src_fps
    rows[:, :3] = smooth_values(rows[:, :3], sigma)
    rows[:, 7:] = smooth_values(rows[:, 7:], sigma)
    # A sign-continuous normalized quaternion mean preserves rotations on S3.
    rows[:, 3:7] = continuous_quaternions(smooth_values(rows[:, 3:7], sigma))
    limits = model.jnt_range[1:]
    violations = int(((rows[:, 7:] < limits[:, 0]) | (rows[:, 7:] > limits[:, 1])).sum())
    rows[:, 7:] = np.clip(rows[:, 7:], limits[:, 0], limits[:, 1])
    source_duration = (len(rows) - 1) / src_fps
    count = max(2, round(source_duration * slowdown * output_fps) + 1)
    samples = np.linspace(0, len(rows) - 1, count)
    result = np.stack(
        [np.interp(samples, np.arange(len(rows)), rows[:, i]) for i in range(36)], axis=1
    )
    result[:, 3:7] = interpolate_quaternions(rows[:, 3:7], samples)
    flight = np.zeros(count, dtype=bool)
    if ground:
        heights = foot_heights(result, model_path).min(axis=1)
        baseline = np.percentile(heights, 10)
        root_baseline = np.percentile(result[:, 2], 10)
        if preserve_flight:
            flight = (heights > baseline + 0.035) & (result[:, 2] > root_baseline + 0.035)
        supported = np.flatnonzero(~flight)
        if len(supported) < 2:
            raise ValueError("cannot estimate ground without support frames")
        correction = np.interp(np.arange(count), supported, heights[supported])
        correction = smooth_values(correction, max(0.0, smooth_seconds * output_fps))
        correction = np.minimum(correction, heights)
        result[:, 2] -= correction
    metadata = {
        "schema_version": 1,
        "fps": output_fps,
        "source_fps": src_fps,
        "source_frames": len(rows),
        "frames": count,
        "duration_seconds": (count - 1) / output_fps,
        "source_duration_seconds": source_duration,
        "slowdown": slowdown,
        "smooth_seconds": smooth_seconds,
        "grounded": ground,
        "flight_frames": int(flight.sum()),
        "preserve_flight": preserve_flight,
        "clipped_joint_values": violations,
        "quaternion_order": "xyzw",
        "joint_names": G1_JOINT_NAMES,
        "columns": 36,
        "face_yaw_deg": face_yaw_deg,
    }
    return result[:, QPOS_TO_CSV], metadata
