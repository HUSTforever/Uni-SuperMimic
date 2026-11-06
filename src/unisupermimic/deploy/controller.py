"""Standalone policy input/action bridge. No training environment on the hot path."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .bundle import DeploymentBundle


@dataclass
class SensorState:
    """Natural-order motor state and *pelvis* IMU, with simulation time in seconds.

    quaternion is wxyz, mapping pelvis vectors into the IMU world frame. Gyroscope
    is already expressed in pelvis coordinates. There is no root-position input.
    """

    q: np.ndarray
    dq: np.ndarray
    quaternion: np.ndarray
    gyro: np.ndarray
    sample_time: float


def finite_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = finite_vector(quaternion, 4, "IMU quaternion")
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if abs(norm - 1) > 0.02:
        raise ValueError("IMU quaternion must have unit norm (wxyz)")
    w, x, y, z = np.array([w, x, y, z]) / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def build_observation(
    state: SensorState,
    default_q: np.ndarray,
    context_offset: np.ndarray,
    previous_action: np.ndarray,
    reference_rotation: np.ndarray,
    world_alignment: np.ndarray | None = None,
) -> np.ndarray:
    """Build raw 99D inputs; normalization and action smoothing live in the model."""
    q = finite_vector(state.q, 29, "joint positions")
    dq = finite_vector(state.dq, 29, "joint velocities")
    gyro = finite_vector(state.gyro, 3, "pelvis body gyro")
    rotation = quaternion_matrix(state.quaternion)
    if world_alignment is not None:
        rotation = world_alignment @ rotation
    ref = np.asarray(reference_rotation)
    if ref.shape != (3, 3) or not np.isfinite(ref).all():
        raise ValueError("reference rotation must be finite 3x3")
    return np.concatenate((
        q - finite_vector(default_q, 29, "default q")
          - finite_vector(context_offset, 29, "context offset"),
        dq, gyro, -rotation[2, :],
        finite_vector(previous_action, 29, "previous applied raw action"),
        (rotation.T @ ref)[:, :2].reshape(6),
    )).astype(np.float32)


class PolicyController:
    """One inference per reference frame; faults remain latched until a new run."""

    def __init__(self, bundle: DeploymentBundle, *, align_heading: bool = True):
        self.bundle = bundle
        self.meta = bundle.metadata
        self.default_q = np.array(self.meta["default_joint_positions"])
        self.offset = np.array(self.meta["context_offset"])
        self.scales = np.array(self.meta["action_scales"])
        self.lower, self.upper = (np.array(self.meta[k]) for k in ("joint_lower", "joint_upper"))
        self.policy = torch.jit.load(str(bundle.policy_path), map_location="cpu").eval()
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.history: np.ndarray | None = None
        self.alignment = np.eye(3)
        self.align_heading = align_heading
        self.last_frame = -1
        self.last_sample_time: float | None = None
        self.fault: str | None = None
        self.clipped_targets = 0
        self.inference_ms: list[float] = []
        self.last_proposed_action = np.zeros(29, dtype=np.float32)
        # Warm the actual exported graph before a real-time simulation starts.
        with torch.inference_mode():
            warm = self.policy(torch.zeros(1, self.meta["hist"], 99),
                               torch.zeros(1, self.meta["goal_dim"]))
        if tuple(warm.shape) != (1, 29) or not bool(torch.isfinite(warm).all()):
            raise ValueError("exported policy must produce finite [1,29] raw actions")

    def step(self, state: SensorState, frame: int) -> np.ndarray:
        if self.fault is not None:
            raise ValueError(f"controller fault is latched: {self.fault}")
        try:
            return self._step(state, frame)
        except (ValueError, RuntimeError, TypeError) as exc:
            self.fault = str(exc)
            raise ValueError(f"controller fault: {exc}") from exc

    def _step(self, state: SensorState, frame: int) -> np.ndarray:
        if (not isinstance(frame, (int, np.integer)) or frame != self.last_frame + 1
                or not 0 <= frame < len(self.bundle.reference["q"]) - 1):
            raise ValueError("reference frame sequence must start at zero and advance once")
        if not math.isfinite(state.sample_time) or state.sample_time < 0:
            raise ValueError("sensor sample time must be finite and nonnegative")
        if self.last_sample_time is not None:
            delta = state.sample_time - self.last_sample_time
            if not 0 < delta <= 0.06:
                raise ValueError("sensor sample time is stale/out of order or has a large gap")
        rotation = quaternion_matrix(state.quaternion)
        if rotation[2, 2] < math.cos(1.2):
            raise ValueError("pelvis tilt exceeds simulation controller limit")
        ref = self.bundle.reference["root_R"][frame]
        if frame == 0 and self.align_heading:
            yaw = math.atan2(ref[1, 0], ref[0, 0]) - math.atan2(rotation[1, 0], rotation[0, 0])
            c, s = math.cos(yaw), math.sin(yaw)
            self.alignment = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        obs = build_observation(state, self.default_q, self.offset, self.previous_action,
                                ref, self.alignment)
        if self.history is None:
            history = np.repeat(obs[None], self.meta["hist"], axis=0)
        else:
            history = np.concatenate((self.history[1:], obs[None]))
        goal = self.bundle.reference["goals"][frame]
        started = time.perf_counter()
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(history[None]), torch.from_numpy(goal[None]))
        self.inference_ms.append((time.perf_counter() - started) * 1000)
        if tuple(action.shape) != (1, 29):
            raise ValueError("policy output must have shape [1,29]")
        proposed = finite_vector(action[0].detach().cpu().numpy(), 29, "policy output")
        target = self.default_q + self.offset + self.scales * proposed
        applied = np.clip(target, self.lower, self.upper)
        self.clipped_targets += int(np.count_nonzero(applied != target))
        self.previous_action = ((applied - self.default_q - self.offset) / self.scales).astype(np.float32)
        self.last_proposed_action = proposed.astype(np.float32)
        self.history = history
        self.last_frame = int(frame)
        self.last_sample_time = float(state.sample_time)
        return applied
