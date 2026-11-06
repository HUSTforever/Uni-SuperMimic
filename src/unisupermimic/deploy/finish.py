"""Normal completion keeps the standing command; faults retain damping behavior."""

import math

import numpy as np

from .controller import finite_vector, quaternion_matrix


def completion_command(metadata, state, last_target, *, completed):
    stand = np.asarray(metadata["default_joint_positions"]) + np.asarray(metadata["context_offset"])
    lower, upper = (np.asarray(metadata[k]) for k in ("joint_lower", "joint_upper"))
    fallback = np.clip(stand, lower, upper)
    metrics = dict(mode="damping", standing_pose_check=False)
    if state is not None:
        try:
            q = finite_vector(state.q, 29, "final joint positions")
            fallback = np.clip(q, lower, upper)
            dq = finite_vector(state.dq, 29, "final joint velocities")
            gyro = finite_vector(state.gyro, 3, "final gyro")
            rotation = quaternion_matrix(state.quaternion)
            error = float(np.sqrt(np.mean((q - stand) ** 2)))
            speed = float(np.sqrt(np.mean(dq**2)))
            tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
            metrics.update(
                joint_rmse_rad=error,
                joint_velocity_rms=speed,
                tilt_rad=tilt,
                gyro_norm=float(np.linalg.norm(gyro)),
            )
            static = error < 0.1 and speed < 0.3 and tilt < 0.2 and np.linalg.norm(gyro) < 0.3
            metrics["standing_pose_check"] = bool(static)
            enabled = bool(metadata.get("sequence", {}).get("fast_finish"))
            if enabled and completed and static:
                target = np.clip(finite_vector(last_target, 29, "last active target"), lower, upper)
                metrics["mode"] = "standing_hold"
                metrics["scope"] = (
                    "last command before simulator shutdown; communication watchdog unchanged"
                )
                return target, False, metrics
        except (ValueError, TypeError, AttributeError) as exc:
            metrics["assessment_error"] = str(exc)
    return fallback, True, metrics
