"""Offline kinematic diagnostics for actual physical trajectories.

Near-ground sole geometry is not a contact-force or support-polygon test.
Metrics use recorded timestamps and the model's named joint mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.controller import quaternion_matrix
from unisupermimic.deploy.run_sim2sim import write_json


def balance_metrics(bundle, scene, qpos, sample_time, *, window=None):
    qpos, t = np.asarray(qpos, dtype=float), np.asarray(sample_time, dtype=float)
    if t.ndim != 1 or len(t) < 3 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("at least three finite increasing timestamps required")
    if qpos.shape != (len(t), 36) or not np.isfinite(qpos).all():
        raise ValueError("finite [time,36] physical qpos required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(name).id] for name in bundle.metadata["joint_names"]]
    vadr = [model.jnt_dofadr[model.joint(name).id] for name in bundle.metadata["joint_names"]]
    feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] in feet
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
        and model.geom_contype[i]
    ]
    if len(geoms) != 8:
        raise ValueError("expected eight G1 spherical sole elements")
    points, ref_points, ref_velocities, rotations = [], [], [], []
    indices = np.clip(np.rint(t / 0.02).astype(int), 0, len(bundle.reference["q"]) - 1)
    ref = bundle.reference
    for row, index in zip(qpos, indices):
        data.qpos[:7] = row[:7]
        data.qpos[qadr] = row[7:]
        mujoco.mj_forward(model, data)
        points.append(data.geom_xpos[geoms].copy())
        rotations.append(quaternion_matrix(row[3:7]))
        data.qpos[:3] = ref["root_p"][index]
        data.qpos[3:7] = ref["root_quat"][index]
        data.qpos[qadr] = ref["q"][index]
        data.qvel[:3] = ref["root_v"][index]
        data.qvel[3:6] = ref["root_R"][index].T @ ref["root_w"][index]
        data.qvel[vadr] = ref["qd"][index]
        mujoco.mj_forward(model, data)
        ref_points.append(data.geom_xpos[geoms].copy())
        row_velocity = []
        for geom in geoms:
            spatial = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom, spatial, 0)
            row_velocity.append(spatial[3:].copy())
        ref_velocities.append(row_velocity)
    points, ref_points, R = np.array(points), np.array(ref_points), np.array(rotations)
    velocity = np.gradient(points, t, axis=0)
    ref_velocity = np.array(ref_velocities)
    bottom = points[..., 2] - model.geom_size[geoms, 0]
    ref_bottom = ref_points[..., 2] - model.geom_size[geoms, 0]
    near = bottom < 0.008
    stance = (ref_bottom < 0.02) & (np.linalg.norm(ref_velocity, axis=-1) < 0.25)
    gravity_error = np.arccos(np.clip((R[:, 2, :] * ref["root_R"][indices, 2, :]).sum(-1), -1, 1))
    derivative = np.gradient(R, t, axis=0)
    skew = R.transpose(0, 2, 1) @ derivative
    gyro = 0.5 * np.stack(
        (
            skew[:, 2, 1] - skew[:, 1, 2],
            skew[:, 0, 2] - skew[:, 2, 0],
            skew[:, 1, 0] - skew[:, 0, 1],
        ),
        axis=-1,
    )
    ref_gyro = (
        ref["root_R"][indices].transpose(0, 2, 1) @ ref["root_w"][indices, ..., None]
    ).squeeze(-1)
    angular_acceleration = np.gradient(gyro, t, axis=0)
    selected = (
        np.ones(len(t), dtype=bool) if window is None else (t >= window[0]) & (t <= window[1])
    )
    if selected.sum() < 3:
        raise ValueError("window requires at least three samples")
    speed2 = (velocity[..., :2] ** 2).sum(-1)[selected]

    def masked_rms(mask):
        values = speed2[mask[selected]]
        return float(np.sqrt(np.mean(values))) if len(values) else None

    return dict(
        samples=int(selected.sum()),
        window_seconds=[float(t[selected][0]), float(t[selected][-1])],
        gravity_error_rms_deg=float(np.degrees(np.sqrt(np.mean(gravity_error[selected] ** 2)))),
        gravity_error_p95_deg=float(np.degrees(np.quantile(gravity_error[selected], 0.95))),
        body_angular_velocity_error_rms_rad_s=float(
            np.sqrt(np.mean((gyro[selected] - ref_gyro[selected]) ** 2))
        ),
        body_angular_acceleration_rms_rad_s2=float(
            np.sqrt(np.mean(angular_acceleration[selected] ** 2))
        ),
        near_ground_point_speed_rms_m_s=masked_rms(near),
        reference_stance_point_speed_rms_m_s=masked_rms(stance),
        no_near_ground_point_fraction=float((~near[selected].any(-1)).mean()),
        geometry_only=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    bundle = DeploymentBundle.load(a.bundle)
    z = np.load(a.trajectory)
    t = z["sample_time"] if "sample_time" in z else z["tick"] * 0.001
    phases = bundle.metadata["sequence"]["segments"]
    result = {
        name: balance_metrics(bundle, a.scene, z["qpos"], t, window=(start * 0.02, end * 0.02))
        for name, (start, end) in phases.items()
    }
    result["hard_20_24"] = balance_metrics(bundle, a.scene, z["qpos"], t, window=(20, 24))
    result["hard_26_30"] = balance_metrics(bundle, a.scene, z["qpos"], t, window=(26, 30))
    write_json(a.output, result)
    print(result["dance"])


if __name__ == "__main__":
    main()
