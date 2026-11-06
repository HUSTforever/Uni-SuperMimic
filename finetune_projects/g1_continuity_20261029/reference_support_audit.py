"""Diagnostic centroidal/contact consistency; not a physical feasibility proof.

Momentum balance includes angular-momentum rate. Derivatives use 50Hz data;
potential support points include sole centers within 25mm of the floor.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.controller import quaternion_matrix
from unisupermimic.deploy.run_sim2sim import write_json

PROJECT = Path("finetune_projects/g1_continuity_20261029")


def hull(points):
    p = sorted(set(map(tuple, points)))
    if len(p) < 3:
        return np.asarray(p)

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    chains = []
    for sequence in (p, p[::-1]):
        chain = []
        for point in sequence:
            while len(chain) > 1 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        chains.append(chain[:-1])
    return np.asarray(chains[0] + chains[1])


def outside_distance(point, polygon):
    if not len(polygon):
        return np.nan
    a, b = polygon, np.roll(polygon, -1, axis=0)
    edge, delta = b - a, point - a
    cross = edge[:, 0] * delta[:, 1] - edge[:, 1] * delta[:, 0]
    if len(polygon) >= 3 and (cross >= -1e-10).all():
        return 0.0
    alpha = np.clip((edge * delta).sum(-1) / np.maximum((edge * edge).sum(-1), 1e-20), 0, 1)
    return float(np.linalg.norm(point - (a + alpha[:, None] * edge), axis=-1).min())


def audit(bundle, poses, linear_velocity, angular_velocity, joint_velocity, output):
    model = mujoco.MjModel.from_xml_path(
        "deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    )
    data = mujoco.MjData(model)
    joints = np.array([model.joint(name).id for name in bundle.metadata["joint_names"]])
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    pelvis = model.body("pelvis").id
    mass = model.body_subtreemass[pelvis]
    feet = {model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")}
    sole = np.array(
        [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] in feet
            and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
            and model.geom_contype[i]
        ]
    )
    com, velocity, momentum, polygons = [], [], [], []
    for frame, pose in enumerate(poses):
        data.qpos[:7], data.qpos[qadr] = pose[:7], pose[7:]
        data.qvel[:3] = linear_velocity[frame]
        data.qvel[3:6] = quaternion_matrix(pose[3:7]).T @ angular_velocity[frame]
        data.qvel[vadr] = joint_velocity[frame]
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        com.append(data.subtree_com[pelvis].copy())
        velocity.append(data.subtree_linvel[pelvis].copy())
        momentum.append(data.subtree_angmom[pelvis].copy())
        near = data.geom_xpos[sole, 2] - model.geom_size[sole, 0] < 0.025
        polygons.append(hull(data.geom_xpos[sole[near], :2]))
    com, velocity, momentum = map(np.asarray, (com, velocity, momentum))
    force = mass * (np.gradient(velocity, 0.02, axis=0) - model.opt.gravity)
    moment_origin = np.gradient(momentum, 0.02, axis=0) + np.cross(com, force)
    valid = force[:, 2] > 0.1 * mass * 9.81
    zmp = np.full((len(com), 2), np.nan)
    zmp[valid, 0] = -moment_origin[valid, 1] / force[valid, 2]
    zmp[valid, 1] = moment_origin[valid, 0] / force[valid, 2]
    distance = np.array(
        [
            outside_distance(point, poly) if ok else np.nan
            for point, poly, ok in zip(zmp, polygons, valid)
        ]
    )
    # Generous sphere-radius allowance; this audit does not assume point feet.
    distance = np.maximum(distance - 0.005, 0)
    phases = {}
    for phase, (a, b) in bundle.metadata["sequence"]["segments"].items():
        if b >= len(poses):
            continue
        selected = distance[a : b + 1]
        finite = np.isfinite(selected)
        values = selected[finite]
        phases[phase] = dict(
            samples=len(selected),
            assessed_samples=int(finite.sum()),
            no_near_sole_fraction=float(np.mean([len(x) == 0 for x in polygons[a : b + 1]])),
            outside_2cm_fraction=float(np.mean(values > 0.02)) if len(values) else None,
            outside_5cm_fraction=float(np.mean(values > 0.05)) if len(values) else None,
            outside_rms_m=float(np.sqrt(np.mean(values**2))) if len(values) else None,
            outside_peak_m=float(values.max()) if len(values) else None,
        )
    worst = np.argsort(np.nan_to_num(distance, nan=-1))[-10:][::-1]
    report = dict(
        mass_kg=float(mass),
        sample_hz=50,
        phases=phases,
        worst=[
            dict(
                frame=int(i),
                time=float(i * 0.02),
                outside_m=float(distance[i]),
                zmp=zmp[i].tolist(),
                support_points=polygons[i].tolist(),
            )
            for i in worst
        ],
        limitation="50Hz momentum derivatives and generous near-floor support; diagnostics only, not proof of infeasibility or acceptance",
        sources=[
            "https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-subtreeangmom",
            "https://underactuated.csail.mit.edu/humanoids.html",
        ],
        goal_complete_claimed=False,
        hardware_ready=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "centroidal.npz",
        com=com,
        com_velocity=velocity,
        momentum=momentum,
        required_force=force,
        zmp=zmp,
        outside_distance=distance,
    )
    write_json(output / "report.json", report)
    print(output.name, json.dumps(phases), flush=True)
    return report


if __name__ == "__main__":
    square = hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    assert outside_distance(np.array([0.5, 0.5]), square) == 0
    assert abs(outside_distance(np.array([1.1, 0.5]), square) - 0.1) < 1e-10
    bundle = DeploymentBundle.load(PROJECT / "candidates/run002_000250")
    ref = bundle.reference
    audit(
        bundle,
        np.c_[ref["root_p"], ref["root_quat"], ref["q"]],
        ref["root_v"],
        ref["root_w"],
        ref["qd"],
        PROJECT / "evaluation/reference_support",
    )
    heldout = torch.load(
        PROJECT / "evaluation/contact_wm_holdout/physical_holdout.pt", weights_only=True
    )
    state = heldout["cases"]["nominal"]["states"].numpy()
    audit(
        bundle,
        np.c_[state[:, :7], state[:, 13:42]],
        state[:, 7:10],
        state[:, 10:13],
        state[:, 42:71],
        PROJECT / "evaluation/physical_support",
    )
