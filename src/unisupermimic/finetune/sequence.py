"""Prepare a continuous, explicitly versioned standing/dance/standing reference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle, _build_reference, _robot_xml_path
from unisupermimic.deploy.mujoco_sim import NativeSimulator
from unisupermimic.rl.algos.supermimic.geometry import (
    mat_to_quat,
    quat_slerp,
    quat_to_mat,
    so3_exp,
    so3_log,
)


def quintic(t, duration, p0, p1, v0=None, v1=None, a0=None, a1=None):
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    z = np.zeros_like(p0)
    v0 = z if v0 is None else np.asarray(v0)
    v1 = z if v1 is None else np.asarray(v1)
    a0 = z if a0 is None else np.asarray(a0)
    a1 = z if a1 is None else np.asarray(a1)
    T = float(duration)
    if not np.isfinite(T) or T <= 0:
        raise ValueError("duration must be positive")
    c0, c1, c2 = p0, v0, a0 / 2
    matrix = np.array(
        [[T**3, T**4, T**5], [3 * T * T, 4 * T**3, 5 * T**4], [6 * T, 12 * T * T, 20 * T**3]]
    )
    c3, c4, c5 = np.linalg.solve(
        matrix, np.stack([p1 - c0 - c1 * T - c2 * T * T, v1 - c1 - 2 * c2 * T, a1 - 2 * c2])
    )
    return sum(np.asarray(t)[:, None] ** i * c for i, c in enumerate((c0, c1, c2, c3, c4, c5)))


def _rotation_blend(R0, R1, t, T):
    phi = so3_log(torch.tensor(R0.T @ R1, dtype=torch.float64)).numpy()
    tangent = quintic(t, T, np.zeros(3), phi)
    return (torch.tensor(R0) @ so3_exp(torch.tensor(tangent))).numpy()


def _ease_dance(reference):
    n = len(reference["q"])
    edge = min(50, (n - 1) // 4)
    if edge < 1:
        raise ValueError("reference too short")
    E = edge * 0.02
    T = (n - 1) * 0.02
    D = T + 2 * E
    times = np.arange(n + 2 * edge) * 0.02
    u = times - E
    begin = times < 2 * E
    x = times[begin] / (2 * E)
    u[begin] = E * (2 * x**3 - x**4)
    end = times > T
    x = (D - times[end]) / (2 * E)
    u[end] = T - E * (2 * x**3 - x**4)
    indices = np.clip(u / 0.02, 0, n - 1)
    lo = np.minimum(np.floor(indices).astype(int), n - 2)
    alpha = indices - lo
    result = {
        k: reference[k][lo] * (1 - alpha[:, None]) + reference[k][lo + 1] * alpha[:, None]
        for k in ("q", "root_p")
    }
    quat = torch.tensor(reference["root_quat"], dtype=torch.float64)
    result["root_R"] = quat_to_mat(quat_slerp(quat[lo], quat[lo + 1], torch.tensor(alpha))).numpy()
    # Copy the untouched core exactly, independent of interpolation roundoff.
    core = slice(2 * edge, n - 1 + 1)
    for key in result:
        result[key][core] = reference[key][edge : n - edge]
    return result, edge


def build_sequence(
    source,
    scene,
    output,
    *,
    head_seconds=2.0,
    entry_seconds=3.0,
    exit_seconds=3.0,
    tail_seconds=5.0,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    counts = []
    for value in (head_seconds, entry_seconds, exit_seconds, tail_seconds):
        if not np.isfinite(value) or value < 0.1 or abs(value / 0.02 - round(value / 0.02)) > 1e-6:
            raise ValueError("phase durations must be positive multiples of20ms")
        counts.append(round(value / 0.02))
    h, e, x, tail = counts
    dance, edge = _ease_dance(source.reference)
    sim = NativeSimulator(source, scene)
    stance = np.array(source.metadata["default_joint_positions"])

    def standing(index):
        R = dance["root_R"][index]
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        sim.reset("reference_zero_velocity")
        sim.data.qpos[:3] = dance["root_p"][index]
        sim.data.qpos[3:7] = mat_to_quat(torch.tensor(rotation)).numpy()
        sim.data.qpos[sim.qadr] = stance
        mujoco.mj_forward(sim.model, sim.data)
        feet = [
            i
            for i in range(sim.model.ngeom)
            if sim.model.body(sim.model.geom_bodyid[i]).name
            in ("left_ankle_roll_link", "right_ankle_roll_link")
            and sim.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        if not feet:
            raise ValueError("standing preparation requires spherical foot contacts")
        bottom = min(sim.data.geom_xpos[i, 2] - sim.model.geom_size[i, 0] for i in feet)
        p = sim.data.qpos[:3].copy()
        p[2] -= bottom
        return p, rotation

    p0, R0 = standing(0)
    p1, R1 = standing(-1)
    start = h + e
    end = start + len(dance["q"]) - 1
    exit_end = end + x
    q = np.concatenate(
        [
            np.repeat(stance[None], h, axis=0),
            quintic(np.arange(e) * 0.02, e * 0.02, stance, dance["q"][0]),
            dance["q"],
            quintic(np.arange(1, x + 1) * 0.02, x * 0.02, dance["q"][-1], stance),
            np.repeat(stance[None], tail, axis=0),
        ]
    )
    p = np.concatenate(
        [
            np.repeat(p0[None], h, axis=0),
            quintic(np.arange(e) * 0.02, e * 0.02, p0, dance["root_p"][0]),
            dance["root_p"],
            quintic(np.arange(1, x + 1) * 0.02, x * 0.02, dance["root_p"][-1], p1),
            np.repeat(p1[None], tail, axis=0),
        ]
    )
    R = np.concatenate(
        [
            np.repeat(R0[None], h, axis=0),
            _rotation_blend(R0, dance["root_R"][0], np.arange(e) * 0.02, e * 0.02),
            dance["root_R"],
            _rotation_blend(dance["root_R"][-1], R1, np.arange(1, x + 1) * 0.02, x * 0.02),
            np.repeat(R1[None], tail, axis=0),
        ]
    )
    quat = mat_to_quat(torch.tensor(R)).numpy()
    raw = np.concatenate([p, quat[:, [1, 2, 3, 0]], q], axis=-1)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".sequence-", dir=output.parent))
    try:
        csv = staged / "sequence.csv"
        np.savetxt(csv, raw, delimiter=",", fmt="%.10g")

        def sha(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        sidecar = csv.with_suffix(".json")
        sidecar.write_text(
            json.dumps(dict(fps=50, frames=len(q), csv_sha256=sha(csv), quaternion_order="xyzw"))
        )
        ref = _build_reference(csv, Path(scene), 10)
        np.savez(staged / "reference.npz", **ref)
        shutil.copyfile(source.policy_path, staged / "policy.ts")
        meta = copy.deepcopy(source.metadata)
        meta.update(
            motion_csv_sha256=sha(csv),
            source_motion_metadata_sha256=sha(sidecar),
            candidate_status="unvalidated",
            accepted_for_delivery=False,
            hardware_ready=False,
        )
        meta["sequence"] = dict(
            generation_scene=str(Path(scene).resolve()),
            generation_scene_sha256=sha(scene),
            generation_robot_xml_sha256=sha(_robot_xml_path(Path(scene))),
            core_numeric_tolerance_rad=2e-5,
            segments=dict(
                standing=[0, h],
                entry=[h, start],
                dance=[start, end],
                exit=[end, exit_end],
                settle=[exit_end, len(q) - 1],
            ),
            initial_velocity="zero",
            source_reference_sha256=sha(source.directory / "reference.npz"),
            edge_source_frames=edge,
            source_frames=len(source.reference["q"]),
            edge_timewarp="C2 rate0->1->0;core copied exactly",
            core_output=[start + 2 * edge, start + len(source.reference["q"]) - 1],
            core_source=[edge, len(source.reference["q"]) - edge - 1],
        )
        manifest = dict(
            schema_version=1,
            files={name: sha(staged / name) for name in ("policy.ts", "reference.npz")},
            metadata=meta,
        )
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        DeploymentBundle.load(staged)
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    print(build_sequence(DeploymentBundle.load(a.source_bundle), a.scene, a.output))
