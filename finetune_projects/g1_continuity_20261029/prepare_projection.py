"""Prepare bounded training-reference adaptations; never edit physical trajectories."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.rl.algos.supermimic.geometry import quat_slerp
from unisupermimic.tasks.motion_tracking.g1.supermimic.preprocess import (
    continuous_quaternions,
    smooth_values,
)

PROJECT = Path("finetune_projects/g1_continuity_20261029")
PHYSICAL = PROJECT / "evaluation/run002_000250/trajectory.npz"
ANCHOR = Path("finetune_projects/g1_continuous_20261029/data/sequence_v2/reference.npz")
SCENE = Path("deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")


def floor_heights(poses):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    feet = {model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")}
    geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] in feet
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
        and model.geom_contype[i]
    ]
    heights = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        heights.append(np.min(data.geom_xpos[geoms, 2] - model.geom_size[geoms, 0]))
    return np.asarray(heights)


def main():
    output = PROJECT / "data/projection_prepared_v3"
    output.mkdir(parents=True, exist_ok=False)
    bundle = DeploymentBundle.load(PROJECT / "candidates/run002_000250")
    ref = bundle.reference
    assert (
        hashlib.sha256(ANCHOR.read_bytes()).hexdigest()
        == "7773170ed06cfcc43e5830e79fe7786c70b696a0709bd8aa33ab7ed7ec033daa"
    )
    old = np.c_[ref["root_p"], ref["root_quat"], ref["q"]].astype(float)
    physical = np.load(PHYSICAL)["qpos"]
    assert old.shape == physical.shape == (1799, 36)
    low = physical.astype(float).copy()
    lower = np.asarray(bundle.metadata["joint_lower"])
    upper = np.asarray(bundle.metadata["joint_upper"])
    clipped = np.clip(low[:, 7:], lower, upper)
    source_clipped_values = int(np.count_nonzero(clipped[:, :15] != low[:, 7:22]))
    # Executed soft-limit overshoot is not an admissible reference target.
    # Bound before the positive Gaussian filter, preserving smooth convexity.
    low[:, 7:] = clipped
    low[:, :3] = smooth_values(low[:, :3], 1.5)
    low[:, 7:] = smooth_values(low[:, 7:], 1.5)
    low[:, 3:7] = continuous_quaternions(smooth_values(continuous_quaternions(low[:, 3:7]), 1.5))
    segments = bundle.metadata["sequence"]["segments"]
    envelope = np.zeros(len(old))
    a, b = segments["entry"]
    u = np.linspace(0, 1, b - a + 1)
    smoothstep = 10 * u**3 - 15 * u**4 + 6 * u**5
    envelope[a : b + 1] = smoothstep
    a, b = segments["dance"]
    envelope[a : b + 1] = 1
    a, b = segments["exit"]
    u = np.linspace(0, 1, b - a + 1)
    envelope[a : b + 1] = 1 - (10 * u**3 - 15 * u**4 + 6 * u**5)
    # Ground clearance has its own short C2 ramp. Reusing the three-second
    # pose blend inflated a 5mm late-exit correction into a 38mm global lift.
    ground_envelope = np.zeros(len(old))
    start, end = segments["entry"][0], segments["exit"][1]
    ground_envelope[start : end + 1] = 1
    u = np.linspace(0, 1, 26)
    ramp = 10 * u**3 - 15 * u**4 + 6 * u**5
    ground_envelope[start : start + 26] = ramp
    ground_envelope[end - 25 : end + 1] = 1 - ramp
    a, b = segments["dance"]
    main_joints = [0, 3, 6, 9]
    for alpha in (0.5, 0.75):
        w = alpha * envelope
        poses = old.copy()
        poses[:, :3] += w[:, None] * (low[:, :3] - old[:, :3])
        poses[:, 7:22] += w[:, None] * (low[:, 7:22] - old[:, 7:22])
        poses[:, 3:7] = quat_slerp(
            torch.tensor(old[:, 3:7]), torch.tensor(low[:, 3:7]), torch.tensor(w)
        ).numpy()
        heights = floor_heights(poses)
        selected = ground_envelope > 1e-4
        lift = max(0.0, float(np.max((-heights[selected] - 0.002) / ground_envelope[selected])))
        poses[:, 2] += lift * ground_envelope
        final_heights = floor_heights(poses)
        original_q, q = old[a : b + 1, 7:], poses[a : b + 1, 7:]
        amplitude = {
            bundle.metadata["joint_names"][j]: float(np.ptp(q[:, j]) / np.ptp(original_q[:, j]))
            for j in main_joints
        }
        correlation = {
            bundle.metadata["joint_names"][j]: float(np.corrcoef(q[:, j], original_q[:, j])[0, 1])
            for j in main_joints
        }
        rms = float(np.sqrt(np.mean((q - original_q) ** 2)))
        arm_error = float(np.abs(poses[:, 22:] - old[:, 22:]).max())
        limit_violation = max(
            0.0, float(np.maximum(lower - poses[:, 7:], poses[:, 7:] - upper).max())
        )
        eligible = bool(
            lift <= 0.02
            and final_heights.min() >= -0.00201
            and rms < 0.10
            and arm_error < 2e-5
            and min(amplitude.values()) >= 0.85
            and min(correlation.values()) > 0.95
            and limit_violation <= 1e-5
        )
        profile = dict(
            method="bounded_low_frequency_physical_projection",
            alpha=alpha,
            sigma_frames=1.5,
            sample_hz=50,
            changed_joint_indices=list(range(15)),
            phase_segments=segments,
            mixed_phases=["entry", "dance", "exit"],
            source_physical_npz=str(PHYSICAL),
            source_physical_sha256=hashlib.sha256(PHYSICAL.read_bytes()).hexdigest(),
            style_anchor=str(ANCHOR),
            style_anchor_sha256=hashlib.sha256(ANCHOR.read_bytes()).hexdigest(),
            floor_lift_peak_m=lift,
            floor_lift_ramp_seconds=0.5,
            min_foot_height_m=float(final_heights.min()),
            reference_joint_rmse_vs_anchor=rms,
            arm_max_error=arm_error,
            max_joint_limit_violation_rad=limit_violation,
            physical_source_lower_body_values_clipped=source_clipped_values,
            main_joint_amplitude_ratios=amplitude,
            main_joint_correlations=correlation,
            preparation_eligible=eligible,
            note="Reference adaptation only; actual rollout must also pass original style-anchor fidelity",
            hardware_ready=False,
            goal_complete_claimed=False,
        )
        name = f"projection_{round(alpha * 100):03d}"
        np.savez_compressed(output / f"{name}.npz", qpos=poses)
        profile["prepared_array_sha256"] = hashlib.sha256(
            (output / f"{name}.npz").read_bytes()
        ).hexdigest()
        (output / f"{name}.json").write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
        print(name, json.dumps(profile), flush=True)


if __name__ == "__main__":
    main()
