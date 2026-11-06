"""Mechanical publication of a prepared reference motion bundle."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle, _build_reference
from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_reference(
    source: DeploymentBundle,
    scene: Path,
    output: Path,
    qpos: Any,
    metadata: dict[str, Any],
) -> Path:
    """Write an unvalidated motion bundle atomically from raw wxyz qpos."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    poses = np.asarray(qpos, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 36:
        raise ValueError("qpos must be a finite Nx36 wxyz pose array")
    if poses.shape[0] < 2:
        raise ValueError("qpos must contain at least two frames")
    if not np.isfinite(poses).all():
        raise ValueError("qpos must be finite")
    lower = np.asarray(source.metadata["joint_lower"], dtype=float)
    upper = np.asarray(source.metadata["joint_upper"], dtype=float)
    if (poses[:, 7:] < lower - 1e-5).any() or (poses[:, 7:] > upper + 1e-5).any():
        raise ValueError("qpos violates joint limits")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        csv = staged / "sequence.csv"
        columns = [0, 1, 2, 4, 5, 6, 3, *range(7, 36)]
        np.savetxt(csv, poses[:, columns], delimiter=",", fmt="%.10g")
        sidecar = csv.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                dict(
                    fps=50,
                    frames=len(poses),
                    csv_sha256=_sha(csv),
                    quaternion_order="xyzw",
                )
            )
        )
        goal_frames = source.metadata.get("goal_frames", 10)
        reference = _build_reference(csv, Path(scene), goal_frames)
        if len(reference["q"]) != len(poses):
            raise ValueError("reference resampling changed frame count")
        expected_rotation = quat_to_mat(torch.as_tensor(poses[:, 3:7], dtype=torch.float64)).numpy()
        if (
            not np.allclose(reference["q"], poses[:, 7:], atol=2e-5, rtol=0)
            or not np.allclose(reference["root_p"], poses[:, :3], atol=2e-5, rtol=0)
            or not np.allclose(reference["root_R"], expected_rotation, atol=2e-5, rtol=0)
        ):
            raise ValueError("reference rebuilding changed prepared poses")
        if (reference["q"] < lower - 1e-5).any() or (reference["q"] > upper + 1e-5).any():
            raise ValueError("rebuilt reference violates joint limits")
        np.savez(staged / "reference.npz", **reference)
        shutil.copyfile(source.policy_path, staged / "policy.ts")
        meta = copy.deepcopy(metadata)
        meta.update(
            motion_csv_sha256=_sha(csv),
            source_motion_metadata_sha256=_sha(sidecar),
            candidate_status="unvalidated",
            accepted_for_delivery=False,
            hardware_ready=False,
        )
        manifest = dict(
            schema_version=1,
            files={name: _sha(staged / name) for name in ("policy.ts", "reference.npz")},
            metadata=meta,
        )
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        DeploymentBundle.load(staged)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staged.rename(output)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return Path(output)
