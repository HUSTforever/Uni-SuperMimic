"""Package an explicitly prepared reference as an unvalidated simulation bundle."""

import argparse
import copy
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from unisupermimic.deploy.bundle import DeploymentBundle, _build_reference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(source, array, provenance, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    extra = json.loads(Path(provenance).read_text())
    if extra.get("preparation_eligible") is not True:
        raise ValueError("reference preparation did not pass its bounds")
    blob = Path(array).read_bytes()
    if extra.get("prepared_array_sha256") != hashlib.sha256(blob).hexdigest():
        raise ValueError("prepared array hash does not match checked provenance")
    with np.load(io.BytesIO(blob), allow_pickle=False) as values:
        poses = np.asarray(values["qpos"], dtype=float)
    if poses.shape != (len(source.reference["q"]), 36) or not np.isfinite(poses).all():
        raise ValueError("finite source-length Nx36 wxyz poses required")
    lower, upper = (
        np.asarray(source.metadata["joint_lower"]),
        np.asarray(source.metadata["joint_upper"]),
    )
    if (poses[:, 7:] < lower - 1e-5).any() or (poses[:, 7:] > upper + 1e-5).any():
        raise ValueError("prepared reference violates joint limits")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        csv = staged / "sequence.csv"
        np.savetxt(csv, poses[:, [0, 1, 2, 4, 5, 6, 3, *range(7, 36)]], delimiter=",", fmt="%.10g")
        sidecar = csv.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                dict(fps=50, frames=len(poses), csv_sha256=sha(csv), quaternion_order="xyzw")
            )
        )
        scene = Path("deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
        reference = _build_reference(csv, scene, 10)
        if len(reference["q"]) != len(poses):
            raise ValueError("reference resampling changed frame count")
        if (reference["q"] < lower - 1e-5).any() or (reference["q"] > upper + 1e-5).any():
            raise ValueError("rebuilt reference violates joint limits")
        np.savez(staged / "reference.npz", **reference)
        shutil.copyfile(source.policy_path, staged / "policy.ts")
        manifest = copy.deepcopy(json.loads((source.directory / "manifest.json").read_text()))
        manifest["files"] = {name: sha(staged / name) for name in ("policy.ts", "reference.npz")}
        manifest["metadata"].update(
            motion_csv_sha256=sha(csv),
            source_motion_metadata_sha256=sha(sidecar),
            reference_projection=extra,
            accepted_for_delivery=False,
            hardware_ready=False,
            candidate_status="unvalidated",
        )
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        DeploymentBundle.load(staged)
        staged.rename(output)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--array", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(pack(DeploymentBundle.load(a.source), a.array, a.provenance, a.output))
