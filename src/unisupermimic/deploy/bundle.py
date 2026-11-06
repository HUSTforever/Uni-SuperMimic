"""Immutable offline bundles for the 29-DoF SuperMimic deployment policy.

The loader deliberately depends only on the bundle itself: it never imports the
training environment, MDP, robot model or TorchScript runtime.  TorchScript is
validated at export time and loaded later by the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
CONTROL_DT = 0.02
OBS_DIM = 99
HIST = 10
GOAL_FRAME_DIM = 73
GOAL_FRAMES = 10
GOAL_DIM = GOAL_FRAME_DIM * GOAL_FRAMES
ACTION_DIM = 29
ACTION_TARGET_FORMULA = "default_q + context_offset + action_scale * raw_action"
OUTPUT_SEMANTICS = "raw_action"

POLICY_NAME = "policy.ts"
REFERENCE_NAME = "reference.npz"
MANIFEST_NAME = "manifest.json"
BUNDLE_FILES = (POLICY_NAME, REFERENCE_NAME)
REFERENCE_KEYS = (
    "q",
    "qd",
    "root_p",
    "root_R",
    "root_v",
    "root_w",
    "root_quat",
    "goals",
)

# The fixed G1 natural joint order used by the training policy.  Keeping this
# constant here is intentional: loading a bundle must not import training code.
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]  # fmt: skip

_VECTOR_KEYS = (
    "kp",
    "kd",
    "effort_limits",
    "velocity_limits",
    "joint_lower",
    "joint_upper",
    "default_joint_positions",
    "action_scales",
    "context_offset",
)
_POSITIVE_VECTOR_KEYS = (
    "kp",
    "kd",
    "effort_limits",
    "velocity_limits",
    "action_scales",
)
_HASH_FIELDS = (
    "source_checkpoint_sha256",
    "motion_csv_sha256",
    "source_motion_metadata_sha256",
    "source_scene_sha256",
    "source_robot_xml_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_matches(value: Any, expected: str) -> bool:
    return isinstance(value, str) and value.lower() == expected.lower()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")


def _require_integer(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metadata {label} must be {expected}, got {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"metadata {label} must be {expected}, got {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or int(numeric) != expected:
        raise ValueError(f"metadata {label} must be {expected}, got {value!r}")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _metadata_vector(metadata: dict[str, Any], key: str, *, positive: bool) -> np.ndarray:
    value = metadata.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != ACTION_DIM:
        raise ValueError(f"metadata {key} must be a {ACTION_DIM}-element vector")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"metadata {key} must contain only numbers")
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"metadata {key} must contain only finite values") from exc
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"metadata {key} must have shape {(ACTION_DIM,)}")
    if not np.isfinite(vector).all():
        raise ValueError(f"metadata {key} must contain only finite values")
    if positive and np.any(vector <= 0):
        raise ValueError(f"metadata {key} must contain only positive values")
    return vector


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("manifest metadata must be a JSON object")

    schema_version = metadata.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        raise ValueError(f"metadata schema_version must be {SCHEMA_VERSION}")

    if metadata.get("output_semantics") != OUTPUT_SEMANTICS:
        raise ValueError("metadata output_semantics must be raw_action")
    if metadata.get("action_target_formula") != ACTION_TARGET_FORMULA:
        raise ValueError("metadata action_target_formula is invalid")

    names = metadata.get("joint_names")
    if names != G1_JOINT_NAMES or len(set(names)) != ACTION_DIM:
        raise ValueError("metadata joint_names must be the fixed unique G1 natural 29 joints")

    _require_integer(metadata.get("obs_dim"), OBS_DIM, "obs_dim")
    _require_integer(metadata.get("hist"), HIST, "hist")
    _require_integer(metadata.get("goal_frames"), GOAL_FRAMES, "goal_frames")
    _require_integer(metadata.get("goal_dim"), GOAL_DIM, "goal_dim")
    _require_integer(metadata.get("action_dim"), ACTION_DIM, "action_dim")

    control_dt = _finite_float(metadata.get("control_dt"))
    if control_dt is None or abs(control_dt - CONTROL_DT) > 1e-12:
        raise ValueError(f"metadata control_dt must be {CONTROL_DT}")

    if "action_smoothing" in metadata and _finite_float(metadata["action_smoothing"]) is None:
        raise ValueError("metadata action_smoothing must be a finite number")

    vectors = {
        key: _metadata_vector(metadata, key, positive=key in _POSITIVE_VECTOR_KEYS)
        for key in _VECTOR_KEYS
    }
    if np.any(vectors["joint_lower"] >= vectors["joint_upper"]):
        raise ValueError("metadata joint_lower must be strictly below joint_upper")
    if not np.array_equal(vectors["context_offset"], np.zeros(ACTION_DIM)):
        raise ValueError("metadata context_offset must be 29 zeros")

    if metadata.get("provenance_checked") is not True:
        raise ValueError("metadata provenance_checked must be true")

    source_scene = metadata.get("source_scene")
    if not isinstance(source_scene, str):
        raise ValueError("metadata source_scene must be an absolute path")
    try:
        absolute_scene = Path(source_scene).is_absolute()
    except (OSError, ValueError) as exc:
        raise ValueError("metadata source_scene must be an absolute path") from exc
    if not absolute_scene:
        raise ValueError("metadata source_scene must be an absolute path")
    for field in _HASH_FIELDS:
        if not _valid_sha256(metadata.get(field)):
            raise ValueError(f"metadata {field} must be a complete SHA256 digest")


def _quat_to_mat(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (quaternion[..., index] for index in range(4))
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _validate_reference(reference: dict[str, np.ndarray]) -> int:
    if set(reference) != set(REFERENCE_KEYS):
        raise ValueError(f"reference must contain exactly {', '.join(REFERENCE_KEYS)}")
    for key in REFERENCE_KEYS:
        value = reference[key]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"reference {key} must be a numpy array")
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"reference {key} must use a floating-point dtype")
        if not np.isfinite(value).all():
            raise ValueError(f"reference {key} must contain only finite values")

    q = reference["q"]
    if q.ndim != 2 or q.shape[1] != ACTION_DIM:
        raise ValueError(f"reference q must have shape (N, {ACTION_DIM}), got {q.shape}")
    num_frames = int(q.shape[0])
    if num_frames < 2:
        raise ValueError("reference must contain at least two frames")

    expected_shapes = {
        "qd": (num_frames, ACTION_DIM),
        "root_p": (num_frames, 3),
        "root_R": (num_frames, 3, 3),
        "root_v": (num_frames, 3),
        "root_w": (num_frames, 3),
        "root_quat": (num_frames, 4),
        "goals": (num_frames, GOAL_DIM),
    }
    for key, shape in expected_shapes.items():
        if reference[key].shape != shape:
            raise ValueError(f"reference {key} must have shape {shape}, got {reference[key].shape}")

    rotation = reference["root_R"].astype(np.float64)
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), rotation.shape)
    orthonormal = np.einsum("nij,nkj->nik", rotation, rotation)
    if not np.allclose(orthonormal, identity, rtol=0, atol=1e-4):
        raise ValueError("reference root_R must contain orthonormal rotations")
    determinant = np.linalg.det(rotation)
    if np.any(determinant <= 0) or not np.allclose(determinant, 1.0, rtol=0, atol=1e-4):
        raise ValueError("reference root_R must have determinant +1 for every frame")

    quaternion = reference["root_quat"].astype(np.float64)
    norms = np.linalg.norm(quaternion, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0, atol=1e-4):
        raise ValueError("reference root_quat must contain unit quaternions")
    if not np.allclose(_quat_to_mat(quaternion), rotation, rtol=0, atol=1e-4):
        raise ValueError("reference root_quat must be consistent with root_R")

    return num_frames


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint could not be loaded with weights_only=True") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if not isinstance(checkpoint.get("pi"), dict):
        raise ValueError("checkpoint must contain a pi state dict")
    return checkpoint


def _validate_policy_weights(policy_path: Path, checkpoint: dict[str, Any]) -> None:
    import torch
    from demo.export_policy import build_policy

    checkpoint_state = checkpoint["pi"]

    try:
        module = torch.jit.load(str(policy_path), map_location="cpu")
        policy_state = module.state_dict()
    except Exception as exc:
        raise ValueError("policy TorchScript could not be loaded for validation") from exc

    if set(policy_state) != set(checkpoint_state):
        raise ValueError("policy TorchScript weights do not match checkpoint keys")
    for key, expected in checkpoint_state.items():
        actual = policy_state[key]
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            raise ValueError(f"policy or checkpoint weight {key!r} is not a tensor")
        actual = actual.detach().cpu()
        expected = expected.detach().cpu()
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"policy or checkpoint weight {key!r} has a mismatched shape/dtype")
        if not torch.equal(actual, expected):
            raise ValueError(f"policy TorchScript weight {key!r} does not match the checkpoint")
    # Traced constants (notably action_smoothing) are absent from state_dict.
    # Validate behavior as well as weights, including nonzero previous actions.
    canonical = build_policy(checkpoint).eval()
    args = checkpoint["args"]
    generator = torch.Generator().manual_seed(20261029)
    history = torch.randn(4, int(args["hist"]), 99, generator=generator)
    history[:, :, 64:93] = torch.linspace(-1, 1, 4*int(args["hist"])*29).reshape(
        4, int(args["hist"]), 29)
    goals = torch.randn(4, 73*int(args["goal_frames"]), generator=generator)
    with torch.inference_mode():
        actual, expected = module(history, goals), canonical(history, goals)
    if (actual.shape != expected.shape or not bool(torch.isfinite(actual).all())
            or not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)):
        raise ValueError("policy forward equivalence with the canonical checkpoint failed")


def _robot_xml_path(scene: Path) -> Path:
    try:
        root = ET.fromstring(scene.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ET.ParseError) as exc:
        raise ValueError(f"scene XML is not readable: {scene}") from exc
    for element in root.iter("include"):
        include = element.get("file")
        if include:
            path = Path(include)
            if not path.is_absolute():
                path = scene.parent / path
            path = path.resolve()
            if not path.is_file():
                raise ValueError(f"scene includes a missing robot XML: {path}")
            return path
    return scene


def _read_joint_limits(scene: Path, joint_names: list[str]) -> tuple[list[float], list[float]]:
    import mujoco

    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
    except Exception as exc:
        raise ValueError(f"MuJoCo scene could not be loaded: {scene}") from exc
    lower: list[float] = []
    upper: list[float] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo scene is missing joint {name!r}")
        lower.append(float(model.jnt_range[joint_id][0]))
        upper.append(float(model.jnt_range[joint_id][1]))
    return lower, upper


def _build_reference(
    motion_csv: Path,
    scene: Path,
    goal_frames: int,
) -> dict[str, np.ndarray]:
    import torch

    from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat
    from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
    from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib

    motion = MotionLib([str(motion_csv)], G1Kinematics(str(scene), "cpu"), "cpu")
    num_frames = int(motion.num_frames)
    if num_frames < 2:
        raise ValueError("motion must resample to at least two reference frames")
    frames = torch.arange(num_frames, dtype=torch.long)
    with torch.no_grad():
        goals = motion.goal(frames, goal_frames).detach().cpu().numpy()
        root_quat = mat_to_quat(motion.root_R).detach().cpu().numpy()
        reference = {
            "q": motion.q.detach().cpu().numpy(),
            "qd": motion.qd.detach().cpu().numpy(),
            "root_p": motion.root_p.detach().cpu().numpy(),
            "root_R": motion.root_R.detach().cpu().numpy(),
            "root_v": motion.root_v.detach().cpu().numpy(),
            "root_w": motion.root_w.detach().cpu().numpy(),
            "root_quat": root_quat,
            "goals": goals,
        }
    return {
        key: np.ascontiguousarray(value, dtype=np.float32) for key, value in reference.items()
    }


def _build_metadata(
    base: dict[str, Any],
    checkpoint: dict[str, Any],
    motion_csv: Path,
    checkpoint_path: Path,
    scene: Path,
) -> dict[str, Any]:
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
        G1_JOINT_NAMES as ROBOT_JOINT_NAMES,
    )
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
        _kd,
        _kp,
        motor_of,
    )

    joint_names = list(ROBOT_JOINT_NAMES)
    if list(base.get("joint_names", [])) != joint_names:
        raise ValueError("policy sidecar joint_names do not match the fixed G1 natural order")

    lower, upper = _read_joint_limits(scene, joint_names)
    robot_xml = _robot_xml_path(scene)
    metadata = dict(base)
    metadata.update(
        {
            "kp": [float(_kp(motor_of(name)[2])) for name in joint_names],
            "kd": [float(_kd(motor_of(name)[2])) for name in joint_names],
            "effort_limits": [float(motor_of(name)[0]) for name in joint_names],
            "velocity_limits": [float(motor_of(name)[1]) for name in joint_names],
            "joint_lower": lower,
            "joint_upper": upper,
            "context_offset": [0.0] * ACTION_DIM,
            "provenance_checked": True,
            "source_checkpoint_sha256": _sha256(checkpoint_path),
            "motion_csv_sha256": _sha256(motion_csv),
            "source_motion_metadata_sha256": _sha256(motion_csv.with_suffix(".json")),
            "source_scene": str(scene),
            "source_scene_sha256": _sha256(scene),
            "source_robot_xml_sha256": _sha256(robot_xml),
        }
    )
    # Keep the checkpoint argument contract explicit in the artifact.
    args = checkpoint.get("args", {})
    if isinstance(args, dict) and "action_smoothing" in args and "action_smoothing" not in metadata:
        metadata["action_smoothing"] = float(args["action_smoothing"])
    return metadata


def _validate_acceptance(
    report: dict[str, Any],
    motion_csv_sha256: str,
    checkpoint_sha256: str,
) -> None:
    if report.get("accepted_for_delivery") is not True:
        raise ValueError("acceptance report accepted_for_delivery must be true")
    if not _hash_matches(report.get("input_motion_sha256"), motion_csv_sha256):
        raise ValueError("acceptance report input_motion_sha256 does not match motion CSV")
    after_training = report.get("after_training")
    if not isinstance(after_training, dict) or not _hash_matches(
        after_training.get("checkpoint_sha256"), checkpoint_sha256
    ):
        raise ValueError("acceptance report after_training checkpoint_sha256 does not match checkpoint")


def export_bundle(
    *,
    policy_path: Path,
    motion_csv: Path,
    checkpoint_path: Path,
    acceptance_report: Path,
    scene: Path,
    output_dir: Path,
) -> Path:
    """Validate inputs and atomically publish an immutable deployment bundle."""
    output_dir = Path(output_dir)
    policy_path = Path(policy_path)
    motion_csv = Path(motion_csv)
    checkpoint_path = Path(checkpoint_path)
    acceptance_report = Path(acceptance_report)
    scene = Path(scene)

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    _require_file(policy_path, "policy")
    _require_file(motion_csv, "motion CSV")
    _require_file(checkpoint_path, "checkpoint")
    _require_file(acceptance_report, "acceptance report")
    _require_file(scene, "scene")

    policy_sha256 = _sha256(policy_path)
    motion_sha256 = _sha256(motion_csv)
    checkpoint_sha256 = _sha256(checkpoint_path)

    report = _load_json_object(acceptance_report, "acceptance report")
    _validate_acceptance(report, motion_sha256, checkpoint_sha256)
    # The accepted evaluation retains its own CSV/JSON copy. CSV hashing alone
    # does not bind fps; compare both artifacts with that independent record.
    accepted_motions = report["after_training"].get("motions", [])
    accepted = next((item for item in accepted_motions
                     if isinstance(item, dict) and item.get("sha256") == motion_sha256), None)
    if accepted is None or not isinstance(accepted.get("path"), str):
        raise ValueError("acceptance report must identify its evaluated reference metadata")
    accepted_csv = Path(accepted["path"])
    if not accepted_csv.is_absolute():
        accepted_csv = acceptance_report.parent / accepted_csv
    _require_file(accepted_csv, "accepted reference CSV")
    metadata_path, accepted_metadata = motion_csv.with_suffix(".json"), accepted_csv.with_suffix(".json")
    _require_file(metadata_path, "motion metadata")
    _require_file(accepted_metadata, "accepted reference metadata")
    if _sha256(accepted_csv) != motion_sha256 or _sha256(metadata_path) != _sha256(accepted_metadata):
        raise ValueError("motion/reference metadata differs from the accepted evaluation")

    checkpoint = _load_checkpoint(checkpoint_path)
    sidecar_path = policy_path.with_suffix(policy_path.suffix + ".json")
    sidecar = _load_json_object(sidecar_path, "policy sidecar")

    from demo.export_policy import export_metadata

    expected_metadata = export_metadata(checkpoint)
    if sidecar != expected_metadata:
        raise ValueError("policy sidecar metadata does not match the checkpoint metadata")

    _validate_policy_weights(policy_path, checkpoint)

    metadata = _build_metadata(
        sidecar,
        checkpoint,
        motion_csv,
        checkpoint_path,
        scene.resolve(),
    )
    _validate_metadata(metadata)
    goal_frames = int(metadata["goal_frames"])
    reference = _build_reference(motion_csv, scene, goal_frames)
    _validate_reference(reference)
    verified_frames = report.get("verification", {}).get("full_clip_torchscript_frames")
    if verified_frames is None or len(reference["q"])-1 != verified_frames:
        raise ValueError("reference frame count differs from the accepted full-clip verification")

    parent = output_dir.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(parent)))
    published = False
    try:
        shutil.copyfile(policy_path, staged / POLICY_NAME)
        if _sha256(staged / POLICY_NAME) != policy_sha256:
            raise ValueError("copied policy.ts SHA256 hash changed during export")
        np.savez(staged / REFERENCE_NAME, **reference)
        reference_sha256 = _sha256(staged / REFERENCE_NAME)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "files": {
                POLICY_NAME: policy_sha256,
                REFERENCE_NAME: reference_sha256,
            },
            "metadata": metadata,
        }
        (staged / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        os.rename(staged, output_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staged, ignore_errors=True)

    return output_dir


class DeploymentBundle:
    """A validated, immutable policy/reference bundle on disk."""

    def __init__(
        self,
        *,
        directory: Path,
        policy_path: Path,
        metadata: dict[str, Any],
        reference: dict[str, np.ndarray],
    ):
        self.directory = Path(directory)
        self.policy_path = Path(policy_path)
        self.metadata = metadata
        self.reference = reference

    @classmethod
    def load(cls, directory: Path) -> DeploymentBundle:
        """Load and validate a bundle without importing training code or TorchScript."""
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"bundle directory does not exist or is not a directory: {directory}")

        manifest = _load_json_object(directory / MANIFEST_NAME, "bundle manifest")
        schema_version = manifest.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
            raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION}")

        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != set(BUNDLE_FILES):
            raise ValueError(
                f"manifest files must contain exactly {', '.join(BUNDLE_FILES)}"
            )
        for name in BUNDLE_FILES:
            if not _valid_sha256(files.get(name)):
                raise ValueError(f"manifest files[{name!r}] must be a complete SHA256 digest")

        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("manifest metadata must be a JSON object")
        _validate_metadata(metadata)

        for name in BUNDLE_FILES:
            path = directory / name
            if not path.is_file():
                raise ValueError(f"bundle file is missing: {path}")
            actual = _sha256(path)
            if actual != files[name].lower():
                raise ValueError(f"SHA256 hash mismatch for {name}: expected {files[name]}, got {actual}")

        try:
            with np.load(directory / REFERENCE_NAME, allow_pickle=False) as data:
                if set(data.files) != set(REFERENCE_KEYS):
                    raise ValueError(
                        f"reference.npz must contain exactly {', '.join(REFERENCE_KEYS)}"
                    )
                raw_reference = {key: np.asarray(data[key]) for key in REFERENCE_KEYS}
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("reference.npz could not be loaded with allow_pickle=False") from exc

        reference: dict[str, np.ndarray] = {}
        for key, value in raw_reference.items():
            if not np.issubdtype(value.dtype, np.floating):
                raise ValueError(f"reference {key} must use a floating-point dtype")
            reference[key] = np.ascontiguousarray(value, dtype=np.float32)
        _validate_reference(reference)

        return cls(
            directory=directory,
            policy_path=directory / POLICY_NAME,
            metadata=metadata,
            reference=reference,
        )


def main() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = export_bundle(
        policy_path=args.policy,
        motion_csv=args.motion,
        checkpoint_path=args.checkpoint,
        acceptance_report=args.acceptance_report,
        scene=args.scene,
        output_dir=args.output,
    )
    print(output)
    return output


if __name__ == "__main__":
    main()
