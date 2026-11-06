"""Export an explicitly unvalidated fine-tuning candidate deployment bundle.

The candidate reuses a trusted source bundle's reference motion and controller
constants while binding a newly fine-tuned policy and checkpoint.  It is an
evaluation artifact only: no acceptance, delivery or hardware-readiness claim
is made.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from unisupermimic.deploy.bundle import (
    MANIFEST_NAME,
    POLICY_NAME,
    REFERENCE_NAME,
    SCHEMA_VERSION,
    DeploymentBundle,
    _validate_policy_weights,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")


def _contract_equal(left: Any, right: Any) -> bool:
    """Compare contract values without conflating booleans and numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            return float(left) == float(right)
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_contract_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _contract_equal(left[key], right[key]) for key in left
        )
    return left == right


def _validate_contract(source_metadata: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    from demo.export_policy import export_metadata

    try:
        expected = export_metadata(checkpoint)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("policy contract metadata could not be derived from checkpoint") from exc
    for key, value in expected.items():
        if key not in source_metadata:
            raise ValueError(
                f"policy contract field {key!r} is missing from the source bundle metadata"
            )
        if not _contract_equal(value, source_metadata[key]):
            raise ValueError(
                f"policy contract field {key!r} changed: checkpoint metadata {value!r} "
                f"does not match source bundle metadata {source_metadata[key]!r}"
            )


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint could not be loaded with weights_only=True") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if not isinstance(checkpoint.get("pi"), dict):
        raise ValueError("checkpoint must contain a pi state dict")
    if not isinstance(checkpoint.get("args"), dict):
        raise ValueError("checkpoint must contain an args dictionary")
    return checkpoint


def _trace_policy(checkpoint: dict[str, Any], policy_path: Path) -> None:
    import torch
    from demo.export_policy import build_policy

    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import (
        GOAL_FRAME_DIM,
        OBS_DIM,
    )

    policy = build_policy(checkpoint).eval()
    args = checkpoint["args"]
    hist = int(args["hist"])
    goal_dim = GOAL_FRAME_DIM * int(args["goal_frames"])
    generator = torch.Generator().manual_seed(20261029)
    history = torch.randn(2, hist, OBS_DIM, generator=generator, dtype=torch.float32)
    goal = torch.randn(2, goal_dim, generator=generator, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(policy, (history, goal), strict=False)
    traced.save(str(policy_path))
    if not policy_path.is_file():
        raise ValueError("canonical policy TorchScript was not written")


def _build_manifest(
    source_bundle: DeploymentBundle,
    checkpoint_sha256: str,
    policy_sha256: str,
    reference_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    metadata = copy.deepcopy(source_bundle.metadata)
    metadata["source_checkpoint_sha256"] = checkpoint_sha256
    metadata["candidate_status"] = "unvalidated"
    metadata["accepted_for_delivery"] = False
    metadata["hardware_ready"] = False
    metadata["fine_tune_source_bundle_sha256"] = source_manifest_sha256
    return {
        "schema_version": SCHEMA_VERSION,
        "files": {
            POLICY_NAME: policy_sha256,
            REFERENCE_NAME: reference_sha256,
        },
        "metadata": metadata,
    }


def export_candidate(
    source_bundle: DeploymentBundle,
    checkpoint_path: Path,
    output_dir: Path,
) -> Path:
    """Validate a new policy/checkpoint and atomically publish a candidate bundle.

    The source bundle is treated as immutable.  The candidate manifest records
    the checkpoint and source-manifest hashes, but explicitly remains
    ``unvalidated`` and not accepted for delivery or hardware use.
    """
    output_dir = Path(output_dir)
    checkpoint_path = Path(checkpoint_path)
    source_directory = Path(source_bundle.directory)
    source_reference = source_directory / REFERENCE_NAME
    source_manifest = source_directory / MANIFEST_NAME

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    _require_file(checkpoint_path, "checkpoint")
    _require_file(source_reference, "source reference")
    _require_file(source_manifest, "source manifest")

    checkpoint_sha256 = _sha256(checkpoint_path)
    source_manifest_sha256 = _sha256(source_manifest)
    source_reference_sha256 = _sha256(source_reference)

    checkpoint = _load_checkpoint(checkpoint_path)
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint changed while it was being loaded")
    _validate_contract(source_bundle.metadata, checkpoint)

    parent = output_dir.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(parent)))
    published = False
    try:
        policy_path = staged / POLICY_NAME
        _trace_policy(checkpoint, policy_path)
        policy_sha256 = _sha256(policy_path)
        _validate_policy_weights(policy_path, checkpoint)
        if _sha256(policy_path) != policy_sha256:
            raise ValueError("staged policy.ts changed during weight validation")

        staged_reference = staged / REFERENCE_NAME
        shutil.copyfile(source_reference, staged_reference)
        reference_sha256 = _sha256(staged_reference)
        if reference_sha256 != source_reference_sha256:
            raise ValueError("copied reference.npz SHA256 does not match the source bundle")
        if _sha256(source_reference) != source_reference_sha256:
            raise ValueError("source reference.npz changed during candidate export")
        if _sha256(checkpoint_path) != checkpoint_sha256:
            raise ValueError("checkpoint changed during candidate export")

        manifest = _build_manifest(
            source_bundle,
            checkpoint_sha256,
            policy_sha256,
            reference_sha256,
            source_manifest_sha256,
        )
        (staged / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Final check: nothing is published unless the staged bundle satisfies
        # exactly the same loader contract as an ordinary deployment bundle.
        DeploymentBundle.load(staged)

        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        os.rename(staged, output_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staged, ignore_errors=True)

    return output_dir


def main() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bundle = DeploymentBundle.load(args.source_bundle)
    output = export_candidate(source_bundle, args.checkpoint, args.output)
    print(output)
    return output


if __name__ == "__main__":
    main()
