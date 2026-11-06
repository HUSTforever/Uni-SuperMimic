"""Retime an existing deployment bundle by retiming its non-stationary phases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle, _build_reference
from unisupermimic.rl.algos.supermimic.geometry import quat_slerp

PHASES = ("standing", "entry", "dance", "exit", "settle")
FIXED_PHASES = ("standing", "settle")
SCENE = Path("deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_segments(metadata: dict) -> dict[str, list[int]]:
    raw = metadata["sequence"]["segments"]
    if not isinstance(raw, dict) or set(raw) != set(PHASES):
        raise ValueError(f"metadata.sequence.segments must contain exactly {', '.join(PHASES)}")
    segments: dict[str, list[int]] = {}
    for phase in PHASES:
        value = raw[phase]
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"metadata.sequence.segments[{phase!r}] must be [start, end]")
        start, end = (int(item) for item in value)
        if start < 0 or end <= start:
            raise ValueError(f"metadata.sequence.segments[{phase!r}] is invalid: {value!r}")
        segments[phase] = [start, end]
    return segments


def _new_segments(
    old: dict[str, list[int]], speed: float
) -> tuple[dict[str, list[int]], dict[str, int]]:
    lengths: dict[str, int] = {}
    for phase in PHASES:
        start, end = old[phase]
        phase_length = end - start
        lengths[phase] = (
            phase_length if phase in FIXED_PHASES else max(1, int(round(phase_length / speed)))
        )
    segments: dict[str, list[int]] = {}
    cursor = 0
    for phase in PHASES:
        segments[phase] = [cursor, cursor + lengths[phase]]
        cursor += lengths[phase]
    return segments, lengths


def _source_mapping(old: dict[str, list[int]], lengths: dict[str, int]) -> np.ndarray:
    pieces = []
    for index, phase in enumerate(PHASES):
        start, end = old[phase]
        values = np.linspace(start, end, lengths[phase] + 1)
        if index < len(PHASES) - 1:
            values = values[:-1]
        pieces.append(values)
    return np.concatenate(pieces)


def _old_to_new_frame(frame: int, old: dict[str, list[int]], new: dict[str, list[int]]) -> int:
    for phase in PHASES:
        start, end = old[phase]
        if start <= frame <= end:
            new_start, new_end = new[phase]
            mapped = new_start + (frame - start) * (new_end - new_start) / (end - start)
            return int(round(mapped))
    raise ValueError(f"core_output frame {frame} is outside the source segments")


def retime(source: DeploymentBundle, speed: float, output: Path) -> Path:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    if not math.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("speed must be finite and in (0, 1]")

    old_segments = _read_segments(source.metadata)
    new_segments, lengths = _new_segments(old_segments, speed)
    mapping = _source_mapping(old_segments, lengths)
    if len(mapping) != new_segments[PHASES[-1]][1] + 1:
        raise ValueError("retimed frame count does not match the new segments")

    reference = source.reference
    num_frames = len(reference["q"])
    lower = np.floor(mapping).astype(np.int64)
    lower = np.clip(lower, 0, num_frames - 2)
    alpha = (mapping - lower).astype(np.float64)

    q = reference["q"][lower] * (1 - alpha[:, None]) + reference["q"][lower + 1] * alpha[:, None]
    root_p = (
        reference["root_p"][lower] * (1 - alpha[:, None])
        + reference["root_p"][lower + 1] * alpha[:, None]
    )
    quaternion = torch.tensor(reference["root_quat"], dtype=torch.float64)
    root_quat = quat_slerp(
        quaternion[lower],
        quaternion[lower + 1],
        torch.tensor(alpha, dtype=torch.float64),
    ).numpy()

    raw = np.concatenate([root_p, root_quat[:, [1, 2, 3, 0]], q], axis=-1)
    if raw.shape[1] != 36:
        raise ValueError("retimed CSV must have 36 columns")

    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        csv = staged / "sequence.csv"
        np.savetxt(csv, raw, delimiter=",", fmt="%.10g")
        sidecar = csv.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                dict(
                    fps=50,
                    frames=len(raw),
                    csv_sha256=_sha256(csv),
                    quaternion_order="xyzw",
                )
            ),
            encoding="utf-8",
        )

        new_reference = _build_reference(csv, SCENE, 10)
        np.savez(staged / "reference.npz", **new_reference)
        shutil.copyfile(source.policy_path, staged / "policy.ts")

        manifest = copy.deepcopy(
            json.loads((source.directory / "manifest.json").read_text(encoding="utf-8"))
        )
        metadata = manifest["metadata"]
        manifest["files"] = {
            "policy.ts": _sha256(staged / "policy.ts"),
            "reference.npz": _sha256(staged / "reference.npz"),
        }
        metadata["motion_csv_sha256"] = _sha256(csv)
        metadata["source_motion_metadata_sha256"] = _sha256(sidecar)
        metadata["sequence"]["segments"] = copy.deepcopy(new_segments)

        core_output = metadata["sequence"].get("core_output")
        if core_output is not None:
            if (
                not isinstance(core_output, list)
                or len(core_output) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in core_output)
            ):
                raise ValueError("metadata.sequence.core_output must contain two integers")
            metadata["sequence"]["core_output"] = [
                _old_to_new_frame(int(frame), old_segments, new_segments) for frame in core_output
            ]

        metadata["retiming"] = {
            "speed": speed,
            "source_sequence": copy.deepcopy(source.metadata["sequence"]),
            "newsegments": copy.deepcopy(new_segments),
            "source_manifest_sha256": _sha256(source.directory / "manifest.json"),
            "source_reference_sha256": _sha256(source.directory / "reference.npz"),
        }
        metadata["sequence"]["edge_timewarp"] = (
            "Moving phases uniformly retimed; original pose curve interpolated with linear/SLERP sampling"
        )
        metadata["accepted_for_delivery"] = False
        metadata["hardware_ready"] = False
        metadata["candidate_status"] = "unvalidated"

        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        DeploymentBundle.load(staged)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output directory already exists: {output}")
        staged.rename(output)
        published = True
    finally:
        if not published:
            shutil.rmtree(staged, ignore_errors=True)
    return output


def main() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = retime(DeploymentBundle.load(args.source), args.speed, args.output)
    print(result)
    return result


if __name__ == "__main__":
    main()
