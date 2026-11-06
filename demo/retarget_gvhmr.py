#!/usr/bin/env python3
"""Retarget a GVHMR prediction to Unitree G1, correcting the Y-up floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

FEET = np.array([7, 8, 10, 11], dtype=np.intp)
GROUND_AXIS = 1


def ground_yup(
    joints: np.ndarray,
    translations: np.ndarray,
    *,
    preserve_flight: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift SMPL-X Y-up joints/translations so the lowest foot is at Y=0.

    Returns new arrays; the inputs are never modified.  X and Z are untouched.
    """
    joints = np.asarray(joints)
    translations = np.asarray(translations)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(f"joints must be NxJx3, got {joints.shape}")
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError(f"translations must be Nx3, got {translations.shape}")
    if joints.shape[0] != translations.shape[0]:
        raise ValueError("joints and translations must have the same frame count")
    if joints.shape[1] < 12:
        raise ValueError(f"joints must have at least 12 joints, got {joints.shape[1]}")
    if not np.isfinite(joints).all() or not np.isfinite(translations).all():
        raise ValueError("joints and translations must be finite")

    offset = joints[:, FEET, GROUND_AXIS].min(axis=1)
    if preserve_flight:
        # A fixed floor retains vertical takeoff; it does not attempt to infer
        # contact or remove monocular world-height drift during airborne phases.
        offset = np.full_like(offset, offset.min())
    shifted_joints = joints.copy()
    shifted_trans = translations.copy()
    shifted_joints[:, :, GROUND_AXIS] -= offset[:, None]
    shifted_trans[:, GROUND_AXIS] -= offset
    return shifted_joints, shifted_trans, offset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr_pred_file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gmr-root", required=True, type=Path)
    parser.add_argument("--src-fps", type=float, default=30.0)
    parser.add_argument("--ground-human", action="store_true")
    parser.add_argument(
        "--preserve-flight",
        action="store_true",
        help="use a fixed floor instead of per-frame supported dance correction",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.src_fps) or args.src_fps <= 0:
        parser.error("--src-fps must be a finite value > 0")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite existing retarget result: {args.output}")
    if args.preserve_flight and not args.ground_human:
        parser.error("--preserve-flight requires --ground-human")

    # GMR and torch are only needed when actually retargeting, so importing this
    # module (e.g. for ground_yup tests) stays cheap and dependency-free.
    import torch
    from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: N817
    from general_motion_retargeting.utils.smpl import (
        get_gvhmr_data_offline_fast,
        load_gvhmr_pred_file,
    )

    pred = args.gvhmr_pred_file
    output = args.output
    smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(
        pred, args.gmr_root / "assets" / "body_models"
    )
    smplx_data["mocap_frame_rate"] = np.array(args.src_fps)
    joints_np = smplx_output.joints.detach().cpu().numpy()
    trans_np = np.asarray(smplx_data["trans"])
    foot_offsets = joints_np[:, FEET, GROUND_AXIS].min(axis=1)
    root_y_range = [float(trans_np[:, GROUND_AXIS].min()), float(trans_np[:, GROUND_AXIS].max())]
    foot_offset_range = [float(foot_offsets.min()), float(foot_offsets.max())]
    ground_axis = None
    print(f"[retarget] human height {float(human_height):.3f} m")
    if args.ground_human:
        shifted_joints, shifted_trans, offset = ground_yup(
            joints_np, trans_np, preserve_flight=args.preserve_flight
        )
        smplx_output.joints = torch.as_tensor(
            shifted_joints,
            dtype=smplx_output.joints.dtype,
            device=smplx_output.joints.device,
        )
        smplx_data["trans"] = shifted_trans
        foot_offset_range = [float(offset.min()), float(offset.max())]
        ground_axis = "y"
        print(f"[retarget] grounded Y: foot offset {offset.min():+.3f}..{offset.max():+.3f} m")

    frames, _ = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.src_fps
    )
    print(f"[retarget] {len(frames)} frames at {args.src_fps:g} fps")

    retarget = GMR(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
    )
    qpos = [np.asarray(retarget.retarget(frame), dtype=np.float64) for frame in frames]
    qpos_array = np.stack(qpos)
    if qpos_array.ndim != 2 or qpos_array.shape[1] != 36:
        raise RuntimeError(f"expected Nx36 qpos, got {qpos_array.shape}")
    if not np.isfinite(qpos_array).all():
        raise RuntimeError("retargeted qpos contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, qpos=qpos_array, fps=np.array([float(args.src_fps)]))
    metadata = {
        "source": str(pred),
        "source_sha256": _sha256(pred),
        "src_fps": float(args.src_fps),
        "ground_axis": ground_axis,
        "contact_mode": "preserve-flight"
        if args.preserve_flight
        else ("support" if args.ground_human else "none"),
        "human_height": float(human_height),
        "root_y_range": root_y_range,
        "foot_offset_range": foot_offset_range,
    }
    with output.with_suffix(".json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    print(f"[retarget] saved {qpos_array.shape} -> {output}")


if __name__ == "__main__":
    main()
