#!/usr/bin/env python3
"""Convert native GMR wxyz qpos to validated, optionally smoothed G1 CSV xyzw."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from unisupermimic.tasks.motion_tracking.g1.supermimic.preprocess import prepare_motion
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=SCENE)
    parser.add_argument(
        "--gmr-model", type=Path, help="optional source XML to verify/reorder joints"
    )
    parser.add_argument("--ground-clamp", action="store_true")
    parser.add_argument("--face-yaw-deg", type=float)
    parser.add_argument("--src-fps", type=float, help="required only if NPZ has no fps")
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--smooth-seconds", type=float, default=0.0)
    parser.add_argument("--slowdown", type=float, default=1.0)
    parser.add_argument(
        "--preserve-flight",
        action="store_true",
        help="preserve detected airborne phases; omit for supported dance",
    )
    args = parser.parse_args()
    with np.load(args.npz, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64).copy()
        if args.src_fps is None and "fps" not in data:
            parser.error("NPZ has no fps; supply --src-fps explicitly")
        fps = args.src_fps if args.src_fps is not None else float(np.asarray(data["fps"]).item())
    if args.gmr_model:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(args.gmr_model))
        names = [model.joint(i).name for i in range(1, model.njnt)]
        if set(names) != set(G1_JOINT_NAMES):
            parser.error("source model does not have the expected G1 29 joints")
        qpos[:, 7:] = qpos[:, 7:][:, [names.index(name) for name in G1_JOINT_NAMES]]
    rows, meta = prepare_motion(
        qpos,
        fps,
        model_path=args.model,
        output_fps=args.output_fps,
        smooth_seconds=args.smooth_seconds,
        slowdown=args.slowdown,
        face_yaw_deg=args.face_yaw_deg,
        ground=args.ground_clamp,
        preserve_flight=args.preserve_flight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, rows, fmt="%.8f", delimiter=",")
    meta.update(
        source=str(args.npz.resolve()),
        source_sha256=hashlib.sha256(args.npz.read_bytes()).hexdigest(),
        csv_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        model=str(args.model.resolve()),
    )
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {rows.shape} at {meta['fps']} fps, {meta['duration_seconds']:.3f}s -> {args.output}"
    )


if __name__ == "__main__":
    main()
