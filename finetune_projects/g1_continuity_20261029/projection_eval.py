"""Physical continuity evaluation plus a fixed original-motion fidelity anchor."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.run_sim2sim import write_json
from unisupermimic.finetune.continuity_eval import evaluate_continuity

PROJECT = Path("finetune_projects/g1_continuity_20261029")
ANCHOR = Path("finetune_projects/g1_continuous_20261029/data/sequence_v2/reference.npz")
ANCHOR_SHA = "7773170ed06cfcc43e5830e79fe7786c70b696a0709bd8aa33ab7ed7ec033daa"
MAIN = [0, 3, 6, 9, 15, 16, 17, 18, 22, 23, 24, 25]


def _metadata_frame_count(metadata):
    if not isinstance(metadata, dict):
        return None
    sequence = metadata.get("sequence")
    segments = sequence.get("segments") if isinstance(sequence, dict) else None
    if not isinstance(segments, dict):
        return None
    names = ("standing", "entry", "dance", "exit", "settle")
    if len(segments) != len(names) or set(segments) != set(names):
        return None
    bounds = []
    for name in names:
        pair = segments[name]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        if any(isinstance(value, bool) or not isinstance(value, int) for value in pair):
            return None
        start, end = pair
        if end <= start:
            return None
        bounds.append((start, end))
    if bounds[0][0] != 0:
        return None
    for (_, previous_end), (start, _) in zip(bounds, bounds[1:]):
        if start != previous_end:
            return None
    if tuple(bounds[2]) != (250, 1398):
        return None
    return bounds[-1][1] + 1


def original_style_metrics(qpos, *, metadata=None):
    if hashlib.sha256(ANCHOR.read_bytes()).hexdigest() != ANCHOR_SHA:
        raise ValueError("original style anchor changed")
    reference = np.load(ANCHOR)["q"]
    expected_frames = len(reference) if metadata is None else _metadata_frame_count(metadata)
    if expected_frames is None:
        return dict(complete=False, passed=False, anchor_sha256=ANCHOR_SHA)
    qpos = np.asarray(qpos)
    if qpos.shape != (expected_frames, 36) or not np.isfinite(qpos).all():
        return dict(complete=False, passed=False, anchor_sha256=ANCHOR_SHA)
    old, actual = reference[250:1399], qpos[250:1399, 7:]
    rms = float(np.sqrt(np.mean((actual - old) ** 2)))
    amplitude, correlation = {}, {}
    for index in MAIN:
        amplitude[str(index)] = float(np.ptp(actual[:, index]) / np.ptp(old[:, index]))
        correlation[str(index)] = (
            float(np.corrcoef(old[:, index], actual[:, index])[0, 1])
            if np.std(actual[:, index]) > 1e-8
            else 0.0
        )
    return dict(
        complete=True,
        anchor_sha256=ANCHOR_SHA,
        dance_joint_rmse_rad=rms,
        main_joint_indices=MAIN,
        amplitude_ratios=amplitude,
        correlations=correlation,
        limits=dict(dance_joint_rmse_rad=0.13, amplitude_ratio_min=0.8, correlation_min=0.9),
        passed=bool(
            rms < 0.13 and min(amplitude.values()) >= 0.8 and min(correlation.values()) >= 0.9
        ),
    )


def evaluate(bundle_path, output, **kwargs):
    output = Path(output)
    bundle = DeploymentBundle.load(bundle_path)
    report = evaluate_continuity(
        bundle, "deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml", output, **kwargs
    )
    trajectory = np.load(output / "trajectory.npz")
    style = original_style_metrics(trajectory["qpos"], metadata=bundle.metadata)
    report["original_style"] = style
    report["full_sequence_continuity_and_original_style_passed"] = bool(
        report["full_sequence_and_continuity_passed"] and style["passed"]
    )
    write_json(output / "report.json", report)
    print(
        dict(
            bundle=str(bundle_path),
            physical=report["all_attempted_phases_passed"],
            continuity=report["continuity"]["passed"],
            original_style=style,
            full_pass=report["full_sequence_continuity_and_original_style_passed"],
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    report = evaluate(a.bundle, a.output)
    raise SystemExit(0 if report["full_sequence_continuity_and_original_style_passed"] else 1)
