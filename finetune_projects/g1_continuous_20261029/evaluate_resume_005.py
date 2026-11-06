"""Mechanically evaluate the run_005 checkpoints recovered after resume.

Run from the repository root.  This script never trains, starts DDS, or renders.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.run_sim2sim import write_json
from unisupermimic.finetune.balance_metrics import balance_metrics
from unisupermimic.finetune.export_candidate import export_candidate
from unisupermimic.finetune.sequence_eval import evaluate_sequence

PROJECT = Path("finetune_projects/g1_continuous_20261029")
SCENE = Path("deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
SOURCE = PROJECT / "data/sequence_v2"
BASELINE = PROJECT / "evaluation/balance_002000/balance_full3d_metrics.json"
ITERATIONS = (250, 500, 750, 1000)
WINDOWS = (
    ("dance", (5, 27.96)),
    ("hard_20_24", (20, 24)),
    ("hard_26_30", (26, 30)),
)
SCORE_KEYS = (
    "gravity_error_rms_deg",
    "body_angular_velocity_error_rms_rad_s",
    "body_angular_acceleration_rms_rad_s2",
    "reference_stance_point_speed_rms_m_s",
)
LIMIT_KEYS = (
    "near_ground_point_speed_rms_m_s",
    "reference_stance_point_speed_rms_m_s",
    "gravity_error_rms_deg",
    "body_angular_velocity_error_rms_rad_s",
)
POLL_SECONDS = 5.0
TIMEOUT_SECONDS = 1200.0


def _wait_for_checkpoint(path: Path, deadline: float) -> None:
    while not (path.is_file() and path.stat().st_size):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out after {TIMEOUT_SECONDS:g}s waiting for {path}")
        time.sleep(min(POLL_SECONDS, remaining))


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a row list in {path}")
    return rows


def _score(dance: dict, baseline: dict) -> float:
    ratios = [float(dance[key]) / float(baseline[key]) for key in SCORE_KEYS]
    return float(sum(ratios) / len(ratios))


def _eligible(
    complete: bool,
    joint_rmse: float | None,
    dance: dict,
    baseline: dict,
    score: float,
) -> bool:
    limits_ok = all(float(dance[key]) <= 1.05 * float(baseline[key]) for key in LIMIT_KEYS)
    return bool(
        complete
        and joint_rmse is not None
        and float(joint_rmse) < 0.13
        and limits_ok
        and score <= 0.98
    )


def main() -> None:
    if not PROJECT.is_dir() or not SCENE.is_file():
        raise SystemExit("run this script from the repository root")
    torch.set_num_threads(1)
    source = DeploymentBundle.load(SOURCE)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["dance"]
    rows = _load_rows(PROJECT / "resume005_candidates.json")
    deadline = time.monotonic() + TIMEOUT_SECONDS
    for iteration in ITERATIONS:
        checkpoint = PROJECT / "run_005" / f"ckpt_{iteration:06d}.pt"
        _wait_for_checkpoint(checkpoint, deadline)
        bundle_dir = PROJECT / "candidates" / f"resume005_{iteration:06d}"
        evaluation_dir = PROJECT / "evaluation" / f"resume005_{iteration:06d}"
        export_candidate(source, checkpoint, bundle_dir)
        bundle = DeploymentBundle.load(bundle_dir)
        report = evaluate_sequence(bundle, SCENE, evaluation_dir)
        with np.load(evaluation_dir / "trajectory.npz") as trajectory:
            qpos = trajectory["qpos"]
            sample_time = trajectory["sample_time"]
            metrics = {
                name: balance_metrics(bundle, SCENE, qpos, sample_time, window=window)
                for name, window in WINDOWS
            }
        write_json(evaluation_dir / "balance_full3d_metrics.json", metrics)
        dance = metrics["dance"]
        joint_rmse = report["sequence"]["phases"]["dance"]["joint_rmse_rad"]
        complete = bool(report["all_attempted_phases_passed"])
        score = _score(dance, baseline)
        row = dict(
            run="run_005",
            iteration=iteration,
            bundle=str(bundle.directory),
            complete=complete,
            dance_joint_rmse=joint_rmse,
            metrics=metrics,
            score=score,
            eligible=_eligible(complete, joint_rmse, dance, baseline, score),
        )
        rows.append(row)
        write_json(PROJECT / "resume005_candidates.json", rows)
        print(json.dumps(row), flush=True)

    eligible_rows = [row for row in rows if row.get("eligible")]
    selected = min(eligible_rows, key=lambda row: row["score"]) if eligible_rows else None
    write_json(
        PROJECT / "resume005_selection.json",
        dict(rows=rows, selected=selected, hardware_ready=False),
    )


if __name__ == "__main__":
    main()
