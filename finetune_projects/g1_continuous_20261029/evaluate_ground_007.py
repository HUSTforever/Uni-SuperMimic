"""Mechanically evaluate the run_007 checkpoints and contact probe results.

Run from the repository root.  This script never trains, starts DDS, or renders.
"""

from __future__ import annotations

import json
import subprocess
import sys
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
BASELINE_METRICS = PROJECT / "evaluation/balance_002000/balance_full3d_metrics.json"
BASELINE_CONTACT = PROJECT / "evaluation/balance_002000/contact_probe.json"
CANDIDATES = PROJECT / "ground007_candidates.json"
SELECTION = PROJECT / "ground007_selection.json"
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
    "gravity_error_rms_deg",
    "body_angular_velocity_error_rms_rad_s",
)
POLL_SECONDS = 5.0
TOTAL_WAIT_SECONDS = 1800.0


def _wait_for_checkpoint(path: Path, deadline: float) -> None:
    while not (path.is_file() and path.stat().st_size):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out after {TOTAL_WAIT_SECONDS:g}s waiting for {path}")
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
    score: float | None,
    contact: dict | None,
    baseline_contact: dict,
) -> bool:
    if not complete or joint_rmse is None or score is None or contact is None:
        return False
    if not contact.get("completed") or contact.get("failure") is not None:
        return False
    if not float(joint_rmse) < 0.13:
        return False
    if not all(float(dance[key]) <= 1.05 * float(baseline[key]) for key in LIMIT_KEYS):
        return False
    if not score <= 1.05:
        return False
    force_speed = contact.get("force_weighted_contact_speed_rms_m_s")
    baseline_force_speed = baseline_contact["force_weighted_contact_speed_rms_m_s"]
    if force_speed is None or float(force_speed) > 0.95 * float(baseline_force_speed):
        return False
    no_contact = contact.get("no_loaded_sole_contact_fraction")
    baseline_no_contact = baseline_contact["no_loaded_sole_contact_fraction"]
    if no_contact is None or float(no_contact) > float(baseline_no_contact) + 0.01:
        return False
    return True


def main() -> None:
    if not PROJECT.is_dir() or not SCENE.is_file():
        raise SystemExit("run this script from the repository root")
    torch.set_num_threads(1)
    source = DeploymentBundle.load(SOURCE)
    baseline = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))["dance"]
    baseline_contact = json.loads(BASELINE_CONTACT.read_text(encoding="utf-8"))
    rows = _load_rows(CANDIDATES)
    deadline = time.monotonic() + TOTAL_WAIT_SECONDS
    for iteration in ITERATIONS:
        checkpoint = PROJECT / "run_007" / f"ckpt_{iteration:06d}.pt"
        _wait_for_checkpoint(checkpoint, deadline)
        bundle_dir = PROJECT / "candidates" / f"ground007_{iteration:06d}"
        evaluation_dir = PROJECT / "evaluation" / f"ground007_{iteration:06d}"
        export_candidate(source, checkpoint, bundle_dir)
        bundle = DeploymentBundle.load(bundle_dir)
        report = evaluate_sequence(bundle, SCENE, evaluation_dir)
        dance_phase = report["sequence"]["phases"].get("dance", {})
        joint_rmse = dance_phase.get("joint_rmse_rad")
        complete = bool(report["all_attempted_phases_passed"])
        if not complete:
            row = dict(
                run="run_007",
                iteration=iteration,
                bundle=str(bundle.directory),
                complete=False,
                dance_joint_rmse=joint_rmse,
                metrics=None,
                contact=None,
                score=None,
                eligible=False,
                report=report,
            )
            rows.append(row)
            write_json(CANDIDATES, rows)
            print(json.dumps(row), flush=True)
            continue
        with np.load(evaluation_dir / "trajectory.npz") as trajectory:
            qpos = trajectory["qpos"]
            sample_time = trajectory["sample_time"]
            metrics = {
                name: balance_metrics(bundle, SCENE, qpos, sample_time, window=window)
                for name, window in WINDOWS
            }
        write_json(evaluation_dir / "balance_full3d_metrics.json", metrics)
        dance = metrics["dance"]
        probe_path = evaluation_dir / "contact_probe.json"
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "contact_probe.py"),
                str(bundle_dir),
                str(probe_path),
            ],
            check=True,
        )
        contact_result = json.loads(probe_path.read_text(encoding="utf-8"))
        contact = {key: value for key, value in contact_result.items() if key != "samples"}
        score = _score(dance, baseline)
        row = dict(
            run="run_007",
            iteration=iteration,
            bundle=str(bundle.directory),
            complete=True,
            dance_joint_rmse=joint_rmse,
            metrics=metrics,
            contact=contact,
            score=score,
            eligible=_eligible(True, joint_rmse, dance, baseline, score, contact, baseline_contact),
            report=report,
        )
        rows.append(row)
        write_json(CANDIDATES, rows)
        print(json.dumps(row), flush=True)

    eligible_rows = [row for row in rows if row.get("eligible")]
    selected = (
        min(
            eligible_rows,
            key=lambda row: (
                float(row["contact"]["force_weighted_contact_speed_rms_m_s"]),
                float(row["score"]),
            ),
        )
        if eligible_rows
        else None
    )
    write_json(
        SELECTION,
        dict(rows=rows, selected=selected, hardware_ready=False),
    )


if __name__ == "__main__":
    main()
