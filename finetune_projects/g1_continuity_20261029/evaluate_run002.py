"""Mechanically evaluate the run_002 continuity checkpoints.

Run from the repository root.  The script only exports candidate bundles and
calls the existing continuity evaluator; it never trains, starts DDS, or
renders.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.run_sim2sim import write_json
from unisupermimic.finetune import continuity_metrics
from unisupermimic.finetune.continuity_eval import evaluate_continuity
from unisupermimic.finetune.continuity_metrics import summarize_continuity
from unisupermimic.finetune.export_candidate import export_candidate

PROJECT = Path("finetune_projects/g1_continuity_20261029")
SCENE = Path("deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
SOURCE = Path("finetune_projects/g1_continuity_20261029/data/damping_kd200")
RUN = "run_002"
ITERATIONS = (250, 500, 750, 1000, 1250, 1500)
POLL_SECONDS = 5.0
TOTAL_WAIT_SECONDS = 1500.0
CANDIDATES = PROJECT / "run002_candidates.json"
SELECTION = PROJECT / "run002_selection.json"


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


def _upsert_row(rows: list[dict], row: dict) -> None:
    for index, existing in enumerate(rows):
        if (
            isinstance(existing, dict)
            and existing.get("run") == row["run"]
            and existing.get("iteration") == row["iteration"]
        ):
            rows[index] = row
            return
    rows.append(row)


def _existing_report(
    evaluation_dir: Path, bundle_dir: Path | None = None
) -> tuple[dict | None, Path]:
    report_path = evaluation_dir / "report.json"
    if not report_path.is_file():
        return None, report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"expected an object report in {report_path}")
    continuity = report.get("continuity")
    if (
        isinstance(continuity, dict)
        and continuity.get("metrics_version") == continuity_metrics.CONTINUITY_METRICS_VERSION
    ):
        return report, report_path
    if bundle_dir is None:
        raise ValueError("bundle_dir required to recompute stale continuity metrics")
    trajectory_path = evaluation_dir / "physics_trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        arrays = {
            key: np.asarray(trajectory[key], dtype=float)
            for key in ("sample_time", "qpos", "joint_velocity", "root_velocity")
        }
    metadata = DeploymentBundle.load(bundle_dir).metadata
    continuity = summarize_continuity(
        metadata,
        arrays["qpos"],
        arrays["joint_velocity"],
        arrays["root_velocity"],
        arrays["sample_time"],
    )
    dance_rmse = _dance_joint_rmse(report)
    fidelity = bool(
        isinstance(dance_rmse, (int, float))
        and not isinstance(dance_rmse, bool)
        and math.isfinite(dance_rmse)
        and dance_rmse < 0.13
    )
    report["continuity"] = continuity
    report["continuity_fidelity_passed"] = fidelity
    report["full_sequence_and_continuity_passed"] = bool(
        report.get("all_attempted_phases_passed") is True and continuity["passed"] and fidelity
    )
    report["goal_complete_claimed"] = False
    write_json(report_path, report)
    return report, report_path


def _dance_joint_rmse(report: dict) -> float | None:
    dance = report.get("sequence", {}).get("phases", {}).get("dance", {})
    return dance.get("joint_rmse_rad") if isinstance(dance, dict) else None


def _make_row(report: dict, report_path: Path, iteration: int, bundle_dir: Path) -> dict:
    continuity = report.get("continuity")
    if not isinstance(continuity, dict):
        raise ValueError("continuity report dictionary required")
    return dict(
        run=RUN,
        iteration=iteration,
        bundle=str(bundle_dir),
        physical_complete=bool(report.get("all_attempted_phases_passed", False)),
        full_pass=bool(report.get("full_sequence_and_continuity_passed", False)),
        dance_joint_rmse=_dance_joint_rmse(report),
        continuity=continuity,
        report=str(report_path),
    )


def _worst_ankle_position(row: dict) -> float | None:
    if row.get("full_pass") is not True:
        return None
    continuity = row.get("continuity")
    if not isinstance(continuity, dict):
        return None
    window = continuity.get("worst_ankle_window")
    if not isinstance(window, dict):
        return None
    value = window.get("ankle_hf_position_max_rad")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _select(rows: list[dict]) -> dict | None:
    ranked = []
    for row in rows:
        score = _worst_ankle_position(row)
        if score is not None:
            ranked.append((score, row))
    return min(ranked, key=lambda item: item[0])[1] if ranked else None


def _print_row(row: dict) -> None:
    continuity = row["continuity"]
    phases = continuity.get("phases", {})
    summary = dict(
        run=row["run"],
        iteration=row["iteration"],
        physical_complete=row["physical_complete"],
        full_pass=row["full_pass"],
        dance_joint_rmse=row["dance_joint_rmse"],
        continuity_passed=continuity.get("passed"),
        phase_ankle_hf_position_max_rad={
            name: phase.get("ankle_hf_position_max_rad")
            for name, phase in phases.items()
            if isinstance(phase, dict)
        },
        worst_ankle_window=continuity.get("worst_ankle_window"),
    )
    print(json.dumps(summary, allow_nan=False), flush=True)


def main() -> None:
    if not PROJECT.is_dir() or not SCENE.is_file():
        raise SystemExit("run this script from the repository root")
    torch.set_num_threads(1)
    source = None
    rows = _load_rows(CANDIDATES)
    deadline = time.monotonic() + TOTAL_WAIT_SECONDS
    for iteration in ITERATIONS:
        candidate_dir = PROJECT / "candidates" / f"run002_{iteration:06d}"
        evaluation_dir = PROJECT / "evaluation" / f"run002_{iteration:06d}"
        report, report_path = _existing_report(evaluation_dir, candidate_dir)
        if report is None:
            checkpoint = PROJECT / "run_002" / f"ckpt_{iteration:06d}.pt"
            _wait_for_checkpoint(checkpoint, deadline)
            if not candidate_dir.exists():
                if source is None:
                    source = DeploymentBundle.load(SOURCE)
                export_candidate(source, checkpoint, candidate_dir)
            if evaluation_dir.exists():
                raise RuntimeError(
                    f"{evaluation_dir} exists without a usable report.json; "
                    "refusing to overwrite an incomplete evaluation"
                )
            bundle = DeploymentBundle.load(candidate_dir)
            report = evaluate_continuity(bundle, SCENE, evaluation_dir)
            report_path = evaluation_dir / "report.json"
        row = _make_row(report, report_path, iteration, candidate_dir)
        _upsert_row(rows, row)
        rows.sort(key=lambda item: int(item["iteration"]))
        write_json(CANDIDATES, rows)
        _print_row(row)
    write_json(SELECTION, dict(rows=rows, selected=_select(rows), hardware_ready=False))


if __name__ == "__main__":
    main()
