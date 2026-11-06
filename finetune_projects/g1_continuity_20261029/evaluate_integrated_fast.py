"""Evaluate integrated fast checkpoints against unchanged continuity/style gates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from projection_eval import evaluate

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.run_sim2sim import write_json
from unisupermimic.finetune.export_candidate import export_candidate
from unisupermimic.finetune.finish import summarize_finish

P = Path("finetune_projects/g1_continuity_20261029")


def main():
    arms = {
        "control": (
            "run_014_fast_control",
            "fast014",
            "FAST_INTEGRATED_CONTROL_RESULTS.json",
        ),
        "strong": (
            "run_015_fast_strong",
            "fast015",
            "FAST_INTEGRATED_STRONG_RESULTS.json",
        ),
        "clock_control": (
            "run_016_clock_control",
            "clock016",
            "CRITIC_CLOCK_CONTROL_RESULTS.json",
        ),
        "clock": ("run_017_clock", "clock017", "CRITIC_CLOCK_RESULTS.json"),
        "full_episode": (
            "run_018_clock_full_episode",
            "full018",
            "FULL_EPISODE_RESULTS.json",
        ),
        "bound_mean": (
            "run_019_bound_mean",
            "bound019",
            "ACTION_BOUNDARY_RESULTS.json",
        ),
        "projected_residual": (
            "run_020_projected_residual",
            "projected020",
            "PROJECTED_RESIDUAL_RESULTS.json",
        ),
        "projected_lr": (
            "run_021_projected_residual_lr",
            "projected021",
            "PROJECTED_RESIDUAL_LR_RESULTS.json",
        ),
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=arms)
    parser.add_argument("iterations", type=int, nargs="+")
    args = parser.parse_args()
    run, prefix, filename = arms[args.arm]
    torch.set_num_threads(1)
    source = DeploymentBundle.load(P / "data/fast_integrated_base_v1")
    results = P / filename
    summary = (
        json.loads(results.read_text())
        if results.exists()
        else dict(
            method="physical_ppo_warm_start",
            rows=[],
            selected=None,
            hardware_ready=False,
            goal_complete_claimed=False,
        )
    )
    for iteration in args.iterations:
        if any(r["iteration"] == iteration for r in summary["rows"]):
            raise ValueError("checkpoint already evaluated")
        candidate = P / f"candidates/{prefix}_{iteration:06d}"
        checkpoint = P / run / f"ckpt_{iteration:06d}.pt"
        export_candidate(source, checkpoint, candidate)
        output = P / f"evaluation/{prefix}_{iteration:06d}"
        report = evaluate(candidate, output)
        with np.load(output / "physics_trajectory.npz") as trajectory:
            metadata = DeploymentBundle.load(candidate).metadata
            try:
                finish = summarize_finish(
                    metadata,
                    trajectory["qpos"],
                    trajectory["joint_velocity"],
                    trajectory["root_velocity"],
                    trajectory["sample_time"],
                )
            except ValueError as exc:
                finish = dict(complete=False, passed=False, diagnostic_error=str(exc))
        report["fast_finish"] = finish
        report["full_native_acceptance"] = bool(
            report["full_sequence_continuity_and_original_style_passed"] and finish["passed"]
        )
        write_json(output / "report.json", report)
        continuity = report["continuity"]
        summary["rows"].append(
            dict(
                iteration=iteration,
                bundle=str(candidate),
                physical_complete=report["all_attempted_phases_passed"],
                original_style=report["original_style"],
                worst_hf=continuity.get("worst_ankle_window", {}).get("ankle_hf_position_max_rad"),
                phase_hf={
                    name: r.get("ankle_hf_position_max_rad")
                    for name, r in continuity.get("phases", {}).items()
                },
                worst_root_hf=continuity.get("worst_root_window", {}).get(
                    "root_vertical_hf_position_rms_m"
                ),
                finish=finish,
                full_pass=report["full_native_acceptance"],
            )
        )
        write_json(results, summary)
        print(json.dumps(summary["rows"][-1]), flush=True)


if __name__ == "__main__":
    main()
