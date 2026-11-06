"""Evaluate physical PPO checkpoints against unchanged continuity/style gates."""

import json
import sys
from pathlib import Path

import torch
from projection_eval import evaluate

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.run_sim2sim import write_json
from unisupermimic.finetune.export_candidate import export_candidate

P = Path("finetune_projects/g1_continuity_20261029")


def main():
    torch.set_num_threads(1)
    source = DeploymentBundle.load(P / "candidates/run002_000250")
    results = P / "PHYSICAL_PPO_MEASURED_RESULTS.json"
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
    for iteration in map(int, sys.argv[1:]):
        if any(r["iteration"] == iteration for r in summary["rows"]):
            raise ValueError("checkpoint already evaluated")
        candidate = P / f"candidates/physical009_{iteration:06d}"
        export_candidate(
            source, P / f"run_009_physical_measured/ckpt_{iteration:06d}.pt", candidate
        )
        output = P / f"evaluation/physical009_{iteration:06d}"
        report = evaluate(candidate, output)
        continuity = report["continuity"]
        summary["rows"].append(
            dict(
                iteration=iteration,
                bundle=str(candidate),
                physical_complete=report["all_attempted_phases_passed"],
                original_style=report["original_style"],
                worst_hf=continuity["worst_ankle_window"]["ankle_hf_position_max_rad"],
                phase_hf={
                    name: r["ankle_hf_position_max_rad"] for name, r in continuity["phases"].items()
                },
                worst_root_hf=continuity["worst_root_window"]["root_vertical_hf_position_rms_m"],
                full_pass=report["full_sequence_continuity_and_original_style_passed"],
            )
        )
        write_json(results, summary)
        print(json.dumps(summary["rows"][-1]), flush=True)


if __name__ == "__main__":
    main()
