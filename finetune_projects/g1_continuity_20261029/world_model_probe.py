"""Check learned contact dynamics against an exactly replayed physical rollout."""

from pathlib import Path

import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.run_sim2sim import write_json
from unisupermimic.finetune.collector import DeploymentCollector
from unisupermimic.rl.algos.supermimic.networks import WorldModel

PROJECT = Path("finetune_projects/g1_continuity_20261029")
ANKLES = [4, 5, 10, 11]


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    output = PROJECT / "evaluation/wm_probe_run002_250"
    output.mkdir(parents=True, exist_ok=False)
    bundle = DeploymentBundle.load(PROJECT / "candidates/run002_000250")
    trajectory = np.load(PROJECT / "evaluation/run002_000250/trajectory.npz")
    collector = DeploymentCollector(
        bundle,
        "deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml",
        num_envs=1,
        randomize=False,
    )
    sim = collector.simulators[0]
    sim.reset("reference_zero_velocity")
    states = [[] for _ in range(6)]
    max_error = 0.0
    for frame in range(len(trajectory["qpos"])):
        max_error = max(max_error, float(np.abs(sim.qpos() - trajectory["qpos"][frame]).max()))
        for destination, value in zip(states, collector._physical_state()):
            destination.append(value)
        if frame + 1 < len(trajectory["qpos"]):
            sim.advance(trajectory["target"][frame])
    assert max_error < 1e-10, max_error
    states = tuple(torch.cat(part, dim=0) for part in states)
    actions = torch.from_numpy(trajectory["raw_action"][:-1]).float()
    context = torch.zeros(len(actions), 33)
    context[:, 0] = 1.0
    normalized_context = collector.supermimic_mdp.normalize_context(context)
    checkpoint = torch.load(
        PROJECT / "run_002/ckpt_000250.pt", map_location="cpu", weights_only=True
    )
    wm = WorldModel(hidden=checkpoint["args"]["wm_hidden"], num_joints=29, context_dim=33).eval()
    wm.load_state_dict(checkpoint["wm"])
    ref_qd = torch.from_numpy(bundle.reference["qd"].copy())
    results = {}
    for horizon in (1, 8, 32):
        starts = torch.arange(0, len(actions) - horizon + 1, 5)
        predicted = tuple(value[starts] for value in states)
        for offset in range(horizon):
            index = starts + offset
            predicted = wm.step(predicted, actions[index], normalized_context[index])
        ends = starts + horizon
        actual = tuple(value[ends] for value in states)
        phases = {}
        for phase, (a, b) in bundle.metadata["sequence"]["segments"].items():
            mask = (starts >= a) & (ends <= b)
            if not mask.any():
                continue
            predicted_cost = (
                ((predicted[5][mask][:, ANKLES] - ref_qd[ends[mask]][:, ANKLES]) / 2)
                .square()
                .mean()
            )
            actual_cost = (
                ((actual[5][mask][:, ANKLES] - ref_qd[ends[mask]][:, ANKLES]) / 2).square().mean()
            )
            phases[phase] = dict(
                samples=int(mask.sum()),
                joint_position_rmse_rad=float(
                    (predicted[4][mask] - actual[4][mask]).square().mean().sqrt()
                ),
                ankle_velocity_rmse_rad_s=float(
                    (predicted[5][mask][:, ANKLES] - actual[5][mask][:, ANKLES])
                    .square()
                    .mean()
                    .sqrt()
                ),
                root_vertical_velocity_rmse_m_s=float(
                    (predicted[2][mask, 2] - actual[2][mask, 2]).square().mean().sqrt()
                ),
                predicted_ankle_velocity_cost=float(predicted_cost),
                actual_ankle_velocity_cost=float(actual_cost),
                cost_ratio=float(predicted_cost / actual_cost) if actual_cost > 1e-8 else None,
            )
        results[str(horizon)] = phases
    report = dict(
        physical_replay_max_qpos_error=max_error,
        horizons_steps=results,
        explanation="Open-loop predicted states use recorded physical actions; no policy optimization or promotion.",
        goal_complete_claimed=False,
        hardware_ready=False,
    )
    write_json(output / "report.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
