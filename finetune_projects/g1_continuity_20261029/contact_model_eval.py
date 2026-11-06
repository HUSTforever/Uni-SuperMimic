"""Independent physical holdout for fixed-policy contact-model recalibration."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.controller import PolicyController
from unisupermimic.deploy.run_sim2sim import write_json
from unisupermimic.finetune.collector import DeploymentCollector
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import flatten_state, unflatten_state

PROJECT = Path("finetune_projects/g1_continuity_20261029")
DATASET = PROJECT / "evaluation/contact_wm_holdout/physical_holdout.pt"
ANKLES = [4, 5, 10, 11]


@torch.inference_mode()
def prepare():
    DATASET.parent.mkdir(parents=True, exist_ok=False)
    bundle = DeploymentBundle.load(PROJECT / "candidates/run002_000250")
    cases = {}
    for name, friction, noise in (("nominal", 1.0, 0.0), ("friction_noise", 0.85, 0.001)):
        collector = DeploymentCollector(
            bundle,
            "deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml",
            num_envs=1,
            randomize=False,
        )
        sim = collector.simulators[0]
        sim.model.geom_friction[:, 0] = friction
        sim.reset("reference_zero_velocity")
        policy = PolicyController(bundle)
        rng = np.random.default_rng(4242)
        states = [flatten_state(collector._physical_state())[0]]
        actions, positions = [], [sim.qpos()]
        failure = None
        for frame in range(len(bundle.reference["q"]) - 1):
            sensor = sim.sensor()
            if noise:
                sensor.q += rng.normal(0, noise, 29)
                sensor.dq += rng.normal(0, noise * 10, 29)
                sensor.gyro += rng.normal(0, noise * 2, 3)
            target = policy.step(sensor, frame)
            actions.append(policy.previous_action.copy())
            sim.advance(target)
            states.append(flatten_state(collector._physical_state())[0])
            positions.append(sim.qpos())
            failure = sim.failure(frame + 1)
            if failure:
                break
        context = torch.zeros(len(actions), 33)
        context[:, 0] = friction
        alignment_error = None
        if name == "nominal":
            old = np.load(PROJECT / "evaluation/run002_000250/trajectory.npz")
            alignment_error = float(np.abs(np.asarray(positions) - old["qpos"]).max())
            assert alignment_error < 1e-10, alignment_error
        cases[name] = dict(
            states=torch.stack(states),
            actions=torch.tensor(np.asarray(actions)).float(),
            context=collector.supermimic_mdp.normalize_context(context),
            frames=torch.arange(len(states)),
            metadata=dict(
                friction=friction,
                noise=noise,
                seed=4242,
                delay_substeps=0,
                physical_complete=len(states) == len(bundle.reference["q"]) and not failure,
                failure=failure,
                samples=len(states),
                nominal_replay_max_error=alignment_error,
            ),
        )
        print(name, cases[name]["metadata"], flush=True)
    torch.save(
        dict(
            cases=cases,
            segments=bundle.metadata["sequence"]["segments"],
            ref_qd=torch.tensor(bundle.reference["qd"].copy()),
            bundle_manifest_sha256=hashlib.sha256(
                (bundle.directory / "manifest.json").read_bytes()
            ).hexdigest(),
        ),
        DATASET,
    )


@torch.inference_mode()
def evaluate(checkpoint, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    networks.AMP_DTYPE = None
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    wm = networks.WorldModel(
        hidden=saved["args"]["wm_hidden"], num_joints=29, context_dim=33
    ).eval()
    wm.load_state_dict(saved["wm"])
    data = torch.load(DATASET, weights_only=True, map_location="cpu")
    cases = {}
    for name, case in data["cases"].items():
        state = unflatten_state(case["states"])
        results = {}
        for horizon in (1, 8, 32):
            starts = torch.arange(0, len(case["actions"]) - horizon + 1, 5)
            predicted = tuple(value[starts] for value in state)
            for offset in range(horizon):
                index = starts + offset
                predicted = wm.step(predicted, case["actions"][index], case["context"][index])
            ends = starts + horizon
            actual = tuple(value[ends] for value in state)
            error = wm.state_error(predicted, actual)
            phases = {}
            for phase, (a, b) in data["segments"].items():
                mask = (starts >= a) & (ends <= b)
                if not mask.any():
                    continue
                reference_qd = data["ref_qd"][ends[mask]][:, ANKLES]
                pcost = ((predicted[5][mask][:, ANKLES] - reference_qd) / 2).square().mean()
                acost = ((actual[5][mask][:, ANKLES] - reference_qd) / 2).square().mean()
                phases[phase] = dict(
                    samples=int(mask.sum()),
                    joint_position_rmse_rad=float(error[mask, 12:41].square().mean().sqrt()),
                    ankle_velocity_rmse_rad_s=float(
                        (predicted[5][mask][:, ANKLES] - actual[5][mask][:, ANKLES])
                        .square()
                        .mean()
                        .sqrt()
                    ),
                    root_linear_velocity_rmse_m_s=float(
                        (predicted[2][mask] - actual[2][mask]).square().mean().sqrt()
                    ),
                    root_position_error_rms_m=float(error[mask, :3].square().sum(-1).mean().sqrt()),
                    root_rotation_error_rms_rad=float(
                        error[mask, 3:6].square().sum(-1).mean().sqrt()
                    ),
                    predicted_ankle_velocity_cost=float(pcost),
                    actual_ankle_velocity_cost=float(acost),
                    cost_ratio=float(pcost / acost) if acost > 1e-8 else None,
                )
            results[str(horizon)] = phases
        cases[name] = dict(metadata=case["metadata"], horizons=results)
    report = dict(
        checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        cases=cases,
        hardware_ready=False,
        goal_complete_claimed=False,
    )
    write_json(output / "report.json", report)
    for case, result in cases.items():
        print(
            case,
            {
                h: {p: round(m["ankle_velocity_rmse_rad_s"], 4) for p, m in phases.items()}
                for h, phases in result["horizons"].items()
            },
            flush=True,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        prepare()
    else:
        if args.checkpoint is None or args.output is None:
            parser.error("--checkpoint and --output are required for evaluation")
        evaluate(args.checkpoint, args.output)
