"""Fixed-policy WM recalibration with identical data and minibatches in both arms."""

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.run_sim2sim import write_json
from unisupermimic.finetune.collector import DeploymentCollector
from unisupermimic.finetune.learner import DeploymentLearner
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig

PROJECT = Path("finetune_projects/g1_continuity_20261029")
SOURCE = PROJECT / "run_003_reproduce/last.pt"


def snapshot(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def unchanged(module, before):
    return all(torch.equal(value, before[key]) for key, value in module.state_dict().items())


def main():
    torch.set_num_threads(1)
    outputs = {name: PROJECT / f"run_005_wm_{name}" for name in ("control", "weighted", "data")}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(path)
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
    allowed = {field.name for field in fields(SuperMimicConfig)}
    raw = {key: value for key, value in checkpoint["args"].items() if key in allowed}
    for key in ("wm_hidden", "policy_hidden", "critic_hidden", "test_hidden_dims"):
        if raw.get(key) is not None:
            raw[key] = tuple(raw[key])
    cfg = replace(
        SuperMimicConfig(**raw),
        iterations=600,
        buffer_steps=600,
        wm_window=8,
        wm_batch=512,
        wm_batches=1,
        lr=1e-5,
        compile=True,
        seed=0,
        log_backend="no_print",
        warmup_steps=64,
    )
    bundle = DeploymentBundle.load(PROJECT / "data/damping_kd200")
    env = DeploymentCollector(
        bundle,
        "deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml",
        num_envs=64,
        seed=0,
        randomize=True,
        balance_weight=1,
        ground_slip_weight=0.35,
        ankle_rate_weight=0.05,
        focus_intervals=((100, 250), (300, 700), (1000, 1400)),
    )
    report = dict(
        method="fixed_policy_world_model_recalibration",
        source_checkpoint=str(SOURCE),
        source_checkpoint_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        fit_seed=20261030,
        collection_seed=0,
        wm_updates_per_arm=600,
        iteration_semantics="checkpoint iteration counts WM updates; no policy updates",
        branches={},
        hardware_ready=False,
        goal_complete_claimed=False,
    )
    common_buffer = None
    try:
        for name, ankle, root in (("control", 1.0, 1.0), ("weighted", 4.0, 2.0)):
            learner = DeploymentLearner(
                env,
                cfg,
                outputs[name],
                device="cuda",
                backend="deployment_mujoco",
                wm_ankle_velocity_weight=ankle,
                wm_root_velocity_weight=root,
            )
            learner.initialize_weights(SOURCE)
            for key, value in learner.wm.state_dict().items():
                assert torch.equal(value.cpu(), checkpoint["wm"][key]), key
            learner.wm.in_norm.max_count = learner.wm.out_norm.max_count = 0
            learner.pi.requires_grad_(False)
            learner.critic.requires_grad_(False)
            modules = dict(
                pi=learner.pi,
                critic=learner.critic,
                wm_in_norm=learner.wm.in_norm,
                wm_out_norm=learner.wm.out_norm,
            )
            before = {key: snapshot(module) for key, module in modules.items()}
            if common_buffer is None:
                learner.collect(600)
                common_buffer = learner.buffer
                outputs["data"].mkdir(parents=True, exist_ok=False)
                dataset = outputs["data"] / "replay_dataset.pt"
                torch.save(
                    dict(
                        data={
                            key: value.detach().cpu() for key, value in common_buffer.data.items()
                        },
                        T=common_buffer.T,
                        N=common_buffer.N,
                        ptr=common_buffer.ptr,
                        size=common_buffer.size,
                    ),
                    dataset,
                )
                report["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
                report["dataset"] = str(dataset)
                report["phase_transition_counts"] = {
                    phase: int(
                        ((common_buffer.data["f"] >= a) & (common_buffer.data["f"] < b)).sum()
                    )
                    for phase, (a, b) in bundle.metadata["sequence"]["segments"].items()
                }
            else:
                learner.buffer = common_buffer
            torch.manual_seed(report["fit_seed"])
            trace = []
            for update in range(1, 601):
                loss, error = learner.wm_update()
                if update % 100 == 0:
                    row = dict(arm=name, wm_updates=update, loss=float(loss), wm_err8=float(error))
                    trace.append(row)
                    print(json.dumps(row), flush=True)
                if update in (200, 400, 600):
                    learner.iteration = update
                    learner.save(
                        outputs[name] / f"wm_{update:06d}.pt", include_optimizers=update == 600
                    )
            proof = {key: unchanged(module, before[key]) for key, module in modules.items()}
            if not all(proof.values()):
                raise RuntimeError(f"Frozen components changed: {proof}")
            report["branches"][name] = dict(
                ankle_velocity_weight=ankle,
                root_velocity_weight=root,
                complete_checkpoint=str(outputs[name] / "wm_000600.pt"),
                unchanged=proof,
                trace=trace,
            )
            write_json(PROJECT / "contact_wm_fit.json", report)
            del learner
            torch.cuda.empty_cache()
    finally:
        env.close()


if __name__ == "__main__":
    main()
