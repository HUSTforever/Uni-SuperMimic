import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def experiment_inputs(tmp_path):
    from demo.export_policy import build_policy, export_metadata

    from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig
    from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import default_joint_pos_list

    cfg = SuperMimicConfig(
        hist=10,
        goal_frames=10,
        z_dim=4,
        wm_hidden=(8,),
        policy_hidden=(8,),
        critic_hidden=(8,),
        action_smoothing=0.35,
        amp="none",
        compile=False,
    )
    pi = Policy(99, 730, hist=10, z_dim=4, hidden=(8,), action_smoothing=0.35)
    ckpt = tmp_path / "source.pt"
    checkpoint = dict(
        pi=pi.state_dict(),
        wm=WorldModel(hidden=(8,)).state_dict(),
        args=asdict(cfg),
        iteration=100,
        wall_time=10.0,
        sim_steps=1000,
    )
    torch.save(checkpoint, ckpt)
    ts = tmp_path / "policy.ts"
    torch.jit.trace(build_policy(checkpoint), (torch.zeros(1, 10, 99), torch.zeros(1, 730))).save(
        str(ts)
    )
    ts.with_suffix(".ts.json").write_text(json.dumps(export_metadata(checkpoint)))
    raw = np.zeros((100, 36))
    raw[:, 0] = np.arange(100) * 0.001
    raw[:, 2] = 0.76
    raw[:, 6] = 1
    raw[:, 7:] = default_joint_pos_list()
    csv = tmp_path / "motion.csv"
    np.savetxt(csv, raw, delimiter=",")

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    metadata = json.dumps(dict(fps=50, frames=100, csv_sha256=digest(csv)))
    csv.with_suffix(".json").write_text(metadata)
    accepted = tmp_path / "accepted.csv"
    accepted.write_bytes(csv.read_bytes())
    accepted.with_suffix(".json").write_text(metadata)
    report = tmp_path / "RESULTS.json"
    report.write_text(
        json.dumps(
            dict(
                accepted_for_delivery=True,
                input_motion_sha256=digest(csv),
                after_training=dict(
                    checkpoint_sha256=digest(ckpt),
                    motions=[dict(path=str(accepted), sha256=digest(csv))],
                ),
                verification=dict(full_clip_torchscript_frames=99),
            )
        )
    )
    scene = (
        Path(__file__).resolve().parents[2]
        / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"
    )
    return dict(
        policy_path=ts, motion_csv=csv, checkpoint_path=ckpt, acceptance_report=report, scene=scene
    )


@pytest.fixture
def adaptation_bundle(experiment_inputs, tmp_path):
    from unisupermimic.deploy.bundle import DeploymentBundle, export_bundle

    export_bundle(output_dir=tmp_path / "base_bundle", **experiment_inputs)
    return DeploymentBundle.load(tmp_path / "base_bundle")
