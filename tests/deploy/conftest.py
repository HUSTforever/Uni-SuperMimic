from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from demo.export_policy import build_policy, export_metadata

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"


@pytest.fixture
def export_inputs(tmp_path):
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import default_joint_pos_list

    args = dict(hist=10, goal_frames=10, z_dim=4, policy_hidden=[8], action_smoothing=0.35)
    from unisupermimic.rl.algos.supermimic.networks import Policy

    policy = Policy(99, 730, hist=10, z_dim=4, hidden=(8,), action_smoothing=0.35).eval()
    checkpoint = dict(pi=policy.state_dict(), args=args)
    ckpt = tmp_path / "ckpt_000001.pt"
    torch.save(checkpoint, ckpt)
    ts = tmp_path / "policy.ts"
    torch.jit.trace(build_policy(checkpoint), (torch.zeros(1, 10, 99), torch.zeros(1, 730))).save(str(ts))
    ts.with_suffix(".ts.json").write_text(json.dumps(export_metadata(checkpoint)))
    raw = np.zeros((3, 36))
    raw[:, 0] = [0, 0.03, 0.06]
    raw[:, 2] = 0.76
    raw[:, 6] = 1
    raw[:, 7:] = default_joint_pos_list()
    csv = tmp_path / "reference.csv"
    np.savetxt(csv, raw, delimiter=",")
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    csv.with_suffix(".json").write_text(json.dumps(dict(fps=30, frames=3, csv_sha256=digest(csv))))
    accepted_csv = tmp_path / "accepted_reference.csv"
    shutil.copyfile(csv, accepted_csv)
    shutil.copyfile(csv.with_suffix(".json"), accepted_csv.with_suffix(".json"))
    accepted = tmp_path / "RESULTS.json"
    accepted.write_text(json.dumps(dict(
        accepted_for_delivery=True,
        selected_checkpoint=str(ckpt),
        input_motion_sha256=digest(csv),
        after_training=dict(checkpoint_sha256=digest(ckpt),
                            motions=[dict(path=str(accepted_csv), sha256=digest(csv))]),
        verification=dict(full_clip_torchscript_frames=3),
    )))
    return dict(policy_path=ts, motion_csv=csv, checkpoint_path=ckpt,
                acceptance_report=accepted, scene=SCENE)


@pytest.fixture
def bundle(export_inputs, tmp_path):
    from unisupermimic.deploy.bundle import DeploymentBundle, export_bundle

    directory = tmp_path / "bundle"
    export_bundle(output_dir=directory, **export_inputs)
    return DeploymentBundle.load(directory)
