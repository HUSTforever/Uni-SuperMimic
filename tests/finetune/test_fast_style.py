import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def style_case(tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[2]
        / "finetune_projects/g1_continuity_20261029/projection_eval.py"
    )
    spec = importlib.util.spec_from_file_location("tested_fast_style", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q = np.sin(np.arange(1799)[:, None] * 0.02 + np.arange(29)[None, :] * 0.2)
    anchor = tmp_path / "anchor.npz"
    np.savez(anchor, q=q)
    monkeypatch.setattr(module, "ANCHOR", anchor)
    monkeypatch.setattr(module, "ANCHOR_SHA", hashlib.sha256(anchor.read_bytes()).hexdigest())
    poses = np.zeros((2039, 36))
    poses[:, 3] = 1
    poses[:1799, 7:] = q
    poses[1799:, 7:] = q[-1]
    meta = dict(
        sequence=dict(
            segments=dict(
                standing=[0, 100],
                entry=[100, 250],
                dance=[250, 1398],
                exit=[1398, 1438],
                settle=[1438, 2038],
            )
        )
    )
    return module, poses, meta


def test_fast_standing_tail_keeps_exact_original_dance_fidelity(style_case):
    module, poses, meta = style_case
    report = module.original_style_metrics(poses, metadata=meta)
    assert report["complete"] and report["passed"]
    assert report["dance_joint_rmse_rad"] == 0
    assert not module.original_style_metrics(poses)["complete"]


def test_fast_style_still_requires_full_physical_length_and_original_timing(style_case):
    module, poses, meta = style_case
    assert not module.original_style_metrics(poses[:-1], metadata=meta)["complete"]
    meta["sequence"]["segments"]["dance"] = [251, 1399]
    assert not module.original_style_metrics(poses, metadata=meta)["complete"]


def test_fast_style_does_not_relax_pose_fidelity_limits(style_case):
    module, poses, meta = style_case
    poses[250:1399, 7:] += 0.14
    report = module.original_style_metrics(poses, metadata=meta)
    assert report["complete"] and not report["passed"]
    assert report["limits"]["dance_joint_rmse_rad"] == 0.13
