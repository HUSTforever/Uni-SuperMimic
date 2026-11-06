"""Unit tests for ``demo/export_policy.py``."""

from __future__ import annotations

import json
import sys

import pytest
import torch
from demo import export_policy

from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import (
    GOAL_FRAME_DIM,
    NUM_J,
    OBS_DIM,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
    G1_JOINT_NAMES,
    action_scale,
    default_joint_pos_list,
)


class _TinyPolicy(torch.nn.Module):
    """Minimal policy stand-in so ``main`` can exercise real tracing/saving."""

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del goal
        return obs[:, -1, :NUM_J]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_forward_does_not_publish_a_model(tmp_path, monkeypatch, value):
    class NonfinitePolicy(torch.nn.Module):
        def forward(self, obs, goal):
            return obs[:, -1, :29] * value

    checkpoint = tmp_path / "ckpt.pt"
    torch.save(_checkpoint(), checkpoint)
    output = tmp_path / "bad.ts"
    monkeypatch.setattr(export_policy, "build_policy", lambda _: NonfinitePolicy())
    monkeypatch.setattr(sys, "argv", ["export", "--ckpt", str(checkpoint), "--output", str(output)])
    with pytest.raises(ValueError, match="finite"):
        export_policy.main()
    assert not output.exists()


def _checkpoint(hist: int = 10, goal_frames: int = 8) -> dict:
    return {"args": {"hist": hist, "goal_frames": goal_frames}}


def test_export_metadata_exactly_matches_robot_constants_and_checkpoint_dims() -> None:
    metadata = export_policy.export_metadata(_checkpoint(hist=6, goal_frames=7))

    assert metadata == {
        "schema_version": 1,
        "output_semantics": "raw_action",
        "joint_names": G1_JOINT_NAMES,
        "default_joint_positions": default_joint_pos_list(),
        "action_scales": [action_scale(name) for name in G1_JOINT_NAMES],
        "control_dt": 0.02,
        "action_smoothing": 1.0,
        "obs_dim": OBS_DIM,
        "hist": 6,
        "goal_frames": 7,
        "goal_dim": GOAL_FRAME_DIM * 7,
        "action_dim": NUM_J,
        "action_target_formula": "default_q + context_offset + action_scale * raw_action",
    }
    assert metadata["obs_dim"] == 99
    assert metadata["action_dim"] == len(G1_JOINT_NAMES) == 29
    assert metadata["goal_dim"] == 73 * 7


@pytest.mark.parametrize(("hist", "goal_frames"), [(1, 2), (5, 16), (10, 8)])
def test_export_metadata_uses_checkpoint_dynamic_dimensions(hist: int, goal_frames: int) -> None:
    metadata = export_policy.export_metadata(_checkpoint(hist=hist, goal_frames=goal_frames))

    assert metadata["hist"] == hist
    assert metadata["goal_frames"] == goal_frames
    assert metadata["goal_dim"] == GOAL_FRAME_DIM * goal_frames


def test_main_writes_metadata_json_beside_torchscript(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint_path = tmp_path / "ckpt.pt"
    output_path = tmp_path / "policy.ts"
    checkpoint = _checkpoint(hist=3, goal_frames=5)
    torch.save(checkpoint, checkpoint_path)
    monkeypatch.setattr(export_policy, "build_policy", lambda _checkpoint: _TinyPolicy())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_policy.py",
            "--ckpt",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--batch",
            "2",
        ],
    )

    export_policy.main()
    capsys.readouterr()

    assert output_path.is_file()
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    assert metadata_path.is_file()
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == export_policy.export_metadata(
        checkpoint
    )
