import json

import numpy as np
import pytest
import torch


def test_bundle_resamples_and_preserves_future_goal_contract(bundle):
    assert bundle.reference["q"].shape == (4, 29)
    assert bundle.reference["goals"].shape == (4, 730)
    assert bundle.reference["goals"][0, 0] == pytest.approx(0.018, abs=1e-6)
    np.testing.assert_allclose(bundle.reference["goals"][0, 3:9], [1, 0, 0, 1, 0, 0])
    assert bundle.reference["goals"][-1, 0] == pytest.approx(0)
    assert bundle.metadata["control_dt"] == 0.02
    assert bundle.metadata["provenance_checked"] is True
    assert len(bundle.metadata["kp"]) == 29
    assert np.all(np.array(bundle.metadata["effort_limits"]) > 0)


def test_bundle_refuses_corrupted_model_before_loading(bundle):
    from unisupermimic.deploy.bundle import DeploymentBundle

    bundle.policy_path.write_bytes(b"bad model")
    with pytest.raises(ValueError, match="hash|digest|SHA"):
        DeploymentBundle.load(bundle.directory)


def test_export_rejects_unaccepted_reference(export_inputs, tmp_path):
    from unisupermimic.deploy.bundle import export_bundle

    report = export_inputs["acceptance_report"]
    data = json.loads(report.read_text())
    data["input_motion_sha256"] = "0" * 64
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="motion|reference"):
        export_bundle(output_dir=tmp_path / "bad", **export_inputs)


def test_export_rejects_model_from_different_checkpoint(export_inputs, tmp_path):
    from unisupermimic.deploy.bundle import export_bundle

    ckpt = export_inputs["checkpoint_path"]
    checkpoint = torch.load(ckpt, weights_only=True)
    checkpoint["pi"]["dec.2.bias"] += 0.4
    torch.save(checkpoint, ckpt)
    # Update provenance to isolate mismatched TorchScript weights.
    import hashlib
    report = export_inputs["acceptance_report"]
    data = json.loads(report.read_text())
    data["after_training"]["checkpoint_sha256"] = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="weight|checkpoint|policy"):
        export_bundle(output_dir=tmp_path / "bad", **export_inputs)


def test_export_never_overwrites_an_existing_directory(export_inputs, tmp_path):
    from unisupermimic.deploy.bundle import export_bundle

    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        export_bundle(output_dir=target, **export_inputs)
    assert sentinel.read_text() == "keep"


def test_export_refuses_wrong_traced_smoothing_even_with_identical_weights(export_inputs, tmp_path):
    from unisupermimic.deploy.bundle import export_bundle
    from unisupermimic.rl.algos.supermimic.networks import Policy

    checkpoint = torch.load(export_inputs["checkpoint_path"], weights_only=True)
    wrong = Policy(99, 730, hist=10, z_dim=4, hidden=(8,), action_smoothing=1.0).eval()
    wrong.load_state_dict(checkpoint["pi"])
    torch.jit.trace(wrong, (torch.zeros(1, 10, 99), torch.zeros(1, 730))).save(str(export_inputs["policy_path"]))
    with pytest.raises(ValueError, match="forward|equivalence|policy"):
        export_bundle(output_dir=tmp_path / "wrong_graph", **export_inputs)


def test_export_refuses_changed_motion_fps_even_when_csv_hash_is_unchanged(export_inputs, tmp_path):
    from unisupermimic.deploy.bundle import export_bundle

    sidecar = export_inputs["motion_csv"].with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    metadata["fps"] = 60
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata|fps|reference"):
        export_bundle(output_dir=tmp_path / "wrong_fps", **export_inputs)


def test_forward_parity_detects_previous_action_from_wrong_history_frame(export_inputs, tmp_path):
    from unisupermimic.deploy.bundle import export_bundle
    from unisupermimic.rl.algos.supermimic.networks import Policy

    class WrongHistoryPolicy(Policy):
        def forward(self, history, goal):
            result = super().forward(history, goal)
            # Deliberate mutation: use the oldest applied action in the filter.
            return result + 0.65*(history[:, 0, 64:93]-history[:, -1, 64:93])

    checkpoint = torch.load(export_inputs["checkpoint_path"], weights_only=True)
    wrong = WrongHistoryPolicy(99, 730, hist=10, z_dim=4, hidden=(8,), action_smoothing=0.35).eval()
    wrong.load_state_dict(checkpoint["pi"])
    torch.jit.trace(wrong, (torch.zeros(1, 10, 99), torch.zeros(1, 730))).save(str(export_inputs["policy_path"]))
    with pytest.raises(ValueError, match="forward|equivalence|policy"):
        export_bundle(output_dir=tmp_path / "wrong_history", **export_inputs)
