import hashlib
import json

import pytest
import torch


def test_candidate_export_binds_new_checkpoint_without_claiming_delivery(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.deploy.bundle import DeploymentBundle
    from unisupermimic.finetune.export_candidate import export_candidate

    target = tmp_path / "candidate"
    export_candidate(adaptation_bundle, experiment_inputs["checkpoint_path"], target)
    result = DeploymentBundle.load(target)
    assert result.metadata["candidate_status"] == "unvalidated"
    assert result.metadata["accepted_for_delivery"] is False
    assert result.metadata["hardware_ready"] is False
    assert (
        result.metadata["source_checkpoint_sha256"]
        == hashlib.sha256(experiment_inputs["checkpoint_path"].read_bytes()).hexdigest()
    )
    assert (target / "reference.npz").read_bytes() == (
        adaptation_bundle.directory / "reference.npz"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        export_candidate(adaptation_bundle, experiment_inputs["checkpoint_path"], target)


def test_candidate_export_rejects_changed_policy_contract(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.finetune.export_candidate import export_candidate

    path = experiment_inputs["checkpoint_path"]
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["args"]["action_smoothing"] = 1.0
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="contract|metadata|smoothing"):
        export_candidate(adaptation_bundle, path, tmp_path / "wrong")
