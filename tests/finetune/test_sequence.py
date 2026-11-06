import numpy as np
import pytest


def test_quintic_matches_position_velocity_and_acceleration():
    from unisupermimic.finetune.sequence import quintic

    # x(t)=t+t², from0to2seconds, has endpoints(0,1,2)and(6,5,2).
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    actual = quintic(
        t,
        2.0,
        np.array([0.0]),
        np.array([6.0]),
        np.array([1.0]),
        np.array([5.0]),
        np.array([2.0]),
        np.array([2.0]),
    )
    np.testing.assert_allclose(actual[:, 0], t + t * t, atol=1e-10)


def test_sequence_preserves_dance_and_has_stationary_endpoints(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.deploy.bundle import DeploymentBundle
    from unisupermimic.finetune.sequence import build_sequence

    out = tmp_path / "sequence"
    build_sequence(
        adaptation_bundle,
        experiment_inputs["scene"],
        out,
        head_seconds=0.2,
        entry_seconds=0.4,
        exit_seconds=0.4,
        tail_seconds=0.2,
    )
    result = DeploymentBundle.load(out)
    seq = result.metadata["sequence"]
    a, b = seq["core_output"]
    c, d = seq["core_source"]
    for key in ("q", "root_p", "root_R"):
        np.testing.assert_allclose(
            result.reference[key][a : b + 1], adaptation_bundle.reference[key][c : d + 1], atol=2e-6
        )
    assert result.metadata["sequence"]["initial_velocity"] == "zero"
    for key in ("qd", "root_v", "root_w"):
        np.testing.assert_allclose(result.reference[key][0], 0, atol=1e-5)
        np.testing.assert_allclose(result.reference[key][-1], 0, atol=1e-5)
    assert result.metadata["hardware_ready"] is False
    with pytest.raises(FileExistsError):
        build_sequence(adaptation_bundle, experiment_inputs["scene"], out)
