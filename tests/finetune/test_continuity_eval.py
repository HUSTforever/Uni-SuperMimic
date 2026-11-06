import numpy as np


def test_continuity_evaluation_keeps_every_physical_step_and_fails_closed(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.deploy.bundle import DeploymentBundle
    from unisupermimic.finetune.continuity_eval import evaluate_continuity
    from unisupermimic.finetune.sequence import build_sequence

    build_sequence(
        adaptation_bundle,
        experiment_inputs["scene"],
        tmp_path / "sequence",
        head_seconds=0.2,
        entry_seconds=0.4,
        exit_seconds=0.4,
        tail_seconds=1.0,
    )
    b = DeploymentBundle.load(tmp_path / "sequence")
    out = tmp_path / "evaluation"
    r = evaluate_continuity(b, experiment_inputs["scene"], out)
    with np.load(out / "physics_trajectory.npz") as z, np.load(out / "trajectory.npz") as control:
        assert len(z["sample_time"]) == round(z["sample_time"][-1] / 0.005) + 1
        np.testing.assert_array_equal(z["qpos"][::4], control["qpos"])
        np.testing.assert_allclose(np.diff(z["sample_time"]), 0.005, atol=1e-10)
    assert r["full_sequence_and_continuity_passed"] == bool(
        r["all_attempted_phases_passed"]
        and r["continuity"]["passed"]
        and r["continuity_fidelity_passed"]
    )
    assert r["goal_complete_claimed"] is False and r["hardware_ready"] is False
