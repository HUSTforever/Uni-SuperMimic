import numpy as np
import pytest


def test_metrics_distinguish_sliding_without_confusing_yaw_with_tilt(
    adaptation_bundle, experiment_inputs
):
    from unisupermimic.finetune.balance_metrics import balance_metrics

    b = adaptation_bundle
    qpos = np.concatenate(
        [b.reference["root_p"][:60], b.reference["root_quat"][:60], b.reference["q"][:60]], axis=1
    )
    qpos[:, :2] = qpos[0, :2]
    t = np.arange(60) * 0.02
    stationary = balance_metrics(b, experiment_inputs["scene"], qpos, t)
    assert stationary["near_ground_point_speed_rms_m_s"] < 1e-6
    assert stationary["gravity_error_rms_deg"] < 0.01
    assert stationary["reference_stance_point_speed_rms_m_s"] < 1e-6
    qpos[:, 0] += t * 0.5
    sliding = balance_metrics(b, experiment_inputs["scene"], qpos, t)
    assert sliding["near_ground_point_speed_rms_m_s"] == pytest.approx(0.5, abs=1e-6)
    assert sliding["reference_stance_point_speed_rms_m_s"] == pytest.approx(0.5, abs=1e-6)
    assert sliding["gravity_error_rms_deg"] < 0.01
    with pytest.raises(ValueError, match="time"):
        balance_metrics(b, experiment_inputs["scene"], qpos, np.zeros(60))


@pytest.mark.parametrize("axis", [0, 2])
def test_reference_stance_uses_recorded_velocity_at_subframe_samples(
    adaptation_bundle, experiment_inputs, axis
):
    from unisupermimic.finetune.balance_metrics import balance_metrics

    b = adaptation_bundle
    # Metadata can encode physically moving feet even when pose samples happen to repeat.
    # Stance must use the trusted velocity, not staircase differentiation after lookup.
    b.reference["root_v"][:, axis] = 0.5
    t = np.arange(60) * 0.005
    frame = np.rint(t / 0.02).astype(int)
    qpos = np.concatenate(
        [b.reference["root_p"][frame], b.reference["root_quat"][frame], b.reference["q"][frame]],
        axis=1,
    )
    qpos[:, :2] = qpos[0, :2]
    r = balance_metrics(b, experiment_inputs["scene"], qpos, t)
    assert r["reference_stance_point_speed_rms_m_s"] is None
