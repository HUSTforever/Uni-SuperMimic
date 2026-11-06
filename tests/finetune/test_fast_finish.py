import numpy as np
import pytest


def test_fast_finish_preserves_main_motion_and_keeps_static_tail(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.deploy.bundle import DeploymentBundle
    from unisupermimic.finetune.finish import build_fast_finish
    from unisupermimic.finetune.sequence import build_sequence

    original = tmp_path / "original"
    build_sequence(
        adaptation_bundle,
        experiment_inputs["scene"],
        original,
        head_seconds=0.2,
        entry_seconds=0.4,
        exit_seconds=0.6,
        tail_seconds=1.0,
    )
    source = DeploymentBundle.load(original)
    out = tmp_path / "fast"
    build_fast_finish(source, experiment_inputs["scene"], out, exit_seconds=0.4, hold_seconds=1.0)
    result = DeploymentBundle.load(out)
    end = source.metadata["sequence"]["segments"]["dance"][1]
    for key in ("q", "root_p", "root_R"):
        np.testing.assert_allclose(
            result.reference[key][: end + 1], source.reference[key][: end + 1], atol=2e-5, rtol=0
        )
    assert result.metadata["sequence"]["segments"]["exit"] == [end, end + 20]
    assert result.metadata["sequence"]["segments"]["settle"] == [end + 20, end + 70]
    assert len(result.reference["q"]) == end + 71
    np.testing.assert_allclose(
        result.reference["q"][-50:],
        np.tile(result.metadata["default_joint_positions"], (50, 1)),
        atol=2e-5,
    )
    for key in ("qd", "root_v", "root_w"):
        np.testing.assert_allclose(result.reference[key][-1], 0, atol=1e-5)
    assert result.policy_path.read_bytes() == source.policy_path.read_bytes()
    assert (
        result.metadata["kp"] == source.metadata["kp"]
        and result.metadata["kd"] == source.metadata["kd"]
    )
    assert result.metadata["sequence"]["fast_finish"]["exit_seconds"] == 0.4
    assert not result.metadata["hardware_ready"]
    with pytest.raises(FileExistsError):
        build_fast_finish(source, experiment_inputs["scene"], out)


@pytest.mark.parametrize("value", [0, -1, float("nan"), 0.21])
def test_invalid_finish_duration_does_not_publish(
    adaptation_bundle, experiment_inputs, tmp_path, value
):
    from unisupermimic.finetune.finish import build_fast_finish

    with pytest.raises(ValueError):
        build_fast_finish(
            adaptation_bundle, experiment_inputs["scene"], tmp_path / "bad", exit_seconds=value
        )
    assert not (tmp_path / "bad").exists()


def finish_trace(duration=0.8):
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import default_joint_pos_list

    segments = dict(
        standing=[0, 10], entry=[10, 20], dance=[20, 40], exit=[40, 90], settle=[90, 690]
    )
    t = np.arange(2761) * 0.005
    stand = np.array(default_joint_pos_list())
    q = np.zeros((len(t), 36))
    q[:, 2] = 0.75
    q[:, 3] = 1
    q[:, 7:] = stand
    u = np.clip((t - 0.8) / duration, 0, 1)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
    arm = [15, 16, 17, 18, 22, 23, 24, 25]
    q[:, np.array(arm) + 7] += 0.8 * (1 - s[:, None])
    dq = np.zeros((len(t), 29))
    dq[:, arm] = -0.8 * ds[:, None]
    meta = dict(
        default_joint_positions=stand.tolist(),
        context_offset=[0.0] * 29,
        sequence=dict(segments=segments),
    )
    return meta, q, dq, np.zeros((len(t), 3)), t


def test_measured_quick_return_and_ten_second_hold_pass():
    from unisupermimic.finetune.finish import summarize_finish

    r = summarize_finish(*finish_trace())
    assert r["passed"] and r["complete"] and r["final_hold_passed"]
    assert 0 < r["recovery_seconds"] < 1.5
    assert r["required_hold_seconds"] == 10


def test_source_joint_violation_cannot_disappear_during_exit_resampling(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.deploy.bundle import DeploymentBundle
    from unisupermimic.finetune.finish import build_fast_finish
    from unisupermimic.finetune.sequence import build_sequence

    original = tmp_path / "original"
    build_sequence(
        adaptation_bundle,
        experiment_inputs["scene"],
        original,
        head_seconds=0.2,
        entry_seconds=0.4,
        exit_seconds=0.6,
        tail_seconds=1.0,
    )
    source = DeploymentBundle.load(original)
    start = source.metadata["sequence"]["segments"]["dance"][1]
    source.reference["q"][start + 1, 0] = source.metadata["joint_upper"][0] + 1
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="source.*joint limits"):
        build_fast_finish(
            source, experiment_inputs["scene"], output, exit_seconds=0.2, hold_seconds=1.0
        )
    assert not output.exists()


def test_slow_actual_return_fails_even_if_eventually_stable():
    from unisupermimic.finetune.finish import summarize_finish

    r = summarize_finish(*finish_trace(3.0))
    assert r["final_hold_passed"] and not r["passed"]
    assert r["recovery_seconds"] > 1.5


def test_late_instability_or_missing_terminal_cannot_pass():
    from unisupermimic.finetune.finish import summarize_finish

    meta, q, dq, v, t = finish_trace()
    v[-50:, 0] = 0.25
    assert not summarize_finish(meta, q, dq, v, t)["passed"]
    assert not summarize_finish(meta, q[:-1], dq[:-1], v[:-1], t[:-1])["complete"]


def test_brief_standing_before_another_wobble_does_not_count_as_recovered():
    from unisupermimic.finetune.finish import summarize_finish

    meta, q, dq, v, t = finish_trace()
    v[(t >= 2.0) & (t <= 2.5), 0] = 0.25
    q[:, 0] = np.cumsum(v[:, 0]) * 0.005
    r = summarize_finish(meta, q, dq, v, t)
    assert r["final_hold_passed"]
    assert not r["passed"]
    assert r["recovery_seconds"] >= 1.7
