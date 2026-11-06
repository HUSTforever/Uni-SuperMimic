import numpy as np
import pytest


def signal_trace():
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES

    t = np.arange(5601) * 0.005
    q = np.zeros((len(t), 36))
    q[:, 2] = 0.75
    q[:, 3] = 1
    q[:, 11] = 0.1 * np.sin(2 * np.pi * 2 * t)
    meta = dict(
        joint_names=G1_JOINT_NAMES,
        sequence=dict(
            segments=dict(
                standing=[0, 100],
                entry=[100, 200],
                dance=[200, 1200],
                exit=[1200, 1300],
                settle=[1300, 1400],
            )
        ),
    )
    return meta, t, q, np.gradient(q[:, 7:], 0.005, axis=0), np.zeros((len(t), 3))


def summarize(meta, t, q, dq, v):
    from unisupermimic.finetune.continuity_metrics import summarize_continuity

    return summarize_continuity(meta, q, dq, v, t)


def test_spectral_rms_is_in_physical_units_and_ignores_slow_motion():
    from unisupermimic.finetune.continuity_metrics import high_frequency_rms

    t = np.arange(400) * 0.005
    assert high_frequency_rms(0.01 * np.sin(2 * np.pi * 10 * t), t)[0] == pytest.approx(
        0.01 / np.sqrt(2), rel=0.002
    )
    assert high_frequency_rms(0.1 * np.sin(2 * np.pi * 2 * t), t)[0] < 1e-5


def test_low_frequency_motion_passes_and_windows_cover_both_endpoints():
    m, t, q, dq, v = signal_trace()
    r = summarize(m, t, q, dq, v)
    assert r["complete"] and r["passed"]
    assert all(x["passed"] for x in r["phases"].values())
    assert r["windows"][0]["start_time"] == 0
    assert r["windows"][-1]["end_time"] == pytest.approx(t[-1])


def test_one_local_burst_cannot_hide_in_phase_average():
    m, t, q, dq, v = signal_trace()
    mask = (t >= 13.5) & (t <= 14.5)
    q[mask, 11] += 0.008 * np.sin(2 * np.pi * 10 * t[mask])
    dq = np.gradient(q[:, 7:], 0.005, axis=0)
    r = summarize(m, t, q, dq, v)
    assert r["phases"]["dance"]["ankle_hf_position_max_rad"] < 0.0025
    assert not r["passed"] and not r["worst_ankle_window"]["passed"]
    assert r["worst_ankle_window"]["ankle_hf_position_max_rad"] > 0.005


@pytest.mark.parametrize("at_start", [True, False])
def test_shaking_at_start_or_end_is_not_exempted(at_start):
    m, t, q, dq, v = signal_trace()
    mask = t < 2 if at_start else t > 26
    q[mask, 11] += 0.02 * np.sin(2 * np.pi * 10 * t[mask])
    dq = np.gradient(q[:, 7:], 0.005, axis=0)
    r = summarize(m, t, q, dq, v)
    assert not r["passed"]
    assert not r["phases"]["standing" if at_start else "settle"]["passed"]


def test_missing_terminal_or_wrong_clock_cannot_pass():
    m, t, q, dq, v = signal_trace()
    r = summarize(m, t[:-1], q[:-1], dq[:-1], v[:-1])
    assert not r["complete"] and not r["passed"]
    wrong = t.copy()
    wrong[30] = wrong[29]
    with pytest.raises(ValueError, match="time|clock|sample"):
        summarize(m, wrong, q, dq, v)
    with pytest.raises(ValueError, match="200|sample"):
        summarize(m, t[::4], q[::4], dq[::4], v[::4])


def test_missing_named_phase_cannot_shrink_the_evaluation():
    m, t, q, dq, v = signal_trace()
    m["sequence"]["segments"].pop("settle")
    keep = t <= 26
    with pytest.raises(ValueError, match="phase"):
        summarize(m, t[keep], q[keep], dq[keep], v[keep])


def test_overlapping_phases_and_nonfinite_state_are_rejected():
    m, t, q, dq, v = signal_trace()
    m["sequence"]["segments"]["entry"] = [50, 200]
    with pytest.raises(ValueError, match="phase"):
        summarize(m, t, q, dq, v)
    m, t, q, dq, v = signal_trace()
    q[20, 11] = np.nan
    with pytest.raises(ValueError, match="finite"):
        summarize(m, t, q, dq, v)


@pytest.mark.parametrize("at_start", [True, False])
@pytest.mark.parametrize("duration", [0.1, 0.2])
def test_short_smooth_endpoint_burst_cannot_hide_under_hann_window(at_start, duration):
    m, t, q, dq, v = signal_trace()
    tau = t if at_start else t[-1] - t
    mask = tau <= duration + 1e-9
    q[mask, 11] += (
        0.1 * np.sin(2 * np.pi * 10 * tau[mask]) * np.sin(np.pi * tau[mask] / duration) ** 2
    )
    dq = np.gradient(q[:, 7:], 0.005, axis=0)
    r = summarize(m, t, q, dq, v)
    assert r["complete"]
    assert not r["passed"]
    assert r["worst_ankle_window"]["ankle_hf_position_max_rad"] > 0.005


@pytest.mark.parametrize("phase", [0.0, 0.4, 1.7])
def test_slow_endpoint_motion_does_not_become_a_reflection_artifact(phase):
    m, t, q, dq, v = signal_trace()
    q[:, 11] = 0.1 * np.sin(2 * np.pi * 2 * t + phase)
    q[:, 2] += 0.02 * np.sin(2 * np.pi * 2 * t + phase)
    dq = np.gradient(q[:, 7:], 0.005, axis=0)
    v = np.gradient(q[:, :3], 0.005, axis=0)
    assert summarize(m, t, q, dq, v)["passed"]


@pytest.mark.parametrize("at_start", [True, False])
def test_short_root_bounce_at_endpoint_is_detected(at_start):
    m, t, q, dq, v = signal_trace()
    tau = t if at_start else t[-1] - t
    mask = tau <= 0.2 + 1e-9
    q[mask, 2] += 0.01 * np.sin(2 * np.pi * 10 * tau[mask]) * np.sin(np.pi * tau[mask] / 0.2) ** 2
    v = np.gradient(q[:, :3], 0.005, axis=0)
    r = summarize(m, t, q, dq, v)
    assert not r["passed"]
    assert r["worst_root_window"]["root_vertical_hf_position_rms_m"] > 0.001


def test_short_valid_windows_keep_enough_boundary_samples_for_spectrum():
    from unisupermimic.finetune.continuity_metrics import ContinuityLimits, summarize_continuity

    m, t, q, dq, v = signal_trace()
    q[:, 11] = 0
    dq[:] = 0
    r = summarize_continuity(
        m, q, dq, v, t, limits=ContinuityLimits(window_seconds=0.08, hop_seconds=0.04)
    )
    assert r["passed"]
