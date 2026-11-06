"""Full-trajectory high-rate continuity diagnostics; never modify the trajectory."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES

ANKLES = tuple(i for i, name in enumerate(G1_JOINT_NAMES) if "ankle" in name)
CONTINUITY_METRICS_VERSION = 2


@dataclass(frozen=True)
class ContinuityLimits:
    frequency_hz: float = 8.0
    phase_ankle_angle_rms: float = 0.0025
    window_ankle_angle_rms: float = 0.005
    phase_root_z_rms: float = 0.0005
    window_root_z_rms: float = 0.001
    window_seconds: float = 2.0
    hop_seconds: float = 0.5

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("continuity limits must be finite and positive")
        if self.hop_seconds > self.window_seconds or self.window_seconds < 0.08:
            raise ValueError("invalid continuity window/hop")


def _clock(sample_time):
    t = np.asarray(sample_time, dtype=float)
    if t.ndim != 1 or len(t) < 16 or not np.isfinite(t).all():
        raise ValueError("at least 16 finite sample times required")
    delta = np.diff(t)
    if (delta <= 0).any() or not np.allclose(delta, delta[0], rtol=1e-5, atol=1e-8):
        raise ValueError("sample clock must be strictly increasing and uniform")
    return t, float(delta.mean())


def high_frequency_rms(values, sample_time, cutoff_hz=8.0):
    """Detrend, Hann-window and integrate one-sided PSD in physical value units."""
    t, dt = _clock(sample_time)
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) != len(t) or not np.isfinite(x).all():
        raise ValueError("finite sample x channel values required")
    if not math.isfinite(cutoff_hz) or not 0 < cutoff_hz < 0.5 / dt:
        raise ValueError("frequency cutoff must lie below sample Nyquist frequency")
    n = len(t)
    centered_t = t - t.mean()
    centered = x - x.mean(0)
    centered -= centered_t[:, None] * ((centered_t @ centered) / (centered_t @ centered_t))[None, :]
    window = np.hanning(n)
    spectrum = np.fft.rfft(centered * window[:, None], axis=0)
    frequencies = np.fft.rfftfreq(n, dt)
    factor = np.full(len(frequencies), 2.0)
    factor[0] = 1
    if n % 2 == 0:
        factor[-1] = 1
    selected = frequencies > cutoff_hz
    power = (np.abs(spectrum[selected]) ** 2 * factor[selected, None]).sum(0)
    return np.sqrt(power / (n * np.sum(window**2)))


def summarize_continuity(
    metadata, qpos, joint_velocity, root_velocity, sample_time, *, limits=None
):
    """Necessary anti-chatter gates across every phase and overlapping full-time windows.

    A passing result is not a universal balance proof. It must be combined with
    physical completion, motion fidelity, robustness and raw-video inspection.
    """
    limits = ContinuityLimits() if limits is None else limits
    t, dt = _clock(sample_time)
    if not math.isclose(dt, 0.005, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("continuity gates require complete 200Hz physics samples")
    q, dq, v = (np.asarray(x, dtype=float) for x in (qpos, joint_velocity, root_velocity))
    for x, shape in ((q, (len(t), 36)), (dq, (len(t), 29)), (v, (len(t), 3))):
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError(
                "finite physical trajectory arrays with matching sample counts required"
            )
    segments = metadata["sequence"]["segments"]
    phase_names = ("standing", "entry", "dance", "exit", "settle")
    if not isinstance(segments, dict) or set(segments) != set(phase_names):
        raise ValueError("all five named sequence phases are required")
    previous_end = 0
    for name in phase_names:
        pair = segments[name]
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(type(x) is not int for x in pair)
            or pair[0] != previous_end
            or pair[1] <= pair[0]
        ):
            raise ValueError(
                "sequence phase bounds must be positive and contiguous from frame zero"
            )
        previous_end = pair[1]
    if tuple(metadata.get("joint_names", ())) != tuple(G1_JOINT_NAMES):
        raise ValueError("canonical G1 joint order required")
    expected_end = max(b for a, b in segments.values()) * 0.02
    expected_samples = round(expected_end / dt) + 1
    complete = len(t) == expected_samples and abs(t[0]) < 1e-7 and abs(t[-1] - expected_end) < 1e-7
    joint_acceleration = np.gradient(dq, t, axis=0)
    joint_jerk = np.gradient(joint_acceleration, t, axis=0)
    root_acceleration = np.gradient(v, t, axis=0)

    def describe(indices, ankle_limit, root_limit, complete_segment=True, reflect_at=None):
        if len(indices) < 16:
            return dict(passed=False, completed=False, samples=int(len(indices)))
        times = t[indices]

        def spectrum(values):
            if reflect_at is None:
                return high_frequency_rms(values, times, limits.frequency_hz)
            # Center a global endpoint under the Hann maximum. Odd reflection
            # preserves the endpoint value and first finite difference, avoiding
            # the velocity cusp of even reflection on ordinary slow motion.
            # Only spectral diagnostics use this extension; recorded states and
            # acceleration/jerk metrics below always use actual physics samples.
            padding = (len(indices) - 1, 0) if reflect_at == "start" else (0, len(indices) - 1)
            values = np.asarray(values).reshape(len(indices), -1)
            extended = np.pad(values, (padding, (0, 0)), mode="reflect", reflect_type="odd")
            return high_frequency_rms(extended, np.arange(len(extended)) * dt, limits.frequency_hz)

        hfq = spectrum(q[indices, 7:])[list(ANKLES)]
        hfv = spectrum(dq[indices])[list(ANKLES)]
        hfz = float(spectrum(q[indices, 2])[0])
        return dict(
            spectral_extension="none" if reflect_at is None else f"odd_reflection_{reflect_at}",
            completed=bool(complete_segment),
            samples=int(len(indices)),
            start_time=float(times[0]),
            end_time=float(times[-1]),
            ankle_hf_position_rms_rad=hfq.tolist(),
            ankle_hf_velocity_rms_rad_s=hfv.tolist(),
            ankle_hf_position_max_rad=float(hfq.max()),
            ankle_hf_velocity_max_rad_s=float(hfv.max()),
            root_vertical_hf_position_rms_m=hfz,
            root_vertical_acceleration_rms=float(
                np.sqrt(np.mean(root_acceleration[indices, 2] ** 2))
            ),
            root_vertical_acceleration_peak=float(np.abs(root_acceleration[indices, 2]).max()),
            joint_acceleration_rms=float(np.sqrt(np.mean(joint_acceleration[indices] ** 2))),
            joint_jerk_rms=float(np.sqrt(np.mean(joint_jerk[indices] ** 2))),
            ankle_passed=bool(complete_segment and hfq.max() <= ankle_limit),
            root_vertical_passed=bool(complete_segment and hfz <= root_limit),
            passed=bool(complete_segment and hfq.max() <= ankle_limit and hfz <= root_limit),
        )

    phases = {}
    for name, (a, b) in segments.items():
        start, end = a * 0.02, b * 0.02
        indices = np.flatnonzero((t >= start - 1e-7) & (t <= end + 1e-7))
        covered = bool(
            len(indices) and abs(t[indices[0]] - start) < 1e-7 and abs(t[indices[-1]] - end) < 1e-7
        )
        phases[name] = describe(
            indices, limits.phase_ankle_angle_rms, limits.phase_root_z_rms, covered
        )
    width = min(round(limits.window_seconds / dt) + 1, len(t))
    hop = max(1, round(limits.hop_seconds / dt))
    last_start = len(t) - width
    starts = sorted(set(range(0, last_start + 1, hop)) | {last_start})
    windows = [
        describe(np.arange(i, i + width), limits.window_ankle_angle_rms, limits.window_root_z_rms)
        for i in starts
    ]
    # Ordinary overlapping windows cannot center the global endpoints. Add
    # dedicated half-window observations with reflected spectral padding, so a
    # short startup/terminal burst is not hidden under every Hann window edge.
    boundary_width = min(max(16, round(limits.window_seconds / (2 * dt)) + 1), len(t))
    windows.insert(
        0,
        describe(
            np.arange(boundary_width),
            limits.window_ankle_angle_rms,
            limits.window_root_z_rms,
            reflect_at="start",
        ),
    )
    windows.append(
        describe(
            np.arange(len(t) - boundary_width, len(t)),
            limits.window_ankle_angle_rms,
            limits.window_root_z_rms,
            reflect_at="end",
        )
    )
    worst_ankle = max(windows, key=lambda x: x["ankle_hf_position_max_rad"])
    worst_root = max(windows, key=lambda x: x["root_vertical_hf_position_rms_m"])
    return dict(
        metrics_version=CONTINUITY_METRICS_VERSION,
        complete=bool(complete),
        expected_samples=expected_samples,
        recorded_samples=len(t),
        physics_hz=1 / dt,
        limits=asdict(limits),
        phases=phases,
        windows=windows,
        worst_ankle_window=worst_ankle,
        worst_root_window=worst_root,
        passed=bool(
            complete
            and all(x["passed"] for x in phases.values())
            and all(x["passed"] for x in windows)
        ),
    )
