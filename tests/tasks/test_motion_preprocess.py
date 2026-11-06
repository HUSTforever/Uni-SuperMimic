"""Regression tests for physical motion conversion and timestamp handling."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

SCENE = (
    Path(__file__).resolve().parents[2] / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"
)


def standing(frames=61):
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import default_joint_pos_list

    rows = np.zeros((frames, 36))
    rows[:, 2] = 0.76
    rows[:, 3] = 1.0
    rows[:, 7:] = default_joint_pos_list()
    return rows


def prepare(rows, **kwargs):
    from unisupermimic.tasks.motion_tracking.g1.supermimic.preprocess import prepare_motion

    return prepare_motion(rows, kwargs.pop("src_fps", 30.0), model_path=SCENE, **kwargs)


def test_world_yaw_rotates_tilt_in_world_frame():
    rows = standing()
    rows[:, 3:7] = [np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0]
    csv, _ = prepare(rows, face_yaw_deg=90)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, csv[0, [6, 3, 4, 5]])
    # A 30 degree roll, followed by a world Z turn of 90 degrees.
    expected = np.array([[0, -np.sqrt(3) / 2, 0.5], [1, 0, 0], [0, 0.5, np.sqrt(3) / 2]])
    np.testing.assert_allclose(rotation.reshape(3, 3), expected, atol=1e-7)


def test_retiming_uses_input_timestamps_and_keeps_endpoints():
    rows = standing(61)
    rows[:, 0] = np.linspace(0, 1, 61)
    csv, meta = prepare(rows, src_fps=60, slowdown=1.5, output_fps=30)
    assert csv.shape == (46, 36)
    assert meta["fps"] == 30
    assert meta["duration_seconds"] == 1.5
    np.testing.assert_allclose(csv[[0, -1], 0], [0, 1])


def test_smoothing_removes_noise_without_erasing_main_motion():
    rows = standing(121)
    t = np.arange(121) / 30
    rows[:, 15] = 0.3 * np.sin(2 * np.pi * 0.5 * t) + 0.035 * (-1.0) ** np.arange(121)
    csv, _ = prepare(rows, smooth_seconds=0.08)
    raw_acc = np.diff(rows[:, 15], n=2)
    new_acc = np.diff(csv[:, 15], n=2)
    assert np.sqrt(np.mean(new_acc**2)) < 0.2 * np.sqrt(np.mean(raw_acc**2))
    assert np.ptp(csv[:, 15]) > 0.5


def test_quaternion_sign_flips_do_not_change_smoothed_pose():
    rows = standing()
    rows[1::2, 3:7] *= -1
    csv, _ = prepare(rows, smooth_seconds=0.08)
    np.testing.assert_allclose(np.linalg.norm(csv[:, 3:7], axis=1), 1, atol=1e-8)
    np.testing.assert_allclose(np.abs(csv[:, 6]), 1, atol=1e-8)


def test_grounding_uses_wxyz_and_actual_foot_collision_geometry():
    from unisupermimic.tasks.motion_tracking.g1.supermimic.preprocess import foot_heights

    rows = standing()
    rows[:, 3:7] = [np.cos(0.05), np.sin(0.05), 0, 0]
    rows[:, 2] -= 0.12
    csv, meta = prepare(rows, ground=True)
    model_qpos = csv[:, [0, 1, 2, 6, 3, 4, 5, *range(7, 36)]]
    minimum = foot_heights(model_qpos, SCENE).min(axis=1)
    assert np.max(np.abs(minimum)) < 0.002
    assert meta["grounded"] is True


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_fps_is_rejected(bad):
    with pytest.raises(ValueError, match="fps"):
        prepare(standing(), src_fps=bad)


def test_nan_and_zero_quaternion_are_rejected():
    rows = standing()
    rows[4, 9] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prepare(rows)
    rows = standing()
    rows[3, 3:7] = 0
    with pytest.raises(ValueError, match="quaternion"):
        prepare(rows)


def test_grounding_preserves_flight_between_support_frames():
    rows = standing(121)
    rows[30:91, 2] += 0.2 * np.sin(np.linspace(0, np.pi, 61))
    csv, _ = prepare(rows, ground=True, preserve_flight=True)
    assert csv[60, 2] - csv[0, 2] > 0.15


def test_explicit_support_mode_keeps_dance_feet_grounded():
    from unisupermimic.tasks.motion_tracking.g1.supermimic.preprocess import foot_heights

    rows = standing(121)
    rows[:, 2] += np.linspace(0, 0.2, 121)
    csv, _ = prepare(rows, ground=True, preserve_flight=False)
    heights = foot_heights(csv[:, [0, 1, 2, 6, 3, 4, 5, *range(7, 36)]], SCENE)
    assert np.abs(heights.min(axis=1)).max() < 0.002


def test_motion_loader_uses_per_file_fps_metadata(tmp_path):
    import json

    from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
    from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib

    rows = standing(5)[:, [0, 1, 2, 4, 5, 6, 3, *range(7, 36)]]
    path = tmp_path / "ten_hz.csv"
    np.savetxt(path, rows, delimiter=",")
    path.with_suffix(".json").write_text(
        json.dumps({"fps": 10, "frames": 5, "quaternion_order": "xyzw"})
    )
    motion = MotionLib([str(path)], G1Kinematics(str(SCENE), "cpu"), "cpu")
    assert motion.clip_len.tolist() == [21]


def test_metrics_report_physical_units_for_constant_joint_velocity():
    from unisupermimic.tasks.motion_tracking.g1.supermimic.preprocess import motion_metrics

    rows = standing(50)
    rows[:, 7] = np.arange(50) * 0.02
    metrics = motion_metrics(rows, 50)
    assert metrics["joint_velocity_peak"] == pytest.approx(1)
    assert metrics["joint_velocity_rms"] == pytest.approx(1 / np.sqrt(29))
    assert metrics["joint_acceleration_rms"] < 1e-9
    assert metrics["joint_jerk_rms"] < 1e-7


def test_motion_loader_rejects_changed_csv_fingerprint(tmp_path):
    import json

    from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
    from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib

    path = tmp_path / "changed.csv"
    np.savetxt(path, standing(5)[:, [0, 1, 2, 4, 5, 6, 3, *range(7, 36)]], delimiter=",")
    path.with_suffix(".json").write_text(json.dumps({"fps": 30, "csv_sha256": "stale"}))
    with pytest.raises(ValueError, match="content changed"):
        MotionLib([str(path)], G1Kinematics(str(SCENE), "cpu"), "cpu")
