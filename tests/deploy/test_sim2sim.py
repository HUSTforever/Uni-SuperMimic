import json
from pathlib import Path

import numpy as np
import pytest

SCENE = Path(__file__).resolve().parents[2] / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"


def test_pd_uses_measured_velocity_and_saturates_each_motor():
    from unisupermimic.deploy.mujoco_sim import pd_torque

    actual, clipped = pd_torque(np.array([0., 1.]), np.array([2., -1.]),
                                np.array([1., 0.]), np.array([10., 20.]),
                                np.array([1., 2.]), np.array([5., 30.]))
    np.testing.assert_allclose(actual, [5., -18.])
    assert clipped == 1


def test_native_sensor_exposes_only_motor_and_pelvis_imu(bundle):
    from unisupermimic.deploy.mujoco_sim import NativeSimulator

    sim = NativeSimulator(bundle, SCENE)
    sim.reset("reference")
    sensor = sim.sensor()
    assert not hasattr(sensor, "root_position")
    np.testing.assert_allclose(sensor.q, bundle.reference["q"][0], atol=1e-6)
    np.testing.assert_allclose(sensor.quaternion, bundle.reference["root_quat"][0], atol=1e-6)
    sim.advance(sensor.q)
    assert sim.sensor().sample_time == pytest.approx(0.02)
    assert sim.qpos().shape == (36,)


def test_physics_pose_and_imu_are_from_the_same_finished_timestep(bundle):
    from unisupermimic.deploy.mujoco_sim import NativeSimulator

    sim = NativeSimulator(bundle, SCENE)
    sim.reset("reference")
    sim.data.qvel[3:6] = [1, 2, 3]
    sim.advance(sim.sensor().q)
    np.testing.assert_allclose(sim.data.xquat[sim.pelvis], sim.sensor().quaternion, atol=1e-10)


def test_dds_tracking_applies_the_same_reference_height_gate_as_native(bundle):
    from unisupermimic.deploy.dds_sim import tracking_failure
    from unisupermimic.deploy.mujoco_sim import NativeSimulator

    sim = NativeSimulator(bundle, SCENE)
    sim.reset("reference")
    sim.data.qpos[2] = bundle.reference["root_p"][1, 2]-0.26
    assert sim.failure() is None
    assert "reference height" in tracking_failure(sim, 20)
    # Exiting the clip has its own absolute-height/tilt gate, no frozen reference.
    assert tracking_failure(sim, 10000) is None


@pytest.mark.slow
def test_actual_two_process_sdk_loopback_completes_short_clip(bundle, tmp_path):
    pytest.importorskip("unitree_sdk2py")
    from unisupermimic.deploy.dds_sim import run_dds

    report = run_dds(bundle, SCENE, tmp_path / "dds", video=False, domain=43)
    assert report["tracking"]["completed"] is True
    assert report["dds"]["received_states"] > 0
    assert report["dds"]["sent_commands"] >= 3
    assert report["simulator"]["watchdog_triggered"] is False


def test_simulator_rejects_missing_or_duplicate_motor_mapping(bundle, tmp_path):
    from unisupermimic.deploy.mujoco_sim import NativeSimulator

    path = tmp_path / "bad.xml"
    path.write_text('<mujoco><worldbody><body><freejoint/>'
                    '<inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>'
                    '</body></worldbody></mujoco>')
    with pytest.raises(ValueError, match="29|joint|motor|mapping"):
        NativeSimulator(bundle, path)


def test_standalone_runner_records_true_completion_and_no_overwrite(bundle, tmp_path):
    from unisupermimic.deploy.run_sim2sim import run_native

    output = tmp_path / "result"
    report = run_native(bundle, SCENE, output, init_mode="reference", video=False)
    assert report["tracking"]["expected_steps"] == 3
    assert report["tracking"]["steps"] <= 3
    assert report["initialization"] == "reference"
    assert report["standing_entry_verified"] is False
    assert report["hardware_ready"] is False
    assert json.loads((output / "report.json").read_text())["transport"] == "native_mujoco"
    with np.load(output / "trajectory.npz", allow_pickle=False) as data:
        assert data["qpos"].shape[1] == 36
    with pytest.raises(FileExistsError):
        run_native(bundle, SCENE, output, init_mode="reference", video=False)


def test_failed_entry_keeps_tracking_empty_instead_of_inventing_a_frame(bundle, tmp_path, monkeypatch):
    from unisupermimic.deploy.mujoco_sim import NativeSimulator
    from unisupermimic.deploy.run_sim2sim import run_native

    monkeypatch.setattr(NativeSimulator, "failure", lambda self, frame=None: "forced entry fall")
    output = tmp_path / "failed_entry"
    report = run_native(bundle, SCENE, output, init_mode="standing", video=False)
    assert report["tracking"]["steps"] == 0
    assert "joint_tracking_rmse_rad" not in report
    with np.load(output / "trajectory.npz", allow_pickle=False) as data:
        assert data["qpos"].shape == (0, 36)
