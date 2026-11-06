import numpy as np
import pytest


def test_observer_copies_each_physical_substep_without_changing_simulation(
    adaptation_bundle, experiment_inputs
):
    from unisupermimic.deploy.mujoco_sim import NativeSimulator

    samples = []

    def record(time, qpos, joint_velocity, root_velocity):
        samples.append((time, qpos.copy(), joint_velocity.copy(), root_velocity.copy()))
        qpos[:] = 999
        joint_velocity[:] = 999
        root_velocity[:] = 999

    baseline = NativeSimulator(adaptation_bundle, experiment_inputs["scene"])
    traced = NativeSimulator(adaptation_bundle, experiment_inputs["scene"], step_observer=record)
    for sim in (baseline, traced):
        sim.reset("reference_zero_velocity")
    np.testing.assert_array_equal(traced.qpos(), baseline.qpos())
    target = np.array(adaptation_bundle.metadata["default_joint_positions"])
    for value in (target, target + 0.01):
        baseline.advance(value)
        traced.advance(value)
        np.testing.assert_array_equal(traced.qpos(), baseline.qpos())
        np.testing.assert_array_equal(traced.data.qvel, baseline.data.qvel)
    np.testing.assert_allclose([s[0] for s in samples], np.arange(9) * 0.005, atol=1e-12)
    np.testing.assert_array_equal(samples[-1][1], traced.qpos())
    np.testing.assert_array_equal(samples[-1][2], traced.data.qvel[traced.vadr])
    np.testing.assert_array_equal(samples[-1][3], traced.data.qvel[:3])
    assert samples[0][0] == 0 and np.all(samples[0][2] == 0)


def test_invalid_physics_observer_is_rejected(adaptation_bundle, experiment_inputs):
    from unisupermimic.deploy.mujoco_sim import NativeSimulator

    with pytest.raises(ValueError, match="observer"):
        NativeSimulator(adaptation_bundle, experiment_inputs["scene"], step_observer=3)
