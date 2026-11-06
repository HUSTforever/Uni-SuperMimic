import numpy as np
import pytest
import torch


def sensor(*, time=0.0, yaw=0.0):
    from unisupermimic.deploy.controller import SensorState

    return SensorState(q=np.arange(29, dtype=np.float32) * 0.01,
                       dq=np.arange(29, dtype=np.float32) * -0.1,
                       quaternion=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
                       gyro=np.array([1.0, 2.0, 3.0]), sample_time=time)


def test_observation_uses_body_gyro_and_row_major_first_two_rotation_columns():
    from unisupermimic.deploy.controller import build_observation

    state = sensor(yaw=np.pi / 2)
    obs = build_observation(state, np.zeros(29), np.ones(29) * 0.2,
                            np.ones(29) * 0.3, np.eye(3))
    np.testing.assert_allclose(obs[:29], state.q - 0.2, atol=1e-7)
    np.testing.assert_allclose(obs[29:58], state.dq)
    np.testing.assert_allclose(obs[58:64], [1, 2, 3, 0, 0, -1], atol=1e-7)
    np.testing.assert_allclose(obs[64:93], 0.3)
    np.testing.assert_allclose(obs[93:], [0, 1, -1, 0, 0, 0], atol=1e-6)


def test_observation_matches_training_oracle_for_tilted_pelvis_and_waist():
    from types import SimpleNamespace

    from unisupermimic.deploy.controller import build_observation, quaternion_matrix
    from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP

    state = sensor()
    state.quaternion = np.array([0.9, 0.1, -0.2, 0.3]) / np.sqrt(0.95)
    state.q[12:15] = [0.8, -0.4, 0.6]  # Waist changes must not transform an already-pelvis IMU.
    rotation = quaternion_matrix(state.quaternion)
    reference = quat_to_mat(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    previous = torch.linspace(-0.4, 0.4, 29)
    default = torch.linspace(-0.2, 0.2, 29)
    context = torch.zeros(1, 33)
    context[0, 1:30] = 0.07
    oracle = SimpleNamespace(default_q=default, motion=SimpleNamespace(root_R=reference))
    R = torch.tensor(rotation, dtype=torch.float32)[None]
    gyro = torch.tensor(state.gyro, dtype=torch.float32)[None]
    fake_state = (torch.zeros(1, 3), R, torch.zeros(1, 3),
                  (R @ gyro[..., None]).squeeze(-1),
                  torch.tensor(state.q)[None], torch.tensor(state.dq)[None])
    expected = MDP.obs(oracle, fake_state, previous[None], context, torch.tensor([0]))[0]
    actual = build_observation(state, default.numpy(), np.full(29, 0.07),
                               previous.numpy(), reference[0].numpy())
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6)


def test_controller_history_initialization_and_progression(bundle):
    from unisupermimic.deploy.controller import PolicyController

    controller = PolicyController(bundle, align_heading=False)
    state = sensor()
    controller.step(state, 0)
    np.testing.assert_allclose(controller.history[0], controller.history[-1])
    first = controller.history[-1].copy()
    state.sample_time = 0.02
    controller.step(state, 1)
    np.testing.assert_allclose(controller.history[-2], first)
    assert np.any(controller.history[-1, 64:93] != 0)
    with pytest.raises(ValueError, match="frame|sequence"):
        controller.step(sensor(time=0.04), 1)


def test_heading_alignment_is_one_time_and_preserves_body_gyro(bundle):
    from unisupermimic.deploy.controller import PolicyController

    a, b = PolicyController(bundle), PolicyController(bundle)
    a.step(sensor(yaw=0), 0)
    b.step(sensor(yaw=1.2), 0)
    np.testing.assert_allclose(a.history, b.history, atol=1e-6)
    b.step(sensor(time=0.02, yaw=1.5), 1)
    assert np.max(np.abs(b.history[-1, 93:] - a.history[-1, 93:])) > 0.1


def test_target_clipping_feedback_is_applied_raw_action(bundle):
    from unisupermimic.deploy.controller import PolicyController

    controller = PolicyController(bundle)
    controller.policy = lambda h, g: torch.full((1, 29), 1000.0)
    target = controller.step(sensor(), 0)
    upper = np.array(bundle.metadata["joint_upper"])
    np.testing.assert_allclose(target, upper)
    expected_raw = ((upper - np.array(bundle.metadata["default_joint_positions"])) /
                    np.array(bundle.metadata["action_scales"]))
    np.testing.assert_allclose(controller.previous_action, expected_raw, rtol=1e-6)
    assert controller.clipped_targets > 0


@pytest.mark.parametrize("mutation", ["nan", "quat", "time", "output"])
def test_invalid_sample_or_policy_output_latches_fault(bundle, mutation):
    from unisupermimic.deploy.controller import PolicyController

    controller = PolicyController(bundle)
    controller.step(sensor(), 0)
    state = sensor(time=0.02)
    if mutation == "nan":
        state.q[0] = np.nan
    elif mutation == "quat":
        state.quaternion[:] = 0
    elif mutation == "time":
        state.sample_time = -0.01
    else:
        controller.policy = lambda h, g: torch.full((1, 29), float("nan"))
    with pytest.raises(ValueError):
        controller.step(state, 1)
    assert controller.fault is not None
    with pytest.raises(ValueError, match="fault"):
        controller.step(sensor(time=0.04), 2)


def test_end_of_reference_is_not_a_silent_hold(bundle):
    from unisupermimic.deploy.controller import PolicyController

    controller = PolicyController(bundle)
    with pytest.raises(ValueError, match="frame|reference"):
        controller.step(sensor(), len(bundle.reference["q"]) - 1)
