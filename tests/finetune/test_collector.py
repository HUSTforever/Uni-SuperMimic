import numpy as np
import pytest
import torch


def create(bundle, scene, **kwargs):
    from unisupermimic.finetune.collector import DeploymentCollector

    return DeploymentCollector(bundle, scene, num_envs=2, seed=7, randomize=False, **kwargs)


def test_nominal_collector_matches_deployment_physics(adaptation_bundle, experiment_inputs):
    from unisupermimic.deploy.mujoco_sim import NativeSimulator

    c = create(adaptation_bundle, experiment_inputs["scene"])
    c.reset_to_frames(torch.tensor([0, 0]), zero_velocity=False, delay_substeps=0)
    direct = NativeSimulator(adaptation_bundle, experiment_inputs["scene"])
    direct.reset("reference")
    action = torch.zeros(2, 29)
    c.step(action)
    direct.advance(np.array(adaptation_bundle.metadata["default_joint_positions"]))
    np.testing.assert_allclose(c.simulators[0].qpos(), direct.qpos(), atol=1e-8)
    assert c.state.info["supermimic_state"]["f"].tolist() == [1, 1]


def test_delay_holds_previous_target_for_exactly_one_substep(adaptation_bundle, experiment_inputs):
    from unisupermimic.deploy.mujoco_sim import NativeSimulator

    c = create(adaptation_bundle, experiment_inputs["scene"])
    c.reset_to_frames(torch.tensor([0, 0]), zero_velocity=False, delay_substeps=1)
    first, second = torch.zeros(2, 29), torch.ones(2, 29) * 0.2
    c.step(first)
    c.step(second)
    direct = NativeSimulator(adaptation_bundle, experiment_inputs["scene"])
    direct.reset("reference")
    old = np.array(adaptation_bundle.metadata["default_joint_positions"])
    new = old + np.array(adaptation_bundle.metadata["action_scales"]) * 0.2
    direct.advance(old)
    direct.advance(old, duration=0.005)
    direct.advance(new, duration=0.015)
    np.testing.assert_allclose(c.simulators[0].qpos(), direct.qpos(), atol=1e-8)


def test_terminal_transition_is_saved_before_row_specific_autoreset(
    adaptation_bundle, experiment_inputs
):
    c = create(adaptation_bundle, experiment_inputs["scene"])
    last = len(adaptation_bundle.reference["q"]) - 1
    c.reset_to_frames(torch.tensor([last - 1, 10]), zero_velocity=False, delay_substeps=0)
    previous = c.state.info["supermimic_state"]["s"].clone()
    out = c.step(torch.zeros(2, 29))
    transition, current = out.info["supermimic"], out.info["supermimic_state"]
    assert transition["f"].tolist() == [last - 1, 10]
    torch.testing.assert_close(transition["s"], previous)
    assert transition["done"].tolist() == [True, False]
    assert current["ep_step"].tolist() == [0, 1]
    assert current["f"][0] < last
    assert current["f"][1] == 11
    assert not torch.equal(transition["s1"][0], current["s"][0])


def test_clipped_target_feedback_and_zero_velocity_reset(adaptation_bundle, experiment_inputs):
    c = create(adaptation_bundle, experiment_inputs["scene"])
    c.reset_to_frames(torch.tensor([0, 0]), zero_velocity=True, delay_substeps=0)
    assert np.all(c.simulators[0].data.qvel == 0)
    out = c.step(torch.full((2, 29), 1000.0))
    applied = out.info["supermimic"]["a"]
    default = torch.tensor(adaptation_bundle.metadata["default_joint_positions"])
    scales = torch.tensor(adaptation_bundle.metadata["action_scales"])
    upper = torch.tensor(adaptation_bundle.metadata["joint_upper"])
    torch.testing.assert_close(default + scales * applied[0], upper, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.info["supermimic_state"]["a_prev"], applied)


def test_invalid_action_fails_before_physics(adaptation_bundle, experiment_inputs):
    c = create(adaptation_bundle, experiment_inputs["scene"])
    before = c.simulators[0].data.time
    with pytest.raises(ValueError, match="finite|action"):
        c.step(torch.full((2, 29), float("nan")))
    assert c.simulators[0].data.time == before


def test_reconstructed_reference_goal_matches_the_trusted_bundle(
    adaptation_bundle, experiment_inputs
):
    c = create(adaptation_bundle, experiment_inputs["scene"])
    frames = torch.tensor([0, 25, 98, 99])
    np.testing.assert_allclose(
        c.motion.goal(frames, 10), adaptation_bundle.reference["goals"][frames], atol=1e-6
    )
