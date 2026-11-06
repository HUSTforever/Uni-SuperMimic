import pytest
import torch

from unisupermimic.rl.algos.supermimic.geometry import flatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner


def learner_for(bundle, scene, output, **weights):
    from unisupermimic.finetune.collector import DeploymentCollector
    from unisupermimic.finetune.learner import DeploymentLearner

    collector = DeploymentCollector(bundle, scene, num_envs=2, randomize=False)
    cfg = SuperMimicConfig(
        iterations=2,
        buffer_steps=16,
        horizon=2,
        hist=10,
        goal_frames=10,
        z_dim=4,
        wm_hidden=(8,),
        policy_hidden=(8,),
        critic_hidden=(8,),
        wm_window=2,
        wm_batches=1,
        wm_batch=4,
        pi_batch=4,
        warmup_steps=10,
        compile=False,
        amp="none",
        log_backend="no_print",
    )
    return DeploymentLearner(collector, cfg, output, device="cpu", **weights)


def physical_state():
    return (
        torch.zeros(1, 2, 3),
        torch.eye(3).expand(1, 2, 3, 3).clone(),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 29),
        torch.zeros(1, 2, 29),
    )


@pytest.mark.parametrize("joint,ratio", [(4, 4), (5, 4), (10, 4), (11, 4), (0, 1), (28, 1)])
def test_only_four_ankle_velocities_get_the_extra_weight(
    adaptation_bundle, experiment_inputs, tmp_path, joint, ratio
):
    learner = learner_for(
        adaptation_bundle,
        experiment_inputs["scene"],
        tmp_path / "fit",
        wm_ankle_velocity_weight=4,
        wm_root_velocity_weight=2,
    )
    actual = physical_state()
    predicted = list(physical_state())
    predicted[5][..., joint] = 1
    target = flatten_state(actual)
    mask = torch.ones(1, 2)
    baseline, last = SuperMimicLearner._wm_err(learner, tuple(predicted), target, mask)
    weighted, last_weighted = learner._wm_err(tuple(predicted), target, mask)
    torch.testing.assert_close(weighted, baseline * ratio)
    torch.testing.assert_close(last_weighted, last, atol=0, rtol=0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_root_linear_velocity_weight_covers_all_local_axes(
    adaptation_bundle, experiment_inputs, tmp_path, axis
):
    learner = learner_for(
        adaptation_bundle,
        experiment_inputs["scene"],
        tmp_path / "fit",
        wm_ankle_velocity_weight=4,
        wm_root_velocity_weight=2,
    )
    actual, predicted = physical_state(), list(physical_state())
    predicted[2][..., axis] = 1
    target, mask = flatten_state(actual), torch.ones(1, 2)
    base, last = SuperMimicLearner._wm_err(learner, tuple(predicted), target, mask)
    loss, weighted_last = learner._wm_err(tuple(predicted), target, mask)
    torch.testing.assert_close(loss, 2 * base)
    torch.testing.assert_close(weighted_last, last, atol=0, rtol=0)


def test_joint_position_is_not_mistaken_for_joint_velocity(
    adaptation_bundle, experiment_inputs, tmp_path
):
    learner = learner_for(
        adaptation_bundle,
        experiment_inputs["scene"],
        tmp_path / "fit",
        wm_ankle_velocity_weight=4,
        wm_root_velocity_weight=2,
    )
    actual, predicted = physical_state(), list(physical_state())
    predicted[4][..., 4] = 1
    target, mask = flatten_state(actual), torch.ones(1, 2)
    base = SuperMimicLearner._wm_err(learner, tuple(predicted), target, mask)
    loss = learner._wm_err(tuple(predicted), target, mask)
    for a, b in zip(base, loss):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_masked_ankle_errors_have_zero_loss_gradient(
    adaptation_bundle, experiment_inputs, tmp_path
):
    learner = learner_for(
        adaptation_bundle, experiment_inputs["scene"], tmp_path / "fit", wm_ankle_velocity_weight=4
    )
    actual, predicted = physical_state(), list(physical_state())
    predicted[5][..., 4] = 1
    predicted[5][0, 1, 4] = 1000
    predicted[5].requires_grad_()
    loss, _ = learner._wm_err(tuple(predicted), flatten_state(actual), torch.tensor([[1.0, 0.0]]))
    loss.backward()
    assert predicted[5].grad[0, 0, 4] > 0
    assert torch.isfinite(predicted[5].grad).all()
    assert predicted[5].grad[0, 1].abs().sum() == 0


def test_default_loss_is_bitwise_identical_to_base(adaptation_bundle, experiment_inputs, tmp_path):
    learner = learner_for(adaptation_bundle, experiment_inputs["scene"], tmp_path / "fit")
    actual, predicted = physical_state(), list(physical_state())
    predicted[2][..., 2] = 0.5
    predicted[5][..., 4] = 0.7
    target, mask = flatten_state(actual), torch.tensor([[1.0, 0.0]])
    for a, b in zip(
        learner._wm_err(tuple(predicted), target, mask),
        SuperMimicLearner._wm_err(learner, tuple(predicted), target, mask),
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("name", ["wm_ankle_velocity_weight", "wm_root_velocity_weight"])
@pytest.mark.parametrize("value", [0, 0.5, -1, float("nan"), float("inf")])
def test_invalid_weights_are_rejected_before_output_is_created(
    adaptation_bundle, experiment_inputs, tmp_path, name, value
):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="weight"):
        learner_for(adaptation_bundle, experiment_inputs["scene"], output, **{name: value})
    assert not output.exists()
