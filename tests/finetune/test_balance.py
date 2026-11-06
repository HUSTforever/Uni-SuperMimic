import numpy as np
import pytest
import torch

from unisupermimic.finetune.collector import DeploymentCollector


def collector(bundle, scene, **kwargs):
    return DeploymentCollector(bundle, scene, num_envs=2, randomize=False, **kwargs)


def test_default_reward_preserves_existing_objective(adaptation_bundle, experiment_inputs):
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP

    c = collector(adaptation_bundle, experiment_inputs["scene"])
    f = torch.tensor([0, 1])
    s = c.motion.ref_state(f)
    a = torch.zeros(2, 29)
    torch.testing.assert_close(
        c.supermimic_mdp.reward(s, f, a, a), MDP.reward(c.supermimic_mdp, s, f, a, a)
    )
    assert c.supermimic_mdp.balance_weight == 0


def test_stability_cost_detects_sliding_rotation_and_has_finite_gradient(
    adaptation_bundle, experiment_inputs
):
    from unisupermimic.finetune.balance import stability_cost
    from unisupermimic.rl.algos.supermimic.geometry import axis_angle_to_mat

    c = collector(adaptation_bundle, experiment_inputs["scene"], balance_weight=1)
    mdp = c.supermimic_mdp
    f = torch.zeros(2, dtype=torch.long)
    s = list(c.motion.ref_state(f))
    # Fixture translates the root at a constant .05m/s: set actual and reference to rest.
    for k in ("root_v", "root_w", "qd", "body_v", "body_w"):
        getattr(mdp.motion, k).zero_()
    s = list(mdp.motion.ref_state(f))
    zero, terms = stability_cost(mdp, tuple(s), f)
    torch.testing.assert_close(zero, torch.zeros(2), atol=1e-6, rtol=0)
    s[2] = torch.tensor([[0.6, 0, 0], [0.6, 0, 0]], requires_grad=True)
    cost, terms = stability_cost(mdp, tuple(s), f)
    assert (terms["stance_slip"] > 0.3).all()
    cost.sum().backward()
    assert torch.isfinite(s[2].grad).all() and (s[2].grad[:, 0] > 0).all()
    s[1] = axis_angle_to_mat(torch.tensor([1.0, 0, 0]), torch.tensor([0.2, 0.2]))
    s[3] = torch.ones(2, 3, requires_grad=True)
    cost, terms = stability_cost(mdp, tuple(s), f)
    assert (terms["gravity_error"] > 0.03).all()
    assert (terms["angular_velocity_error"] > 2).all()
    cost.sum().backward()
    assert torch.isfinite(s[3].grad).all()
    a = torch.zeros(2, 29)
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP

    base = MDP.reward(mdp, tuple(s), f, a, a)
    torch.testing.assert_close(mdp.reward(tuple(s), f, a, a), base - 0.02 * cost)
    assert mdp.to("cpu").balance_weight == 1


@pytest.mark.parametrize("axis", [0, 2])
def test_swinging_reference_does_not_pin_feet(adaptation_bundle, experiment_inputs, axis):
    from unisupermimic.finetune.balance import stability_cost

    c = collector(adaptation_bundle, experiment_inputs["scene"], balance_weight=1)
    mdp = c.supermimic_mdp
    f = torch.zeros(2, dtype=torch.long)
    mdp.motion.body_v[..., [3, 6], axis] = 1.0
    s = list(mdp.motion.ref_state(f))
    s[2] = torch.ones(2, 3)
    _, terms = stability_cost(mdp, tuple(s), f)
    torch.testing.assert_close(terms["stance_slip"], torch.zeros(2))


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_balance_weight_is_rejected(adaptation_bundle, experiment_inputs, value):
    with pytest.raises(ValueError, match="balance"):
        collector(adaptation_bundle, experiment_inputs["scene"], balance_weight=value)


def test_focused_starts_retain_entry_and_uniform_sampling(adaptation_bundle, experiment_inputs):
    c = collector(adaptation_bundle, experiment_inputs["scene"], focus_intervals=[(20, 30)])
    frames = c._sample_frames(10000)
    # Existing non-sequence hard interval in this short fixture has no overlap with focus.
    assert 0.15 < np.mean((frames >= 20) & (frames < 30)) < 0.20
    assert 0.23 < np.mean(frames == 0) < 0.28
    with pytest.raises(ValueError, match="focus"):
        collector(adaptation_bundle, experiment_inputs["scene"], focus_intervals=[(20, 1000)])
