import pytest
import torch

from unisupermimic.finetune.collector import DeploymentCollector


def make(bundle, scene, **kw):
    return DeploymentCollector(bundle, scene, num_envs=2, randomize=False, **kw)


def test_reference_ankle_velocity_has_zero_extra_cost(adaptation_bundle, experiment_inputs):
    from unisupermimic.finetune.continuity import ankle_rate_cost

    c = make(adaptation_bundle, experiment_inputs["scene"], ankle_rate_weight=0.2)
    mdp = c.supermimic_mdp
    f = torch.zeros(2, dtype=torch.long)
    idx = [4, 5, 10, 11]
    mdp.motion.qd[:, idx] = 1
    state = mdp.motion.ref_state(f)
    previous = torch.zeros(2, 29)
    a = previous.clone()
    a[:, idx] = 0.02 / c.scales[idx]
    cost, _ = ankle_rate_cost(mdp, state, f, a, previous)
    torch.testing.assert_close(cost, torch.zeros(2), atol=1e-10, rtol=0)
    assert isinstance(mdp.action_scales_values, tuple)


def test_command_and_measured_ankle_chatter_have_finite_gradients(
    adaptation_bundle, experiment_inputs
):
    from unisupermimic.finetune.continuity import ankle_rate_cost
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP

    c = make(adaptation_bundle, experiment_inputs["scene"], ankle_rate_weight=0.2)
    mdp = c.supermimic_mdp
    f = torch.zeros(2, dtype=torch.long)
    idx = [4, 5, 10, 11]
    state = list(mdp.motion.ref_state(f))
    state[5] = state[5].clone()
    state[5][:, idx] = 2
    state[5].requires_grad_()
    a = torch.zeros(2, 29)
    a[:, idx] = 0.1 / c.scales[idx]
    a.requires_grad_()
    previous = torch.zeros_like(a)
    cost, terms = ankle_rate_cost(mdp, tuple(state), f, a, previous)
    torch.testing.assert_close(cost, torch.full((2,), 6.75))
    cost.sum().backward()
    assert torch.isfinite(a.grad).all() and (a.grad[:, idx] > 0).all()
    assert torch.isfinite(state[5].grad).all() and (state[5].grad[:, idx] > 0).all()
    base = MDP.reward(mdp, tuple(state), f, a, previous)
    torch.testing.assert_close(mdp.reward(tuple(state), f, a, previous), base - 0.2 * 0.02 * cost)
    _, details = mdp.reward(tuple(state), f, a, previous, terms=True)
    assert "ankle_rate_penalty" in details
    assert (terms["ankle_command_velocity_error"] > 0).all()


def test_non_ankle_motion_is_not_charged(adaptation_bundle, experiment_inputs):
    from unisupermimic.finetune.continuity import ankle_rate_cost

    c = make(adaptation_bundle, experiment_inputs["scene"], ankle_rate_weight=0.2)
    m = c.supermimic_mdp
    f = torch.zeros(2, dtype=torch.long)
    s = list(m.motion.ref_state(f))
    s[5] = s[5].clone()
    s[5][:, 20] = 100
    a = torch.zeros(2, 29)
    a[:, 20] = 100
    cost, _ = ankle_rate_cost(m, tuple(s), f, a, torch.zeros_like(a))
    torch.testing.assert_close(cost, torch.zeros(2))


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_ankle_weight_is_rejected(adaptation_bundle, experiment_inputs, weight):
    with pytest.raises(ValueError, match="ankle"):
        make(adaptation_bundle, experiment_inputs["scene"], ankle_rate_weight=weight)


def test_default_reward_is_exactly_preserved(adaptation_bundle, experiment_inputs):
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP

    c = make(adaptation_bundle, experiment_inputs["scene"])
    m = c.supermimic_mdp
    f = torch.zeros(2, dtype=torch.long)
    a = torch.zeros(2, 29)
    s = m.motion.ref_state(f)
    torch.testing.assert_close(m.reward(s, f, a, a), MDP.reward(m, s, f, a, a), atol=0, rtol=0)
    assert m.ankle_rate_weight == 0
