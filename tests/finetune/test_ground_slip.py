import pytest
import torch

from unisupermimic.finetune.collector import DeploymentCollector


def make(bundle, scene, **kw):
    return DeploymentCollector(bundle, scene, num_envs=2, randomize=False, balance_weight=1, **kw)


def test_fast_horizontal_reference_still_penalizes_ground_sliding(
    adaptation_bundle, experiment_inputs
):
    from unisupermimic.finetune.balance import stability_cost

    c = make(adaptation_bundle, experiment_inputs["scene"], ground_slip_weight=1)
    m = c.supermimic_mdp
    f = torch.zeros(2, dtype=torch.long)
    m.motion.body_v[..., [3, 6], 0] = 1
    state = list(m.motion.ref_state(f))
    state[2] = torch.tensor([[0.6, 0, 0], [0.6, 0, 0]], requires_grad=True)
    cost, terms = stability_cost(m, tuple(state), f)
    torch.testing.assert_close(terms["stance_slip"], torch.zeros(2))
    assert (terms["ground_slip"] > 0.3).all()
    cost.sum().backward()
    assert torch.isfinite(state[2].grad).all() and (state[2].grad[:, 0] > 0).all()
    m.ground_slip_weight = 0
    old_cost, _ = stability_cost(m, tuple(state), f)
    torch.testing.assert_close(cost - old_cost, terms["ground_slip"])
    assert m.to("cpu").ground_slip_weight == 0


@pytest.mark.parametrize("mode", ["vertical", "high"])
def test_swinging_foot_does_not_get_ground_penalty(adaptation_bundle, experiment_inputs, mode):
    from unisupermimic.finetune.balance import stability_cost

    c = make(adaptation_bundle, experiment_inputs["scene"], ground_slip_weight=1)
    m = c.supermimic_mdp
    f = torch.zeros(2, dtype=torch.long)
    if mode == "vertical":
        m.motion.body_v[..., [3, 6], 2] = 1
    else:
        m.motion.body_p[..., [3, 6], 2] = 0.5
    state = list(m.motion.ref_state(f))
    state[2] = torch.ones(2, 3)
    _, terms = stability_cost(m, tuple(state), f)
    torch.testing.assert_close(terms["ground_slip"], torch.zeros(2))


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_ground_weight_fails(adaptation_bundle, experiment_inputs, weight):
    with pytest.raises(ValueError, match="ground"):
        make(adaptation_bundle, experiment_inputs["scene"], ground_slip_weight=weight)


def test_ground_weight_requires_enabled_balance(adaptation_bundle, experiment_inputs):
    with pytest.raises(ValueError, match="balance"):
        DeploymentCollector(
            adaptation_bundle, experiment_inputs["scene"], num_envs=1, ground_slip_weight=1
        )
