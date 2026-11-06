import copy
import math

import pytest
import torch


def _base():
    from unisupermimic.rl.algos.supermimic.networks import Policy

    base = Policy(15, 4, hist=2, z_dim=3, num_joints=3, hidden=(8,), action_smoothing=0.35)
    with torch.no_grad():
        base.dec[-1].weight.zero_()
        base.dec[-1].bias.copy_(torch.tensor([-20.0, 0.0, 20.0]))
    return base.eval()


def _limits():
    default = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    scale = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
    lower = torch.tensor([-1.0, -0.8, -0.7], dtype=torch.float64)
    upper = torch.tensor([1.0, 1.1, 0.9], dtype=torch.float64)
    return default, scale, lower, upper


def test_zero_residual_keeps_actual_targets_with_saturation_and_previous_actions():
    from unisupermimic.rl.algos.supermimic.projected_policy import ProjectedResidualPolicy

    base = _base()
    default, scale, lower, upper = _limits()
    wrapped = ProjectedResidualPolicy(base, (lower - default) / scale, (upper - default) / scale)
    history = torch.zeros(2, 2, 15)
    history[:, -1, 12:] = torch.tensor([-0.3, 0.2, 0.4])
    goal = torch.zeros(2, 4)
    with torch.no_grad():
        original = base(history, goal)
        actual = wrapped(history, goal)
    assert not torch.equal(actual, original)  # The raw center changes; mechanical targets do not.
    old_targets = torch.clamp(default + scale * original.double(), min=lower, max=upper)
    new_targets = torch.clamp(default + scale * actual.double(), min=lower, max=upper)
    assert torch.equal(old_targets, new_targets)
    assert old_targets[0, 0] == -1.0 and old_targets[0, 2] == 0.9


def test_residual_can_change_saturated_target_without_backbone_gradients_or_norm_updates():
    from unisupermimic.rl.algos.supermimic.projected_policy import ProjectedResidualPolicy

    base = _base()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    default, scale, lower, upper = _limits()
    wrapped = ProjectedResidualPolicy(base, (lower - default) / scale, (upper - default) / scale)
    wrapped.train()
    history, goal = torch.zeros(2, 2, 15), torch.zeros(2, 4)
    mean = wrapped(history, goal)
    target_before = torch.clamp(default + scale * mean.detach().double(), min=lower, max=upper)
    mean.square().mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in wrapped.base.parameters())
    assert wrapped.residual[-1].bias.grad[2] > 0
    optimizer = torch.optim.SGD([wrapped.residual[-1].bias], lr=0.1)
    optimizer.step()
    target_after = torch.clamp(
        default + scale * wrapped(history, goal).double(), min=lower, max=upper
    )
    assert target_after[0, 2] < target_before[0, 2]
    assert target_after[0, 0] > target_before[0, 0]
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize(
    "lower,upper",
    [([-1, -1], [1, 1]), ([1, 1, 1], [1, 1, 1]), ([math.nan, -1, -1], [1, 1, 1])],
)
def test_projected_policy_rejects_invalid_raw_bounds(lower, upper):
    from unisupermimic.rl.algos.supermimic.projected_policy import ProjectedResidualPolicy

    with pytest.raises(ValueError):
        ProjectedResidualPolicy(_base(), lower, upper)


def test_trace_roundtrip_preserves_nonzero_residual_and_different_batch(tmp_path):
    from unisupermimic.rl.algos.supermimic.projected_policy import ProjectedResidualPolicy

    wrapped = ProjectedResidualPolicy(_base(), [-1, -1, -1], [1, 1, 1]).eval()
    with torch.no_grad():
        wrapped.residual[-1].bias.fill_(0.2)
    traced = torch.jit.trace(wrapped, (torch.zeros(2, 2, 15), torch.zeros(2, 4)))
    path = tmp_path / "residual.ts"
    traced.save(str(path))
    loaded = torch.jit.load(str(path))
    history, goal = torch.randn(5, 2, 15), torch.randn(5, 4)
    with torch.no_grad():
        assert torch.equal(loaded(history, goal), wrapped(history, goal))
        baseline = torch.clamp(wrapped.base(history, goal), min=-1.0, max=1.0)
        assert (loaded(history, goal) - baseline).abs().max() > 0.05


@pytest.mark.parametrize("flag", [1, "yes", None])
def test_projected_mode_flag_must_be_boolean(flag):
    from unisupermimic.finetune.physical_ppo import PhysicalPPOConfig

    with pytest.raises(ValueError):
        PhysicalPPOConfig(projected_residual=flag)


def test_policy_factory_rejects_unknown_architecture(experiment_inputs):
    from demo.export_policy import build_policy

    checkpoint = torch.load(experiment_inputs["checkpoint_path"], weights_only=True)
    checkpoint["args"]["policy_type"] = "unknown_model"
    with pytest.raises(ValueError, match="policy"):
        build_policy(checkpoint)


@pytest.mark.parametrize("corruption", ["value", "dtype", "shape"])
def test_factory_rejects_projection_buffers_inconsistent_with_declared_bounds(
    experiment_inputs, corruption
):
    from demo.export_policy import build_policy

    from unisupermimic.rl.algos.supermimic.projected_policy import ProjectedResidualPolicy

    source = torch.load(experiment_inputs["checkpoint_path"], weights_only=True)
    actor = ProjectedResidualPolicy(build_policy(source), [-1.0] * 29, [1.0] * 29)
    source["args"].update(
        policy_type="projected_residual_v1",
        residual_hidden=128,
        residual_raw_lower=[-1.0] * 29,
        residual_raw_upper=[1.0] * 29,
    )
    source["pi"] = {k: v.clone() for k, v in actor.state_dict().items()}
    if corruption == "value":
        source["pi"]["raw_upper_out"][0] = 0.25
    elif corruption == "dtype":
        source["pi"]["raw_upper_out"] = source["pi"]["raw_upper_out"].double()
    else:
        source["pi"]["raw_upper_out"] = source["pi"]["raw_upper_out"][:-1]
    with pytest.raises(ValueError, match="projection.*bounds"):
        build_policy(source)


def test_real_migration_training_export_reload_and_source_immutability(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.deploy.bundle import DeploymentBundle
    from unisupermimic.finetune.export_candidate import export_candidate
    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig

    cfg = PhysicalPPOConfig(
        num_envs=2, rollout_steps=4, epochs=1, batch=4, randomize=False, projected_residual=True
    )
    source = torch.load(experiment_inputs["checkpoint_path"], weights_only=True)
    learner = PhysicalPPO(
        adaptation_bundle, experiment_inputs["scene"], experiment_inputs["checkpoint_path"], cfg
    )
    try:
        assert all(torch.equal(v, learner.pi.base.state_dict()[k]) for k, v in source["pi"].items())
        assert all(not p.requires_grad for p in learner.pi.base.parameters())
        optimized = {id(p) for g in learner.opt_pi.param_groups for p in g["params"]}
        assert optimized == {id(p) for p in learner.pi.residual.parameters()}
        before = {k: v.clone() for k, v in learner.pi.state_dict().items()}
        learner.update(learner.collect(), actor=False)
        assert all(torch.equal(v, learner.pi.state_dict()[k]) for k, v in before.items())
        learner.update(learner.collect(), actor=True)
        assert any(
            not torch.equal(v, learner.pi.state_dict()[k])
            for k, v in before.items()
            if k.startswith("residual.")
        )
        assert all(torch.equal(v, learner.pi.base.state_dict()[k]) for k, v in source["pi"].items())
        saved_path = learner.save(tmp_path / "residual.pt", 2)
        saved = torch.load(saved_path, weights_only=True)
        assert saved["args"]["policy_type"] == "projected_residual_v1"
        assert "policy_type" not in learner.source["args"]
        assert all(torch.equal(v, saved["wm"][k]) for k, v in source["wm"].items())
        exported = export_candidate(adaptation_bundle, saved_path, tmp_path / "candidate")
        new_bundle = DeploymentBundle.load(exported)
        restored = PhysicalPPO(new_bundle, experiment_inputs["scene"], saved_path, cfg)
        try:
            assert all(
                torch.equal(v, restored.pi.state_dict()[k])
                for k, v in learner.pi.state_dict().items()
            )
        finally:
            restored.close()
        with pytest.raises(ValueError, match="remove.*projected"):
            PhysicalPPO(
                new_bundle, experiment_inputs["scene"], saved_path, PhysicalPPOConfig(num_envs=2)
            )
        changed = copy.deepcopy(saved)
        changed["args"]["residual_raw_lower"][0] -= 0.1
        wrong = tmp_path / "wrong_limits.pt"
        torch.save(changed, wrong)
        with pytest.raises(ValueError, match="bounds"):
            PhysicalPPO(new_bundle, experiment_inputs["scene"], wrong, cfg)
    finally:
        learner.close()
