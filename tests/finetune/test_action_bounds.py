import math

import pytest
import torch


def test_boundary_penalty_has_exact_loss_and_gradient_toward_feasible_mean():
    from unisupermimic.finetune.action_bounds import action_boundary_loss

    mean = torch.tensor([[-2.0, 0.0, 3.0]], requires_grad=True)
    lower, upper = torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0])
    loss = action_boundary_loss(mean, lower, upper, 0.5)
    assert loss.item() == pytest.approx(20 / 3)
    loss.backward()
    torch.testing.assert_close(mean.grad, torch.tensor([[-8 / 3, 0.0, 16 / 3]]))
    moved = mean.detach() - 0.1 * mean.grad
    assert action_boundary_loss(moved, lower, upper, 0.5).item() < loss.item()


def test_feasible_means_including_exact_endpoints_have_no_penalty_or_gradient():
    from unisupermimic.finetune.action_bounds import action_boundary_loss

    mean = torch.tensor([[-2.0, 1.0, 0.5]], requires_grad=True)
    loss = action_boundary_loss(mean, torch.tensor([-2.0, -1.0, 0.0]), torch.ones(3), 0.03)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(mean.grad, torch.zeros_like(mean))


@pytest.mark.parametrize("std", [0, -1, math.inf, math.nan])
def test_boundary_penalty_rejects_invalid_exploration_scale(std):
    from unisupermimic.finetune.action_bounds import action_boundary_loss

    with pytest.raises(ValueError):
        action_boundary_loss(torch.zeros(2, 3), -torch.ones(3), torch.ones(3), std)


@pytest.mark.parametrize("lower", [torch.ones(3), torch.zeros(2), torch.full((3,), math.nan)])
def test_boundary_penalty_rejects_invalid_joint_bounds(lower):
    from unisupermimic.finetune.action_bounds import action_boundary_loss

    with pytest.raises(ValueError):
        action_boundary_loss(torch.zeros(2, 3), lower, torch.ones(3), 0.03)


@pytest.mark.parametrize("weight", [-1, math.inf, math.nan])
def test_boundary_weight_rejects_negative_or_nonfinite_values(weight):
    from unisupermimic.finetune.physical_ppo import PhysicalPPOConfig

    with pytest.raises(ValueError):
        PhysicalPPOConfig(action_bound_weight=weight)


def test_real_ppo_boundary_gradient_is_optional_and_preserves_warmup_and_export(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.deploy.bundle import DeploymentBundle
    from unisupermimic.finetune.export_candidate import export_candidate
    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig

    initial_actions = []
    for weight in (0.0, 1.0):
        cfg = PhysicalPPOConfig(
            num_envs=2,
            rollout_steps=1,
            epochs=1,
            batch=2,
            randomize=False,
            anchor_weight=0,
            action_bound_weight=weight,
        )
        learner = PhysicalPPO(
            adaptation_bundle,
            experiment_inputs["scene"],
            experiment_inputs["checkpoint_path"],
            cfg,
        )
        try:
            learner.env.reset_to_frames([0, 0])
            j = 5
            meta = adaptation_bundle.metadata
            limit = (meta["joint_upper"][j] - meta["default_joint_positions"][j]) / meta[
                "action_scales"
            ][j]
            with torch.no_grad():
                learner.pi.dec[-1].weight.zero_()
                learner.pi.dec[-1].bias.zero_()
                learner.pi.dec[-1].bias[j] = (limit + 0.5) / learner.pi.action_smoothing
            before = {k: v.clone() for k, v in learner.pi.state_dict().items()}
            previous_bias = learner.pi.dec[-1].bias[j].item()
            batch = learner.collect()
            initial_actions.append(batch["actions"].clone())
            assert batch["mean"][0, 0, j].item() == pytest.approx(limit + 0.5)
            batch["advantage"].zero_()  # Isolate the auxiliary gradient on a real rollout.
            learner.update(batch, actor=False)
            assert all(torch.equal(v, learner.pi.state_dict()[k]) for k, v in before.items())
            stats = learner.update(batch, actor=True)
            if weight:
                assert stats["action_bound_loss"] == pytest.approx(9.5785440613, rel=1e-5)
                assert learner.pi.dec[-1].bias[j].item() < previous_bias
            else:
                assert stats["action_bound_loss"] == 0
                assert all(torch.equal(v, learner.pi.state_dict()[k]) for k, v in before.items())
            for k, v in before.items():
                if k.startswith(("obs_norm.", "goal_norm.")):
                    assert torch.equal(v, learner.pi.state_dict()[k])
            checkpoint = learner.save(tmp_path / f"bound_{weight}.pt", 1)
            saved = torch.load(checkpoint, weights_only=True)
            original = torch.load(experiment_inputs["checkpoint_path"], weights_only=True)
            assert all(torch.equal(v, saved["wm"][k]) for k, v in original["wm"].items())
            exported = export_candidate(adaptation_bundle, checkpoint, tmp_path / f"out_{weight}")
            assert not DeploymentBundle.load(exported).metadata["accepted_for_delivery"]
        finally:
            learner.close()
    assert torch.equal(*initial_actions)
