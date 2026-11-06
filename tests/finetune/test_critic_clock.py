import math

import pytest
import torch


def test_clock_gate_matches_finite_constant_reward_and_terminal_zero():
    from unisupermimic.finetune.clock import ClockCritic
    from unisupermimic.finetune.ppo_math import generalized_advantages

    critic = ClockCritic(3, hidden=(4,), gamma=0.99)
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        critic.net[-1].bias.fill_(10)
    x = torch.zeros(4, 3)
    x[:, -1] = torch.tensor([3, 2, 1, 0]) / 100
    values = critic(x)
    expected = torch.tensor([10 * (1 - 0.99**n) for n in (3, 2, 1, 0)])
    torch.testing.assert_close(values, expected, atol=1e-6, rtol=1e-6)
    terminal = torch.tensor([[False], [False], [True]])
    advantage, _ = generalized_advantages(
        torch.full((3, 1), 0.1), values[:3, None], values[1:, None], terminal, terminal
    )
    torch.testing.assert_close(advantage, torch.zeros_like(advantage), atol=2e-6, rtol=0)


def test_clock_weight_migration_keeps_all_existing_columns_and_buffers():
    from unisupermimic.finetune.clock import with_clock_column
    from unisupermimic.rl.algos.supermimic.networks import Critic

    old = Critic(3, hidden=(4,)).state_dict()
    new = with_clock_column(old, 4)
    torch.testing.assert_close(new["net.0.weight"][:, :3], old["net.0.weight"], atol=0, rtol=0)
    assert (new["net.0.weight"][:, -1] == 0).all()
    assert new["norm.mean"][-1] == 0 and new["norm.var"][-1] == 1
    assert old["net.0.weight"].shape == (4, 3)
    for key in old:
        if key not in ("net.0.weight", "norm.mean", "norm.var"):
            torch.testing.assert_close(new[key], old[key], atol=0, rtol=0)
    with pytest.raises(ValueError):
        with_clock_column(old, 2)


@pytest.mark.parametrize("gamma", [0, 1, math.nan])
def test_invalid_clock_discount_is_rejected(gamma):
    from unisupermimic.finetune.clock import ClockCritic

    with pytest.raises(ValueError):
        ClockCritic(3, hidden=(4,), gamma=gamma)


def test_real_ppo_clock_migrates_and_keeps_policy_export(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.finetune.export_candidate import export_candidate
    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig

    cfg = PhysicalPPOConfig(num_envs=2, rollout_steps=2, epochs=1, batch=4, randomize=False)
    legacy = PhysicalPPO(
        adaptation_bundle, experiment_inputs["scene"], experiment_inputs["checkpoint_path"], cfg
    )
    try:
        source = legacy.save(tmp_path / "legacy.pt", 0)
        old = {k: v.clone() for k, v in legacy.critic.state_dict().items()}
        legacy_batch = legacy.collect()
    finally:
        legacy.close()
    clock = PhysicalPPO(
        adaptation_bundle,
        experiment_inputs["scene"],
        source,
        PhysicalPPOConfig(
            num_envs=2, rollout_steps=2, epochs=1, batch=4, randomize=False, critic_clock=True
        ),
    )
    try:
        new = clock.critic.state_dict()
        torch.testing.assert_close(new["net.0.weight"][:, :-1], old["net.0.weight"], atol=0, rtol=0)
        batch = clock.collect()
        torch.testing.assert_close(batch["actions"], legacy_batch["actions"], atol=0, rtol=0)
        assert batch["critic_input"].shape[-1] == old["norm.mean"].numel() + 1
        assert (batch["critic_input"][..., -1] >= 0).all()
        before = {k: v.clone() for k, v in clock.pi.state_dict().items()}
        clock.update(batch, actor=False)
        assert all(torch.equal(v, clock.pi.state_dict()[k]) for k, v in before.items())
        saved = clock.save(tmp_path / "clock.pt", 1)
        assert torch.load(saved, weights_only=True)["ppo_config"]["critic_clock"]
        export_candidate(adaptation_bundle, saved, tmp_path / "candidate")
        reloaded = PhysicalPPO(
            adaptation_bundle,
            experiment_inputs["scene"],
            saved,
            PhysicalPPOConfig(num_envs=2, critic_clock=True),
        )
        reloaded.close()
        with pytest.raises(ValueError, match="remove"):
            PhysicalPPO(
                adaptation_bundle,
                experiment_inputs["scene"],
                saved,
                PhysicalPPOConfig(num_envs=2),
            )
    finally:
        clock.close()


def test_declared_clock_source_must_really_have_the_clock_input(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig
    from unisupermimic.rl.algos.supermimic.networks import Critic

    checkpoint = torch.load(experiment_inputs["checkpoint_path"], weights_only=True)
    checkpoint["critic"] = Critic(914, hidden=(8,)).state_dict()
    checkpoint["ppo_config"] = {"critic_clock": True}
    source = tmp_path / "wrong_clock.pt"
    torch.save(checkpoint, source)
    with pytest.raises(ValueError, match="clock metadata"):
        PhysicalPPO(
            adaptation_bundle,
            experiment_inputs["scene"],
            source,
            PhysicalPPOConfig(num_envs=2, critic_clock=True),
        )


def test_clock_bootstraps_pre_reset_terminal_frame_not_new_episode(
    adaptation_bundle, experiment_inputs
):
    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig

    learner = PhysicalPPO(
        adaptation_bundle,
        experiment_inputs["scene"],
        experiment_inputs["checkpoint_path"],
        PhysicalPPOConfig(num_envs=2, rollout_steps=1, randomize=False, critic_clock=True),
    )
    try:
        last = learner.env.motion.num_frames - 1
        learner.env.reset_to_frames([last - 1, last - 1])
        batch = learner.collect()
        torch.testing.assert_close(batch["critic_input"][0, :, -1], torch.full((2,), 0.01))
        assert (batch["next_critic_input"][..., -1] == 0).all()
        assert batch["terminal"].all() and batch["done"].all()
        assert (batch["next_value"] == 0).all()
        assert (learner.env.f < last).all()
    finally:
        learner.close()
