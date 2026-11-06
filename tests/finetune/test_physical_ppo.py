from dataclasses import replace

import numpy as np
import pytest
import torch


def test_gae_bootstraps_timeouts_without_crossing_reset_or_terminal():
    from unisupermimic.finetune.ppo_math import generalized_advantages

    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.tensor([[3.0, 3.0], [4.0, 4.0]])
    next_values = torch.tensor([[4.0, 4.0], [10.0, 10.0]])
    terminal = torch.tensor([[False, False], [True, False]])
    done = torch.tensor([[False, True], [True, True]])
    advantage, returns = generalized_advantages(
        rewards, values, next_values, terminal, done, gamma=0.5, lam=1.0
    )
    torch.testing.assert_close(advantage, torch.tensor([[-1.0, 0.0], [-2.0, 3.0]]))
    torch.testing.assert_close(returns, torch.tensor([[2.0, 3.0], [2.0, 7.0]]))


def test_ppo_clipping_handles_positive_and_negative_advantages():
    from unisupermimic.finetune.ppo_math import clipped_surrogate

    loss, stats = clipped_surrogate(
        torch.tensor([1.3, 0.7, 1.3, 0.7]).log(),
        torch.zeros(4),
        torch.tensor([1.0, 1.0, -1.0, -1.0]),
        clip=0.1,
    )
    assert loss.item() == pytest.approx(0.1, abs=1e-6)
    assert stats["clip_fraction"].item() == 1
    assert stats["approx_kl"].item() > 0


def test_measured_velocity_weight_can_increase_without_changing_command_cost():
    from types import SimpleNamespace

    from unisupermimic.finetune.continuity import ankle_rate_cost

    mdp = SimpleNamespace(
        action_scales_values=(1.0,) * 29, motion=SimpleNamespace(qd=torch.zeros(1, 29))
    )
    state = (None, None, None, None, None, torch.full((1, 29), 2.0))
    frames = torch.zeros(1, dtype=torch.long)
    action, previous = torch.full((1, 29), 0.01), torch.zeros(1, 29)
    old, old_terms = ankle_rate_cost(mdp, state, frames, action, previous)
    mdp.ankle_measured_weight = 10.0
    new, terms = ankle_rate_cost(mdp, state, frames, action, previous)
    torch.testing.assert_close(old, torch.tensor([0.5625]))
    torch.testing.assert_close(new, torch.tensor([10.0625]))
    for key in old_terms:
        torch.testing.assert_close(old_terms[key], terms[key])


@pytest.mark.parametrize(
    "changes",
    [{"action_std": 0}, {"lr": float("nan")}, {"epochs": 0}, {"measured_velocity_multiplier": 0}],
)
def test_invalid_physical_ppo_settings_are_rejected(changes):
    from unisupermimic.finetune.physical_ppo import PhysicalPPOConfig

    with pytest.raises(ValueError):
        replace(PhysicalPPOConfig(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"ankle_hf_weight": -1},
        {"root_hf_weight": float("nan")},
        {"focus_start": 20, "focus_end": 10},
    ],
)
def test_invalid_spectral_and_focus_settings_are_rejected(changes):
    from unisupermimic.finetune.physical_ppo import PhysicalPPOConfig

    with pytest.raises(ValueError):
        replace(PhysicalPPOConfig(), **changes)


def test_ppo_spectral_opt_in_and_focus_are_recorded(adaptation_bundle, experiment_inputs):
    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig
    from unisupermimic.finetune.spectral import SpectralCollector

    cfg = PhysicalPPOConfig(
        num_envs=2,
        rollout_steps=2,
        epochs=1,
        batch=4,
        ankle_hf_weight=0.2,
        root_hf_weight=0.2,
        focus_start=10,
        focus_end=20,
    )
    learner = PhysicalPPO(
        adaptation_bundle, experiment_inputs["scene"], experiment_inputs["checkpoint_path"], cfg
    )
    try:
        assert isinstance(learner.env, SpectralCollector)
        assert learner.env.focus_intervals == ((10, 20),)
        batch = learner.collect()
        assert batch["spectral_cost"].shape == (2, 2, 2)
        assert (batch["spectral_penalty"] >= 0).all()
        stats = learner.update(batch, actor=False)
        assert stats["hf_penalty_mean"] >= 0
    finally:
        learner.close()


def test_focus_interval_can_begin_at_frame_zero():
    from unisupermimic.finetune.physical_ppo import PhysicalPPOConfig

    cfg = PhysicalPPOConfig(focus_start=0, focus_end=20)
    assert cfg.focus_start == 0 and cfg.focus_end == 20


def test_focus_interval_cannot_exceed_reference(adaptation_bundle, experiment_inputs):
    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig

    with pytest.raises(ValueError, match="focus"):
        PhysicalPPO(
            adaptation_bundle,
            experiment_inputs["scene"],
            experiment_inputs["checkpoint_path"],
            PhysicalPPOConfig(num_envs=2, focus_start=10, focus_end=10000),
        )


def test_physical_rollout_logs_latent_probability_and_pre_reset_bootstrap(
    adaptation_bundle, experiment_inputs
):
    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state

    cfg = PhysicalPPOConfig(
        num_envs=2, rollout_steps=1, max_episode_steps=1, action_std=5, randomize=False
    )
    learner = PhysicalPPO(
        adaptation_bundle, experiment_inputs["scene"], experiment_inputs["checkpoint_path"], cfg
    )
    try:
        learner.env.reset_to_frames([0, 0])
        batch = learner.collect()
        distribution = torch.distributions.Normal(batch["mean"], cfg.action_std)
        torch.testing.assert_close(
            batch["log_prob"], distribution.log_prob(batch["actions"]).sum(-1)
        )
        assert not torch.allclose(batch["actions"], batch["applied_actions"])
        assert batch["done"].all()
        transition = learner.env.state.info["supermimic"]
        frame = transition["f"] + 1
        mdp = learner.env.supermimic_mdp
        features = torch.cat(
            (
                mdp.critic_features(
                    unflatten_state(transition["s1"]),
                    frame,
                    mdp.normalize_context(transition["xi"]),
                ),
                mdp.goal(frame, 10),
            ),
            -1,
        )
        torch.testing.assert_close(batch["next_critic_input"][0], features)
        with torch.no_grad():
            torch.testing.assert_close(batch["next_value"][0], learner.critic(features))
        initial_obs = batch["history"][0]
        torch.testing.assert_close(initial_obs[:, 0], initial_obs[:, -1])
    finally:
        learner.close()


def test_critic_warmup_actor_update_and_original_export_contract(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.deploy.bundle import DeploymentBundle
    from unisupermimic.finetune.export_candidate import export_candidate
    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig

    cfg = PhysicalPPOConfig(
        num_envs=2,
        rollout_steps=4,
        epochs=2,
        batch=4,
        randomize=False,
        measured_velocity_multiplier=20,
    )
    learner = PhysicalPPO(
        adaptation_bundle, experiment_inputs["scene"], experiment_inputs["checkpoint_path"], cfg
    )
    before = {k: v.clone() for k, v in learner.pi.state_dict().items()}
    try:
        assert learner.env.supermimic_mdp.ankle_measured_weight == 10
        learner.update(learner.collect(), actor=False)
        assert all(torch.equal(v, learner.pi.state_dict()[k]) for k, v in before.items())
        stats = learner.update(learner.collect(), actor=True)
        assert all(np.isfinite(v) for v in stats.values())
        after = learner.pi.state_dict()
        assert any(not torch.equal(v, after[k]) for k, v in before.items() if "weight" in k)
        for k in before:
            if "norm." in k:
                assert torch.equal(before[k], after[k])
        saved = learner.save(tmp_path / "new.pt", 2)
        checkpoint = torch.load(saved, weights_only=True)
        original = torch.load(experiment_inputs["checkpoint_path"], weights_only=True)
        assert all(torch.equal(v, checkpoint["wm"][k]) for k, v in original["wm"].items())
        assert "ppo_optimizer_policy" in checkpoint and "opt_pi" not in checkpoint
        assert not checkpoint["hardware_ready"]
        exported = export_candidate(adaptation_bundle, saved, tmp_path / "candidate")
        result = DeploymentBundle.load(exported)
        assert result.reference.keys() == adaptation_bundle.reference.keys()
        assert not result.metadata["accepted_for_delivery"]
    finally:
        learner.close()


def test_mismatched_source_policy_is_rejected(adaptation_bundle, experiment_inputs, tmp_path):
    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig

    checkpoint = torch.load(experiment_inputs["checkpoint_path"], weights_only=True)
    checkpoint["pi"]["dec.2.bias"] += 1
    path = tmp_path / "mismatch.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="does not match"):
        PhysicalPPO(adaptation_bundle, experiment_inputs["scene"], path, PhysicalPPOConfig())


def test_source_weight_difference_hidden_from_fixed_probes_is_rejected(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from demo.export_policy import build_policy

    from unisupermimic.finetune.physical_ppo import PhysicalPPO, PhysicalPPOConfig

    checkpoint = torch.load(experiment_inputs["checkpoint_path"], weights_only=True)
    original = build_policy(checkpoint)
    features = []
    hook = original.dec[-1].register_forward_pre_hook(
        lambda module, args: features.append(args[0].detach())
    )
    generator = torch.Generator().manual_seed(20261030)
    history = torch.randn(4, 10, 99, generator=generator)
    goal = torch.randn(4, 730, generator=generator)
    with torch.no_grad():
        expected = original(history, goal)
    hook.remove()
    direction = torch.linalg.svd(features[0].double(), full_matrices=True).Vh[-1].float()
    checkpoint["pi"][f"dec.{len(original.dec) - 1}.weight"][0] += direction
    changed = build_policy(checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(changed(history, goal), expected, atol=1e-6, rtol=1e-6)
        new_history = torch.randn(4, 10, 99, generator=generator)
        assert (changed(new_history, goal) - original(new_history, goal)).abs().max() > 1e-4
    path = tmp_path / "hidden_mismatch.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        PhysicalPPO(adaptation_bundle, experiment_inputs["scene"], path, PhysicalPPOConfig())


def test_physical_training_publishes_bounded_run_and_never_overwrites(
    adaptation_bundle, experiment_inputs, tmp_path
):
    import json

    from unisupermimic.finetune.physical_ppo import PhysicalPPOConfig
    from unisupermimic.finetune.physical_train import run_physical_training

    cfg = PhysicalPPOConfig(
        iterations=2,
        critic_warmup=1,
        num_envs=2,
        rollout_steps=2,
        epochs=1,
        batch=4,
        checkpoint_every=1,
        randomize=False,
    )
    output = tmp_path / "physical"
    result = run_physical_training(
        adaptation_bundle,
        experiment_inputs["scene"],
        experiment_inputs["checkpoint_path"],
        output,
        cfg,
    )
    assert result["physical_sim_steps"] == 8
    assert result["source_checkpoint_unchanged"] and not result["hardware_ready"]
    initial = torch.load(output / "ckpt_000000.pt", weights_only=True)
    final = torch.load(output / "last.pt", weights_only=True)
    assert any(not torch.equal(v, final["pi"][k]) for k, v in initial["pi"].items())
    rows = [json.loads(s) for s in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["actor_updates"] == 0 and rows[1]["actor_updates"] > 0
    before = (output / "last.pt").read_bytes()
    with pytest.raises(FileExistsError):
        run_physical_training(
            adaptation_bundle,
            experiment_inputs["scene"],
            experiment_inputs["checkpoint_path"],
            output,
            cfg,
        )
    assert before == (output / "last.pt").read_bytes()
