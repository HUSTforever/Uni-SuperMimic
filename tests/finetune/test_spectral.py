import math

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("frequency", [2.0, 8.0, 12.0, 40.0])
def test_highpass_matches_fourth_order_analytic_gain(frequency):
    from unisupermimic.finetune.spectral import HighpassBank

    bank = HighpassBank(1)
    bank.reset(0, np.zeros(5))
    t = np.arange(1200) / 200
    signal = np.sin(2 * math.pi * frequency * t)
    result = np.array([bank.observe(0, np.full(5, x))[0] for x in signal])
    expected = 1 / math.sqrt(
        1 + (math.tan(math.pi * 8 / 200) / math.tan(math.pi * frequency / 200)) ** 8
    )
    gain = np.sqrt(np.mean(result[400:] ** 2) / np.mean(signal[400:] ** 2))
    assert gain == pytest.approx(expected, abs=1e-5)


def test_constant_and_ramp_initialization_do_not_create_false_high_frequency():
    from unisupermimic.finetune.spectral import HighpassBank

    bank = HighpassBank(2)
    initial = np.linspace(-2, 2, 5)
    velocity = np.linspace(0.1, 0.5, 5)
    bank.reset(0, initial)
    bank.reset(1, initial, velocity)
    for step in range(1, 501):
        np.testing.assert_allclose(bank.observe(0, initial), 0, atol=1e-12)
        np.testing.assert_allclose(bank.observe(1, initial + velocity * step / 200), 0, atol=1e-12)


def test_reset_isolates_rows_and_rejects_invalid_filter_samples():
    from unisupermimic.finetune.spectral import HighpassBank

    a, b = HighpassBank(2), HighpassBank(2)
    for bank in (a, b):
        for row in (0, 1):
            bank.reset(row, np.zeros(5))
    for step in range(20):
        value = np.full(5, math.sin(step))
        a.observe(1, value)
        b.observe(1, value)
    a.reset(0, np.full(5, 1000))
    np.testing.assert_array_equal(a.observe(1, np.ones(5)), b.observe(1, np.ones(5)))
    with pytest.raises(ValueError):
        a.observe(1, np.full(5, np.nan))
    with pytest.raises(ValueError):
        HighpassBank(1, cutoff_hz=100)


@pytest.mark.parametrize("delay", [0, 1])
def test_spectral_observer_preserves_physics_and_episode_reward_across_resets(
    adaptation_bundle, experiment_inputs, delay
):
    from unisupermimic.finetune.collector import DeploymentCollector
    from unisupermimic.finetune.spectral import SpectralCollector

    kwargs = dict(num_envs=2, randomize=False, seed=13, max_episode_steps=3)
    baseline = DeploymentCollector(adaptation_bundle, experiment_inputs["scene"], **kwargs)
    spectral = SpectralCollector(
        adaptation_bundle,
        experiment_inputs["scene"],
        ankle_hf_weight=0.2,
        root_hf_weight=0.2,
        **kwargs,
    )
    reward_sum, penalty_sum = torch.zeros(2), torch.zeros(2)
    generator = torch.Generator().manual_seed(42)
    try:
        for collector in (baseline, spectral):
            collector.reset_to_frames([0, 20], delay_substeps=delay)
        for _ in range(9):
            actions = torch.randn(2, 29, generator=generator) * 0.05
            old = baseline.step(actions)
            new = spectral.step(actions)
            ot, nt = old.info["supermimic"], new.info["supermimic"]
            for key in ("s", "s1", "a", "done", "term"):
                torch.testing.assert_close(ot[key], nt[key], atol=0, rtol=0)
            np.testing.assert_array_equal(spectral.spectral_samples, [4, 4])
            penalty = nt["spectral_penalty"]
            assert torch.isfinite(penalty).all() and (penalty >= 0).all()
            torch.testing.assert_close(new.reward, old.reward - penalty)
            torch.testing.assert_close(nt["r"], new.reward)
            reward_sum += old.reward
            penalty_sum += penalty
            done = nt["done"]
            if bool(done.any()):
                expected = (reward_sum[done] - penalty_sum[done]) / nt["ep_len"] / 0.1
                torch.testing.assert_close(nt["ep_ret_norm"], expected, atol=2e-5, rtol=2e-5)
                reward_sum[done], penalty_sum[done] = 0, 0
            torch.testing.assert_close(
                spectral.episode_return[~done],
                (reward_sum - penalty_sum)[~done],
                atol=2e-6,
                rtol=2e-6,
            )
    finally:
        baseline.close()
        spectral.close()
