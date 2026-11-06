"""PPO objectives with separate terminal and rollout-boundary semantics."""

import math

import torch


def generalized_advantages(rewards, values, next_values, terminal, done, *, gamma=0.99, lam=0.95):
    if not (0 < gamma <= 1 and 0 <= lam <= 1):
        raise ValueError("invalid discount or GAE lambda")
    if (
        rewards.ndim != 2
        or not rewards.numel()
        or any(x.shape != rewards.shape for x in (values, next_values, terminal, done))
        or terminal.dtype != torch.bool
        or done.dtype != torch.bool
        or bool((terminal & ~done).any())
        or not all(bool(torch.isfinite(x).all()) for x in (rewards, values, next_values))
    ):
        raise ValueError("matching finite T,N values and boolean episode boundaries required")
    advantage = torch.zeros_like(rewards)
    following = torch.zeros_like(rewards[0])
    for step in range(len(rewards) - 1, -1, -1):
        delta = rewards[step] + gamma * (~terminal[step]) * next_values[step] - values[step]
        following = delta + gamma * lam * (~done[step]) * following
        advantage[step] = following
    return advantage, advantage + values


def clipped_surrogate(log_prob, old_log_prob, advantage, *, clip=0.1):
    if not math.isfinite(clip) or not 0 < clip < 1:
        raise ValueError("PPO clip must be between zero and one")
    if (
        not log_prob.numel()
        or log_prob.shape != old_log_prob.shape
        or log_prob.shape != advantage.shape
        or not all(bool(torch.isfinite(x).all()) for x in (log_prob, old_log_prob, advantage))
    ):
        raise ValueError("finite matching probabilities and advantages required")
    difference = log_prob - old_log_prob
    ratio = difference.exp()
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("PPO likelihood ratio overflow")
    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
    return loss, dict(
        approx_kl=((ratio - 1) - difference).mean().detach(),
        clip_fraction=((ratio - 1).abs() > clip).float().mean().detach(),
    )
