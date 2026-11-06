"""Finite-horizon training values with an explicit, privileged clip clock."""

import math

import torch

from unisupermimic.rl.algos.supermimic.networks import Critic


class ClockCritic(Critic):
    def __init__(self, in_dim, hidden=(256, 128), *, gamma=0.99):
        if not math.isfinite(gamma) or not 0 < gamma < 1:
            raise ValueError(
                "clock critic requires a finite discount strictly between zero and one"
            )
        super().__init__(in_dim, hidden=hidden)
        self.log_gamma = math.log(gamma)

    def forward(self, x):
        remaining = (x[..., -1] * 100).clamp_min(0)
        horizon = -torch.expm1(remaining * self.log_gamma)
        return super().forward(x) * horizon


def with_clock_column(state, in_dim):
    """Clone a critic state, optionally appending one initially inactive input."""
    previous = state["norm.mean"].numel()
    if in_dim not in (previous, previous + 1):
        raise ValueError("critic clock migration only permits one additional input column")
    result = {key: value.detach().clone() for key, value in state.items()}
    if in_dim == previous:
        return result
    weight = result["net.0.weight"]
    if weight.shape[-1] != previous or result["norm.var"].shape != (previous,):
        raise ValueError("critic normalization and input dimensions do not match")
    result["net.0.weight"] = torch.cat((weight, weight.new_zeros(weight.shape[0], 1)), -1)
    result["norm.mean"] = torch.cat((result["norm.mean"], result["norm.mean"].new_zeros(1)))
    result["norm.var"] = torch.cat((result["norm.var"], result["norm.var"].new_ones(1)))
    return result
