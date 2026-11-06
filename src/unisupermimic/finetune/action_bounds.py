"""Differentiable training gradients outside physical raw-action bounds."""

import math

import torch


def action_boundary_loss(mean, lower, upper, std):
    """Mean squared violations in exploration-standard-deviation units."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("finite positive action standard deviation required")
    if (
        mean.ndim != 2
        or mean.numel() == 0
        or lower.shape != (mean.shape[-1],)
        or upper.shape != lower.shape
        or lower.device != mean.device
        or upper.device != mean.device
    ):
        raise ValueError("matching nonempty batch and joint bounds on the same device required")
    if (
        not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(lower).all())
        or not bool(torch.isfinite(upper).all())
        or not bool((lower < upper).all())
    ):
        raise ValueError("finite means and strictly ordered joint bounds required")
    violation = (lower - mean).clamp_min(0) + (mean - upper).clamp_min(0)
    return (violation / std).square().mean()
