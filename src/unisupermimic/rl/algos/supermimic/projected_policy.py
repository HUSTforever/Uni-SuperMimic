"""Frozen source controller with a trainable correction to its executed action."""

import torch
from torch import nn

from .networks import Policy, run_mlp

PROJECTED_KIND = "projected_residual_v1"


class ProjectedResidualPolicy(nn.Module):
    def __init__(self, base: Policy, raw_lower, raw_upper, hidden=128):
        super().__init__()
        if not isinstance(base, Policy):
            raise ValueError("projected residual requires an unwrapped Policy")
        if type(hidden) is not int or hidden < 1:
            raise ValueError("residual hidden width must be a positive integer")
        lower = torch.as_tensor(raw_lower, dtype=torch.float64).detach().cpu()
        upper = torch.as_tensor(raw_upper, dtype=torch.float64).detach().cpu()
        if (
            lower.shape != (base.num_joints,)
            or upper.shape != lower.shape
            or not bool(torch.isfinite(lower).all())
            or not bool(torch.isfinite(upper).all())
            or not bool((lower < upper).all())
        ):
            raise ValueError("finite strictly ordered physical raw bounds required")
        lower32, upper32 = lower.float(), upper.float()
        if not bool(torch.isfinite(lower32).all() and torch.isfinite(upper32).all()):
            raise ValueError("raw bounds must be representable in float32")
        self.base = base.requires_grad_(False).eval()
        self.base.obs_norm.max_count = self.base.goal_norm.max_count = 0
        self.hist = base.hist
        self.action_smoothing = base.action_smoothing
        self.num_joints = base.num_joints
        self.register_buffer(
            "raw_lower_out", torch.nextafter(lower32, torch.full_like(lower32, -torch.inf))
        )
        self.register_buffer(
            "raw_upper_out", torch.nextafter(upper32, torch.full_like(upper32, torch.inf))
        )
        feature_dim = base.obs_norm.mean.numel() * base.hist + base.enc[-1].out_features
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ELU(), nn.Linear(hidden, base.num_joints)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.to(next(base.parameters()).device)

    @property
    def obs_norm(self):
        return self.base.obs_norm

    @property
    def goal_norm(self):
        return self.base.goal_norm

    def forward(self, obs_hist, goal):
        center = torch.clamp(
            self.base(obs_hist, goal), min=self.raw_lower_out, max=self.raw_upper_out
        )
        latent = torch.tanh(run_mlp(self.base.enc, self.base.goal_norm(goal)))
        observation = self.base.obs_norm(obs_hist).flatten(-2)
        features = torch.cat((latent, observation), -1)
        return center + self.action_smoothing * run_mlp(self.residual, features)
