"""Deployment action projection and optional contact-dynamics loss weighting."""

import math

import torch

from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner


def project_raw_action(action, default_q, scales, lower, upper):
    target = default_q + scales * action
    return (torch.clamp(target, min=lower, max=upper) - default_q) / scales


class DeploymentLearner(SuperMimicLearner):
    def __init__(self, *args, wm_ankle_velocity_weight=1.0, wm_root_velocity_weight=1.0, **kwargs):
        for name, value in (
            ("wm_ankle_velocity_weight", wm_ankle_velocity_weight),
            ("wm_root_velocity_weight", wm_root_velocity_weight),
        ):
            if not math.isfinite(value) or value < 1:
                raise ValueError(f"{name} must be finite and >= 1")
        self.wm_ankle_velocity_weight = float(wm_ankle_velocity_weight)
        self.wm_root_velocity_weight = float(wm_root_velocity_weight)
        super().__init__(*args, **kwargs)
        self.action_scales = self.env.scales.to(self.device)
        # Keep the trusted policy input normalization fixed during adaptation,
        # including WM-only warmup. World-model normalization may still update.
        self.pi.obs_norm.max_count = 0
        self.pi.goal_norm.max_count = 0

    def _wm_err(self, predicted, next_states, mask):
        if self.wm_ankle_velocity_weight == 1 and self.wm_root_velocity_weight == 1:
            return super()._wm_err(predicted, next_states, mask)
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state

        from .continuity import ANKLE_INDICES

        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        squared = (self.wm.state_error(predicted, unflatten_state(next_states)) / std).square()
        weights = torch.ones_like(std)
        weights[6:9] = self.wm_root_velocity_weight
        indices = [12 + self.mdp.num_joints + joint for joint in ANKLE_INDICES]
        weights[indices] = self.wm_ankle_velocity_weight
        weighted = (squared * weights).mean(-1)
        # Keep wm_err8 comparable with old logs; only the training loss changes.
        return (weighted * mask).sum() / next_states.shape[0], squared.mean(-1)[:, -1]

    def _imag_step(
        self, state, obs_hist, previous_action, frames, context, normalized_context, goal
    ):
        action = project_raw_action(
            self.pi(obs_hist, goal),
            self.mdp.default_q,
            self.action_scales,
            self.mdp.kin.q_lo,
            self.mdp.kin.q_hi,
        )
        next_state = self.wm.step(state, action, normalized_context)
        next_frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, context, next_frames))
        next_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        return (
            next_state,
            action,
            next_hist,
            next_frames,
            self.mdp.goal(next_frames, self.cfg.goal_frames),
        )
