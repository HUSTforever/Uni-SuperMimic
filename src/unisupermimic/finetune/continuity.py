"""Differentiable ankle command/velocity regularization in physical units."""

from __future__ import annotations

from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES

ANKLE_INDICES = tuple(i for i, name in enumerate(G1_JOINT_NAMES) if "ankle" in name)


def ankle_rate_cost(mdp, state, frames, action, previous_action):
    """Penalize extra command and measured ankle speed while keeping reference motion.

    Scales are immutable Python values so copied MDPs create constants on the
    action's device. This is a training objective, never a runtime action filter.
    """
    scales = action.new_tensor(mdp.action_scales_values)
    requested_velocity = (action - previous_action) * scales / 0.02
    reference = mdp.motion.qd[frames][..., ANKLE_INDICES]
    command_error = ((requested_velocity[..., ANKLE_INDICES] - reference) / 2).square().mean(-1)
    measured_error = ((state[5][..., ANKLE_INDICES] - reference) / 2).square().mean(-1)
    measured_weight = getattr(mdp, "ankle_measured_weight", 0.5)
    return command_error + measured_weight * measured_error, dict(
        ankle_command_velocity_error=command_error,
        ankle_measured_velocity_error=measured_error,
    )
