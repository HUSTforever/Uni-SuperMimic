"""Differentiable deployment stability proxies; no force/contact observations."""

from __future__ import annotations

import torch

# Centers of the official G1's four spherical sole collision elements, radius .005m.
SOLE_POINTS = (
    (-0.05, 0.025, -0.03),
    (-0.05, -0.025, -0.03),
    (0.12, 0.03, -0.03),
    (0.12, -0.03, -0.03),
)


def stability_cost(mdp, state, frames):
    """Track reference lean/gyro and discourage movement of reference stance points.

    Stance labels use only reference geometry and velocity. Fast pivots and swing
    points are excluded, and raising the actual foot cannot disable its cost.
    These are kinematic proxies, not a contact-force or balance guarantee.
    """
    p, R, v, w, q, qd = state
    motion = mdp.motion
    body_p, body_R, body_v, body_w = mdp.kin.forward(p, R, v, w, q, qd)
    names = [mdp.kin.body_names[i] for i in mdp.kin.tracked_idx]
    feet = [names.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    points = q.new_tensor(SOLE_POINTS)

    def sole_features(bp, bR, bv, bw):
        offset = (bR[..., feet, None, :, :] @ points[..., None]).squeeze(-1)
        pos = bp[..., feet, None, :] + offset
        vel = bv[..., feet, None, :] + torch.cross(
            bw[..., feet, None, :].expand_as(offset), offset, dim=-1
        )
        return pos, vel

    _, actual_v = sole_features(body_p, body_R, body_v, body_w)
    ref_p, ref_v = sole_features(
        motion.body_p[frames], motion.body_R[frames], motion.body_v[frames], motion.body_w[frames]
    )
    stance = ((ref_p[..., 2] - 0.005 < 0.02) & (ref_v.square().sum(-1) < 0.25**2)).to(q.dtype)
    slip = (stance * actual_v[..., :2].square().sum(-1)).sum((-1, -2)) / stance.sum((-1, -2)).clamp(
        min=1
    )
    gravity = (R[..., 2, :] - motion.root_R[frames, 2, :]).square().sum(-1)
    local_w = (R.transpose(-1, -2) @ w[..., None]).squeeze(-1)
    ref_local_w = (
        motion.root_R[frames].transpose(-1, -2) @ motion.root_w[frames, ..., None]
    ).squeeze(-1)
    angular = (local_w - ref_local_w).square().sum(-1)
    ground_weight = getattr(mdp, "ground_slip_weight", 0.0)
    ground_slip = torch.zeros_like(slip)
    if ground_weight:
        # Fast horizontal reference motion may itself be a ground slide. Only
        # vertical swing speed disables this optional, stricter ground cost.
        ground = ((ref_p[..., 2] - 0.005 < 0.02) & (ref_v[..., 2].abs() < 0.25)).to(q.dtype)
        ground_slip = (ground * actual_v[..., :2].square().sum(-1)).sum((-1, -2)) / ground.sum(
            (-1, -2)
        ).clamp(min=1)
    terms = dict(
        gravity_error=gravity,
        angular_velocity_error=angular,
        stance_slip=slip,
        ground_slip=ground_slip,
    )
    return 4 * gravity + 0.15 * angular + slip + ground_weight * ground_slip, terms
