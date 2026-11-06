"""Warm-start the unchanged deployment policy using actual physics returns."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from demo.export_policy import build_policy

from unisupermimic.deploy.bundle import _validate_policy_weights
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Critic
from unisupermimic.rl.algos.supermimic.projected_policy import (
    PROJECTED_KIND,
    ProjectedResidualPolicy,
)

from .collector import DeploymentCollector
from .export_candidate import _validate_contract
from .ppo_math import clipped_surrogate, generalized_advantages


@dataclass(frozen=True)
class PhysicalPPOConfig:
    iterations: int = 200
    num_envs: int = 32
    rollout_steps: int = 32
    epochs: int = 4
    batch: int = 256
    critic_warmup: int = 10
    checkpoint_every: int = 25
    max_episode_steps: int = 500
    seed: int = 0
    randomize: bool = True
    action_std: float = 0.03
    lr: float = 1e-5
    critic_lr: float = 1e-4
    clip: float = 0.1
    gamma: float = 0.99
    lam: float = 0.95
    target_kl: float = 0.015
    anchor_weight: float = 0.1
    action_bound_weight: float = 0.0
    balance_weight: float = 1.0
    ground_slip_weight: float = 0.35
    ankle_rate_weight: float = 0.05
    action_rate_weight: float = -0.3
    measured_velocity_multiplier: float = 1.0
    ankle_hf_weight: float = 0.0
    root_hf_weight: float = 0.0
    focus_start: int = 0
    focus_end: int = 0
    critic_clock: bool = False
    projected_residual: bool = False

    def __post_init__(self):
        for name in (
            "iterations",
            "num_envs",
            "rollout_steps",
            "epochs",
            "batch",
            "checkpoint_every",
            "max_episode_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.critic_warmup) is not int or not 0 <= self.critic_warmup < self.iterations:
            raise ValueError("critic warmup must leave at least one actor iteration")
        if type(self.seed) is not int or self.seed < 0 or type(self.randomize) is not bool:
            raise ValueError("invalid seed or randomization flag")
        for name in ("action_std", "lr", "critic_lr", "target_kl"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "anchor_weight",
            "action_bound_weight",
            "balance_weight",
            "ground_slip_weight",
            "ankle_rate_weight",
            "ankle_hf_weight",
            "root_hf_weight",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.focus_start) is not int or type(self.focus_end) is not int:
            raise ValueError("focus bounds must be integers")
        if (self.focus_start, self.focus_end) != (0, 0) and not (
            0 <= self.focus_start < self.focus_end
        ):
            raise ValueError("focus interval must be disabled or use ordered nonnegative bounds")
        if not math.isfinite(self.action_rate_weight) or self.action_rate_weight > 0:
            raise ValueError("action rate weight must be finite and nonpositive")
        if (
            not math.isfinite(self.measured_velocity_multiplier)
            or self.measured_velocity_multiplier < 1
        ):
            raise ValueError("measured velocity multiplier must be finite and at least 1")
        if not (0 < self.clip < 1 and 0 < self.gamma <= 1 and 0 <= self.lam <= 1):
            raise ValueError("invalid clipping/discount/GAE settings")
        if type(self.critic_clock) is not bool:
            raise ValueError("critic clock flag must be a boolean")
        if type(self.projected_residual) is not bool:
            raise ValueError("projected residual flag must be a boolean")
        if self.critic_clock and not 0 < self.gamma < 1:
            raise ValueError("clock critic requires discount strictly between zero and one")
        if self.ground_slip_weight and not self.balance_weight:
            raise ValueError("ground slip requires balance weight")


class PhysicalPPO:
    def __init__(self, bundle, scene, checkpoint_path, config, *, device="cpu"):
        self.cfg = config
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or (
            self.device.type == "cuda" and not torch.cuda.is_available()
        ):
            raise ValueError("available CPU or CUDA device required")
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.source = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        _validate_contract(bundle.metadata, self.source)
        _validate_policy_weights(bundle.policy_path, self.source)
        if np.any(np.asarray(bundle.metadata["context_offset"]) != 0):
            raise ValueError("physical collector requires zero context offsets")
        self.policy_args = copy.deepcopy(self.source["args"])
        self.pi = build_policy(self.source)
        source_kind = self.source["args"].get("policy_type", "standard")
        if not config.projected_residual and source_kind == PROJECTED_KIND:
            raise ValueError("cannot remove source projected residual")
        if config.projected_residual:
            metadata = bundle.metadata
            joint_lower = np.asarray(metadata["joint_lower"], dtype=np.float64)
            joint_upper = np.asarray(metadata["joint_upper"], dtype=np.float64)
            default_joint_positions = np.asarray(
                metadata["default_joint_positions"], dtype=np.float64
            )
            context_offset = np.asarray(metadata["context_offset"], dtype=np.float64)
            action_scales = np.asarray(metadata["action_scales"], dtype=np.float64)
            lo = (joint_lower - default_joint_positions - context_offset) / action_scales
            hi = (joint_upper - default_joint_positions - context_offset) / action_scales
            if source_kind == PROJECTED_KIND:
                try:
                    matches = np.array_equal(
                        np.asarray(self.source["args"]["residual_raw_lower"], dtype=np.float64),
                        lo,
                    ) and np.array_equal(
                        np.asarray(self.source["args"]["residual_raw_upper"], dtype=np.float64),
                        hi,
                    )
                except (KeyError, TypeError, ValueError):
                    matches = False
                if not matches:
                    raise ValueError(
                        "source projected residual bounds do not match bundle metadata"
                    )
            else:
                self.pi = ProjectedResidualPolicy(self.pi, lo, hi, hidden=128)
                self.policy_args.update(
                    policy_type=PROJECTED_KIND,
                    residual_hidden=128,
                    residual_raw_lower=lo.tolist(),
                    residual_raw_upper=hi.tolist(),
                )
        self.pi = self.pi.to(self.device)
        self.pi.obs_norm.max_count = self.pi.goal_norm.max_count = 0
        self.anchor = copy.deepcopy(self.pi).requires_grad_(False)
        env_kwargs = dict(
            bundle=bundle,
            scene=scene,
            num_envs=config.num_envs,
            seed=config.seed,
            randomize=config.randomize,
            max_episode_steps=config.max_episode_steps,
            balance_weight=config.balance_weight,
            ground_slip_weight=config.ground_slip_weight,
            ankle_rate_weight=config.ankle_rate_weight,
            focus_intervals=(
                [(config.focus_start, config.focus_end)] if config.focus_end else None
            ),
        )
        if config.ankle_hf_weight > 0 or config.root_hf_weight > 0:
            from .spectral import SpectralCollector

            self.env = SpectralCollector(
                **env_kwargs,
                ankle_hf_weight=config.ankle_hf_weight,
                root_hf_weight=config.root_hf_weight,
            )
        else:
            self.env = DeploymentCollector(**env_kwargs)
        metadata = bundle.metadata
        raw_lower = (
            np.asarray(metadata["joint_lower"], dtype=np.float32)
            - np.asarray(metadata["default_joint_positions"], dtype=np.float32)
            - np.asarray(metadata["context_offset"], dtype=np.float32)
        ) / np.asarray(metadata["action_scales"], dtype=np.float32)
        raw_upper = (
            np.asarray(metadata["joint_upper"], dtype=np.float32)
            - np.asarray(metadata["default_joint_positions"], dtype=np.float32)
            - np.asarray(metadata["context_offset"], dtype=np.float32)
        ) / np.asarray(metadata["action_scales"], dtype=np.float32)
        self.raw_lower = torch.as_tensor(raw_lower, device=self.device)
        self.raw_upper = torch.as_tensor(raw_upper, device=self.device)
        self.env.supermimic_mdp.cfg = replace(
            self.env.supermimic_mdp.cfg, w_action_rate=config.action_rate_weight
        )
        self.env.supermimic_mdp.ankle_measured_weight = 0.5 * config.measured_velocity_multiplier
        info = self.env.state.info["supermimic_state"]
        dimension = self._critic_input(info["s"], info["f"], info["xi"]).shape[-1]
        hidden = tuple(
            self.source["args"].get("test_hidden_dims") or self.source["args"]["critic_hidden"]
        )
        if config.critic_clock:
            from .clock import ClockCritic, with_clock_column

            self.critic = ClockCritic(dimension, hidden=hidden, gamma=config.gamma).to(self.device)
        else:
            self.critic = Critic(dimension, hidden=hidden).to(self.device)
        if "critic" in self.source:
            if config.critic_clock:
                source_dim = self.source["critic"]["norm.mean"].numel()
                source_clock = self.source.get("ppo_config", {}).get("critic_clock") is True
                if source_clock != (source_dim == dimension):
                    raise ValueError("source critic clock metadata does not match")
                state = with_clock_column(self.source["critic"], dimension)
                self.critic.load_state_dict(state)
            else:
                if self.source.get("ppo_config", {}).get("critic_clock") is True:
                    raise ValueError("cannot remove source critic clock")
                self.critic.load_state_dict(self.source["critic"])
        self.critic.norm.max_count = 0
        self.opt_pi = torch.optim.Adam(
            (p for p in self.pi.parameters() if p.requires_grad), lr=config.lr
        )
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.history = torch.zeros(config.num_envs, 10, 99, device=self.device)
        self.sim_steps = 0
        # Model construction may consume different amounts of randomness after
        # a critic migration. Keep exploration and update ordering independent.
        self.rollout_rng = torch.Generator(device=self.device).manual_seed(config.seed)
        self.update_rng = torch.Generator(device=self.device).manual_seed(config.seed + 1)

    def _critic_input(self, state, frames, context):
        mdp = self.env.supermimic_mdp
        critic_input = torch.cat(
            (
                mdp.critic_features(unflatten_state(state), frames, mdp.normalize_context(context)),
                mdp.goal(frames, 10),
            ),
            -1,
        )
        if self.cfg.critic_clock:
            end = self.env.motion.clip_end[self.env.motion.clip_id[frames]]
            remaining = (end - frames).clamp_min(0).float() / 100
            critic_input = torch.cat((critic_input, remaining.unsqueeze(-1)), -1)
        return critic_input.to(self.device)

    @torch.no_grad()
    def collect(self):
        samples = []
        for _ in range(self.cfg.rollout_steps):
            info = self.env.state.info["supermimic_state"]
            obs = self.env.state.obs["policy"].to(self.device)
            shifted = torch.cat((self.history[:, 1:], obs[:, None]), 1)
            self.history = torch.where(
                (info["ep_step"] == 0).to(self.device)[:, None, None],
                obs[:, None].expand_as(self.history),
                shifted,
            )
            goal = self.env.state.obs["future"].to(self.device)
            critic_input = self._critic_input(info["s"], info["f"], info["xi"])
            mean = self.pi(self.history, goal)
            distribution = torch.distributions.Normal(mean, self.cfg.action_std)
            latent = mean + self.cfg.action_std * torch.randn(
                mean.shape, dtype=mean.dtype, device=self.device, generator=self.rollout_rng
            )
            result = self.env.step(latent.cpu())
            transition = result.info["supermimic"]
            next_frame = transition["f"] + 1
            # Never bootstrap the automatically reset observation.
            next_input = self._critic_input(transition["s1"], next_frame, transition["xi"])
            terminal = transition["term"] | (next_frame >= self.env.motion.num_frames - 1)
            samples.append(
                dict(
                    history=self.history.clone(),
                    goal=goal,
                    actions=latent,
                    mean=mean,
                    applied_actions=transition["a"].to(self.device),
                    log_prob=distribution.log_prob(latent).sum(-1),
                    critic_input=critic_input,
                    value=self.critic(critic_input),
                    next_critic_input=next_input,
                    next_value=self.critic(next_input),
                    reward=transition["r"].to(self.device),
                    terminal=terminal.to(self.device),
                    done=transition["done"].to(self.device),
                )
            )
            if "spectral_penalty" in transition:
                samples[-1]["spectral_penalty"] = transition["spectral_penalty"].to(self.device)
                samples[-1]["spectral_cost"] = transition["spectral_cost"].to(self.device)
        batch = {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
        batch["advantage"], batch["return"] = generalized_advantages(
            batch["reward"],
            batch["value"],
            batch["next_value"],
            batch["terminal"],
            batch["done"],
            gamma=self.cfg.gamma,
            lam=self.cfg.lam,
        )
        self.sim_steps += self.cfg.rollout_steps * self.cfg.num_envs
        return batch

    def update(self, batch, *, actor=True):
        flat = {k: value.flatten(0, 1) for k, value in batch.items()}
        adv = flat["advantage"]
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
        count = len(adv)
        stats = dict(
            policy_loss=0.0,
            anchor_loss=0.0,
            action_bound_loss=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
            value_loss=0.0,
            actor_updates=0.0,
            reward_mean=float(flat["reward"].mean()),
        )
        if "spectral_penalty" in flat:
            stats["hf_penalty_mean"] = float(flat["spectral_penalty"].mean())
            stats["hf_ankle_cost_mean"] = float(flat["spectral_cost"][:, 0].mean())
            stats["hf_root_cost_mean"] = float(flat["spectral_cost"][:, 1].mean())
        actor_enabled = actor
        updates = 0
        for _ in range(self.cfg.epochs):
            order = torch.randperm(count, device=self.device, generator=self.update_rng)
            for indices in order.split(self.cfg.batch):
                value = self.critic(flat["critic_input"][indices])
                value_loss = 0.5 * (value - flat["return"][indices]).square().mean()
                self.opt_v.zero_grad(set_to_none=True)
                value_loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                if not bool(torch.isfinite(grad)):
                    raise RuntimeError("non-finite physical critic gradient")
                self.opt_v.step()
                stats["value_loss"] += float(value_loss.detach())
                updates += 1
                if not actor_enabled:
                    continue
                history, goal = flat["history"][indices], flat["goal"][indices]
                mean = self.pi(history, goal)
                log_prob = (
                    torch.distributions.Normal(mean, self.cfg.action_std)
                    .log_prob(flat["actions"][indices])
                    .sum(-1)
                )
                loss, diagnostics = clipped_surrogate(
                    log_prob, flat["log_prob"][indices], adv[indices], clip=self.cfg.clip
                )
                stats["approx_kl"] = max(stats["approx_kl"], float(diagnostics["approx_kl"]))
                if float(diagnostics["approx_kl"]) > self.cfg.target_kl:
                    actor_enabled = False
                    continue
                with torch.no_grad():
                    anchor_mean = self.anchor(history, goal)
                anchor_loss = ((mean - anchor_mean) / self.cfg.action_std).square().mean()
                actor_loss = loss + self.cfg.anchor_weight * anchor_loss
                if self.cfg.action_bound_weight > 0:
                    from .action_bounds import action_boundary_loss

                    bound_loss = action_boundary_loss(
                        mean, self.raw_lower, self.raw_upper, self.cfg.action_std
                    )
                    actor_loss = actor_loss + self.cfg.action_bound_weight * bound_loss
                    stats["action_bound_loss"] += float(bound_loss.detach())
                self.opt_pi.zero_grad(set_to_none=True)
                actor_loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
                if not bool(torch.isfinite(grad)):
                    raise RuntimeError("non-finite physical policy gradient")
                self.opt_pi.step()
                stats["policy_loss"] += float(loss.detach())
                stats["anchor_loss"] += float(anchor_loss.detach())
                stats["clip_fraction"] += float(diagnostics["clip_fraction"])
                stats["actor_updates"] += 1
        for key in ("policy_loss", "anchor_loss", "action_bound_loss", "clip_fraction"):
            stats[key] /= max(stats["actor_updates"], 1)
        stats["value_loss"] /= max(updates, 1)
        return stats

    def save(self, path, iteration):
        path = Path(path)
        checkpoint = {k: v for k, v in self.source.items() if not k.startswith("opt_")}
        checkpoint.update(
            pi={k: v.detach().cpu().clone() for k, v in self.pi.state_dict().items()},
            critic={k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
            iteration=iteration,
            source_iteration=self.source.get("iteration"),
            sim_steps=self.sim_steps,
            training_method="physical_ppo_warm_start",
            args=copy.deepcopy(self.policy_args),
            ppo_config=asdict(self.cfg),
            ppo_optimizer_policy=self.opt_pi.state_dict(),
            ppo_optimizer_value=self.opt_v.state_dict(),
            hardware_ready=False,
        )
        with path.open("xb") as stream:
            torch.save(checkpoint, stream)
        return path

    def close(self):
        self.env.close()
