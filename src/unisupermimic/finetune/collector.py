"""CPU deployment-physics collector for the existing SuperMimic learner protocol."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from unilab.base.torch_env import TorchEnvState

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.controller import quaternion_matrix
from unisupermimic.deploy.mujoco_sim import NativeSimulator
from unisupermimic.rl.algos.supermimic.geometry import flatten_state
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib


class DeploymentMDP(MDP):
    """Use the existing tracking objective plus deployment absolute survival gates."""

    balance_weight = 0.0
    ground_slip_weight = 0.0
    ankle_rate_weight = 0.0

    def reward(self, state, frames, action, previous_action, terms=False):
        result = super().reward(state, frames, action, previous_action, terms)
        penalty, extra = None, {}
        if self.balance_weight != 0:
            from .balance import stability_cost

            cost, balance_terms = stability_cost(self, state, frames)
            penalty = self.balance_weight * cost * 0.02
            extra.update(balance_terms)
            extra["balance_penalty"] = penalty
        if self.ankle_rate_weight != 0:
            from .continuity import ankle_rate_cost

            cost, ankle_terms = ankle_rate_cost(self, state, frames, action, previous_action)
            ankle_penalty = self.ankle_rate_weight * cost * 0.02
            penalty = ankle_penalty if penalty is None else penalty + ankle_penalty
            extra.update(ankle_terms)
            extra["ankle_rate_penalty"] = ankle_penalty
        if penalty is None:
            return result
        if terms:
            reward, details = result
            return reward - penalty, {**details, **extra}
        return result - penalty

    def termination(self, state, frames):
        return (
            super().termination(state, frames)
            | (state[0][..., 2] < 0.35)
            | (state[1][..., 2, 2] < math.cos(1.2))
        )

    def soft_termination(self, state, frames):
        relative_survival = 1 - super().soft_termination(state, frames)
        height_survival = torch.sigmoid((state[0][..., 2] - 0.35) / 0.02)
        tilt_survival = torch.sigmoid((state[1][..., 2, 2] - math.cos(1.2)) / 0.04)
        return 1 - relative_survival * height_survival * tilt_survival


class DeploymentCollector:
    """Synchronous CPU tensor boundary; simulator root truth is training-only.

    Each row owns a MuJoCo simulator. A delayed row holds its previous motor
    target for5ms, then applies the new target for15ms. This approximates local
    DDS actuation delay; it is not an exact model of asynchronous communications.
    """

    produces_completion_receipts = False
    device = torch.device("cpu")

    def __init__(
        self,
        bundle: DeploymentBundle,
        scene: Path,
        *,
        num_envs: int = 64,
        seed: int = 0,
        randomize: bool = True,
        max_episode_steps: int = 500,
        balance_weight: float = 0.0,
        ground_slip_weight: float = 0.0,
        ankle_rate_weight: float = 0.0,
        focus_intervals=None,
    ):
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if type(max_episode_steps) is not int or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not math.isfinite(balance_weight) or balance_weight < 0:
            raise ValueError("balance_weight must be finite and nonnegative")
        if not math.isfinite(ground_slip_weight) or ground_slip_weight < 0:
            raise ValueError("ground_slip_weight must be finite and nonnegative")
        if not math.isfinite(ankle_rate_weight) or ankle_rate_weight < 0:
            raise ValueError("ankle_rate_weight must be finite and nonnegative")
        if ground_slip_weight > 0 and not balance_weight > 0:
            raise ValueError("ground_slip_weight requires balance_weight > 0")
        self.focus_intervals = tuple(tuple(pair) for pair in (focus_intervals or ()))
        for pair in self.focus_intervals:
            if (
                len(pair) != 2
                or any(type(x) is not int for x in pair)
                or not 0 <= pair[0] < pair[1] <= len(bundle.reference["q"]) - 1
            ):
                raise ValueError("invalid focus interval")
        self.bundle, self.scene, self.num_envs = bundle, Path(scene), num_envs
        self.randomize, self.rng = randomize, np.random.default_rng(seed)
        self.max_episode_steps = max_episode_steps
        # Offline motion source is bound to the bundle's recorded CSV identity.
        # Reconstruct MotionLib-compatible reference features directly from the
        # trusted bundle to avoid a mutable CSV/fps dependency during training.
        self.kin = G1Kinematics(str(scene), "cpu")
        self.motion = self._motion_from_bundle(bundle, self.kin)
        self.default_q = torch.tensor(
            bundle.metadata["default_joint_positions"], dtype=torch.float32
        )
        self.scales = torch.tensor(bundle.metadata["action_scales"], dtype=torch.float32)
        self.supermimic_mdp = DeploymentMDP(self.kin, self.motion, self.default_q)
        self.supermimic_mdp.balance_weight = balance_weight
        self.supermimic_mdp.ground_slip_weight = ground_slip_weight
        self.supermimic_mdp.ankle_rate_weight = ankle_rate_weight
        self.supermimic_mdp.action_scales_values = tuple(
            float(x) for x in bundle.metadata["action_scales"]
        )
        self.simulators = [self._make_simulator(bundle, scene, row) for row in range(num_envs)]
        self.f = torch.zeros(num_envs, dtype=torch.long)
        self.ep_step = torch.zeros(num_envs, dtype=torch.int32)
        self.a_prev = torch.zeros(num_envs, 29)
        self.xi = torch.zeros(num_envs, 33)
        self.delay_substeps = np.zeros(num_envs, dtype=int)
        self.previous_targets = np.zeros((num_envs, 29))
        self.episode_return = torch.zeros(num_envs)
        self.state = None
        self.reset()

    def _make_simulator(self, bundle, scene, row):
        return NativeSimulator(bundle, scene)

    @staticmethod
    def _motion_from_bundle(bundle, kin):
        motion = MotionLib.__new__(MotionLib)
        motion.device = torch.device("cpu")
        ref = bundle.reference
        for name in ("q", "qd", "root_p", "root_R", "root_v", "root_w"):
            setattr(motion, name, torch.from_numpy(ref[name].copy()))
        n = len(ref["q"])
        motion.num_frames = n
        motion.names = ["deployment_reference"]
        motion.clip_start = torch.tensor([0])
        motion.clip_end = torch.tensor([n - 1])
        motion.clip_len = torch.tensor([n])
        motion.clip_id = torch.zeros(n, dtype=torch.long)
        motion.body_p, motion.body_R, motion.body_v, motion.body_w = kin.forward(
            motion.root_p, motion.root_R, motion.root_v, motion.root_w, motion.q, motion.qd
        )
        rotation_t = motion.root_R.transpose(-1, -2)
        motion.root_v_local = (rotation_t @ motion.root_v[..., None]).squeeze(-1)
        motion.root_w_local = (rotation_t @ motion.root_w[..., None]).squeeze(-1)
        motion.root_gz = motion.root_R[:, 2, 2]
        return motion

    def _sample_frames(self, count):
        last = self.motion.num_frames - 1
        frames = self.rng.integers(0, last, size=count)
        choice = self.rng.random(count)
        frames[choice < 0.25] = 0
        hard = (choice >= 0.25) & (choice < 0.5)
        if self.focus_intervals:
            focus = (choice >= 0.375) & (choice < 0.5)
            for row in np.flatnonzero(focus):
                a, b = self.focus_intervals[int(self.rng.integers(len(self.focus_intervals)))]
                frames[row] = self.rng.integers(a, b)
            hard &= ~focus
        if "sequence" in self.bundle.metadata:
            segments = self.bundle.metadata["sequence"]["segments"]
            for row in np.flatnonzero(hard):
                a, b = segments["entry" if self.rng.random() < 0.5 else "exit"]
                frames[row] = self.rng.integers(a, b)
            return frames
        lower, upper = min(700, max(last - 2, 0)), min(900, last)
        if upper > lower:
            frames[hard] = self.rng.integers(lower, upper, size=int(hard.sum()))
        return frames

    def _reset_rows(self, ids, frames, *, zero_velocity=None, delay_substeps=None):
        ref = self.bundle.reference
        for row, frame in zip(ids, frames):
            sim = self.simulators[int(row)]
            friction = float(self.rng.uniform(0.7, 1.1)) if self.randomize else 1.0
            sim.model.geom_friction[:, 0] = friction
            mujoco.mj_resetData(sim.model, sim.data)
            sim.data.qpos[:3] = ref["root_p"][frame]
            sim.data.qpos[3:7] = ref["root_quat"][frame]
            sim.data.qpos[sim.qadr] = ref["q"][frame]
            zero = (
                bool(self.rng.random() < 0.25)
                if zero_velocity is None and self.randomize
                else bool(zero_velocity)
            )
            sim.data.qvel[:3] = ref["root_v"][frame]
            sim.data.qvel[3:6] = ref["root_R"][frame].T @ ref["root_w"][frame]
            sim.data.qvel[sim.vadr] = ref["qd"][frame]
            if self.randomize:
                sim.data.qpos[:3] += self.rng.uniform([-0.01, -0.01, -0.005], [0.01, 0.01, 0.005])
                rotation_vector = self.rng.uniform(-0.03, 0.03, 3)
                angle = np.linalg.norm(rotation_vector)
                if angle > 0:
                    delta, rotated = np.empty(4), np.empty(4)
                    mujoco.mju_axisAngle2Quat(delta, rotation_vector / angle, angle)
                    mujoco.mju_mulQuat(rotated, delta, sim.data.qpos[3:7])
                    sim.data.qpos[3:7] = rotated
                    sim.data.qvel[3:6] = quaternion_matrix(rotated).T @ ref["root_w"][frame]
                sim.data.qpos[sim.qadr] += self.rng.uniform(-0.03, 0.03, 29)
                sim.data.qvel *= self.rng.uniform(0.7, 1.1)
                if not zero:
                    sim.data.qvel[:3] += self.rng.uniform(-0.05, 0.05, 3)
                    sim.data.qvel[3:6] += self.rng.uniform(-0.08, 0.08, 3)
            sim.data.qpos[sim.qadr] = np.clip(
                sim.data.qpos[sim.qadr],
                self.bundle.metadata["joint_lower"],
                self.bundle.metadata["joint_upper"],
            )
            if zero:
                sim.data.qvel[:] = 0
            mujoco.mj_forward(sim.model, sim.data)
            self.f[row], self.ep_step[row] = int(frame), 0
            self.a_prev[row] = 0
            self.xi[row] = 0
            self.xi[row, 0] = friction
            self.episode_return[row] = 0
            self.delay_substeps[row] = (
                int(self.rng.integers(0, 2))
                if delay_substeps is None and self.randomize
                else int(delay_substeps or 0)
            )
            self.previous_targets[row] = sim.data.qpos[sim.qadr]

    def _physical_state(self):
        states = [[] for _ in range(6)]
        for sim in self.simulators:
            sensor = sim.sensor()
            R = quaternion_matrix(sensor.quaternion)
            values = (
                sim.data.qpos[:3].copy(),
                R,
                sim.data.qvel[:3].copy(),
                R @ sensor.gyro,
                sensor.q,
                sensor.dq,
            )
            for destination, value in zip(states, values):
                destination.append(value)
        return tuple(torch.tensor(np.stack(values), dtype=torch.float32) for values in states)

    def _info(self, physical=None):
        physical = self._physical_state() if physical is None else physical
        return dict(
            s=flatten_state(physical),
            f=self.f.clone(),
            a_prev=self.a_prev.clone(),
            xi=self.xi.clone(),
            ep_step=self.ep_step.clone(),
        )

    def _publish(
        self, physical=None, reward=None, terminated=None, truncated=None, transition=None
    ):
        info = self._info(physical)
        mdp = self.supermimic_mdp
        state = self._physical_state() if physical is None else physical
        obs = dict(policy=mdp.obs(state, self.a_prev, self.xi, self.f), future=mdp.goal(self.f, 10))
        extra = {"supermimic_state": info}
        if transition is not None:
            extra["supermimic"] = transition
        self.state = TorchEnvState(
            obs,
            torch.zeros(self.num_envs) if reward is None else reward,
            torch.zeros(self.num_envs, dtype=torch.bool) if terminated is None else terminated,
            torch.zeros(self.num_envs, dtype=torch.bool) if truncated is None else truncated,
            extra,
        )
        return self.state

    def reset_to_frames(self, frames, *, zero_velocity=False, delay_substeps=0):
        frames = torch.as_tensor(frames, dtype=torch.long)
        if frames.shape != (self.num_envs,) or bool(
            ((frames < 0) | (frames >= self.motion.num_frames - 1)).any()
        ):
            raise ValueError("invalid reset frames")
        if delay_substeps not in (0, 1):
            raise ValueError("delay_substeps must be0or1")
        self._reset_rows(
            np.arange(self.num_envs),
            frames.numpy(),
            zero_velocity=zero_velocity,
            delay_substeps=delay_substeps,
        )
        return self._publish()

    def reset(self, env_ids=None):
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids, dtype=int)
        self._reset_rows(ids, self._sample_frames(len(ids)))
        self._publish()
        return self.state.obs

    def init_state(self):
        return self.state

    @torch.no_grad()
    def step(self, actions):
        from .learner import project_raw_action

        if (
            actions.shape != (self.num_envs, 29)
            or actions.device.type != "cpu"
            or not bool(torch.isfinite(actions).all())
        ):
            raise ValueError("actions must be finite CPU[num_envs,29]")
        before = self._info()
        applied = project_raw_action(
            actions, self.default_q, self.scales, self.kin.q_lo, self.kin.q_hi
        )
        targets = (self.default_q + self.scales * applied).numpy()
        for row, sim in enumerate(self.simulators):
            if self.delay_substeps[row] and self.ep_step[row] > 0:
                sim.advance(self.previous_targets[row], duration=0.005)
                sim.advance(targets[row], duration=0.015)
            else:
                sim.advance(targets[row])
            self.previous_targets[row] = targets[row]
        physical = self._physical_state()
        next_f = self.f + 1
        reward = self.supermimic_mdp.reward(physical, next_f, applied, self.a_prev)
        hard = self.supermimic_mdp.termination(physical, next_f)
        self.ep_step += 1
        timeout = self.ep_step >= self.max_episode_steps
        clip_end = next_f >= self.motion.num_frames - 1
        done = hard | timeout | clip_end
        self.episode_return += reward
        transition = dict(
            s=before["s"],
            s1=flatten_state(physical),
            a=applied.clone(),
            a_prev=before["a_prev"],
            xi=before["xi"],
            r=reward.clone(),
            f=before["f"],
            ep_step=before["ep_step"],
            term=hard.clone(),
            done=done.clone(),
        )
        self.f = next_f
        self.a_prev = applied.clone()
        if bool(done.any()):
            transition["ep_len"] = self.ep_step[done].float().clone()
            transition["ep_ret_norm"] = (
                self.episode_return[done] / self.ep_step[done] / self.supermimic_mdp.max_step_reward
            ).clone()
            ids = done.nonzero().flatten().numpy()
            self._reset_rows(ids, self._sample_frames(len(ids)))
            physical = self._physical_state()
        return self._publish(physical, reward, hard, timeout | clip_end, transition)

    def close(self):
        self.simulators.clear()
