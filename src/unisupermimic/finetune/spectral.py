"""Causal training costs from unmodified 200Hz physical tracking errors."""

from __future__ import annotations

import math

import numpy as np
import torch

from unisupermimic.deploy.mujoco_sim import NativeSimulator

from .collector import DeploymentCollector
from .continuity import ANKLE_INDICES


class HighpassBank:
    """Independent fourth-order Butterworth highpasses, transposed direct form II.

    Biquad coefficients: https://www.w3.org/TR/audio-eq-cookbook/#hpf
    State is initialized for the local linear continuation, avoiding a reset
    impulse when an episode starts at a nonzero pose or velocity error.
    """

    def __init__(self, num_envs, channels=5, cutoff_hz=8.0, sample_hz=200.0):
        if type(num_envs) is not int or num_envs < 1 or type(channels) is not int or channels < 1:
            raise ValueError("positive integer environment/channel counts required")
        if not (
            math.isfinite(cutoff_hz) and math.isfinite(sample_hz) and 0 < cutoff_hz < sample_hz / 2
        ):
            raise ValueError("finite cutoff between zero and Nyquist required")
        self.dt = 1 / sample_hz
        self.channels = channels
        omega = 2 * math.pi * cutoff_hz / sample_hz
        cosine, sine = math.cos(omega), math.sin(omega)
        coefficients = []
        for angle in (math.pi / 8, 3 * math.pi / 8):
            quality = 1 / (2 * math.cos(angle))
            alpha = sine / (2 * quality)
            a0 = 1 + alpha
            b0 = (1 + cosine) / (2 * a0)
            coefficients.append((b0, -2 * b0, b0, -2 * cosine / a0, (1 - alpha) / a0))
        self.coefficients = np.asarray(coefficients)
        self.state = np.zeros((num_envs, 2, 2, channels))
        self.initialized = np.zeros(num_envs, dtype=bool)

    def _values(self, row, values):
        if type(row) is not int or not 0 <= row < len(self.state):
            raise ValueError("invalid filter row")
        values = np.asarray(values, dtype=float)
        if values.shape != (self.channels,) or not np.isfinite(values).all():
            raise ValueError("finite matching filter channels required")
        return values

    def reset(self, row, values, velocity=None):
        values = self._values(row, values)
        velocity = np.zeros(self.channels) if velocity is None else self._values(row, velocity)
        self.state[row] = 0
        b0, _, b2, _, _ = self.coefficients[0]
        self.state[row, 0, 0] = -b0 * (values + velocity * self.dt)
        self.state[row, 0, 1] = b2 * values
        self.initialized[row] = True

    def observe(self, row, values):
        values = self._values(row, values)
        if not self.initialized[row]:
            raise ValueError("filter row must be reset before observing")
        for section, (b0, b1, b2, a1, a2) in enumerate(self.coefficients):
            delay = self.state[row, section]
            output = b0 * values + delay[0]
            delay[0] = b1 * values - a1 * output + delay[1]
            delay[1] = b2 * values - a2 * output
            values = output
        if not np.isfinite(values).all():
            raise ValueError("nonfinite highpass result")
        return values


class SpectralCollector(DeploymentCollector):
    def __init__(self, bundle, scene, *, ankle_hf_weight=0.0, root_hf_weight=0.0, **kwargs):
        if any(not math.isfinite(w) or w < 0 for w in (ankle_hf_weight, root_hf_weight)):
            raise ValueError("spectral weights must be finite and nonnegative")
        self.ankle_hf_weight, self.root_hf_weight = ankle_hf_weight, root_hf_weight
        count = kwargs.get("num_envs", 64)
        self.highpass = HighpassBank(count)
        self.spectral_samples = np.zeros(count, dtype=int)
        self._spectral_sum = np.zeros((count, 2))
        self._step_frame = np.zeros(count, dtype=int)
        self._last_time = np.zeros(count)
        self._collecting = False
        self._reference_features = np.c_[
            bundle.reference["q"][:, ANKLE_INDICES], bundle.reference["root_p"][:, 2]
        ].astype(float)
        super().__init__(bundle, scene, **kwargs)

    def _make_simulator(self, bundle, scene, row):
        return NativeSimulator(
            bundle,
            scene,
            step_observer=lambda t, q, dq, v: self._observe(row, t, q),
        )

    @staticmethod
    def _features(qpos):
        return np.r_[qpos[7 + np.asarray(ANKLE_INDICES)], qpos[2]]

    def _reset_rows(self, ids, frames, **kwargs):
        super()._reset_rows(ids, frames, **kwargs)
        for row, frame in zip(ids, frames):
            row, frame = int(row), int(frame)
            sim = self.simulators[row]
            error = self._features(sim.qpos()) - self._reference_features[frame]
            velocity = np.r_[sim.data.qvel[sim.vadr[list(ANKLE_INDICES)]], sim.data.qvel[2]]
            reference_velocity = (
                self._reference_features[frame + 1] - self._reference_features[frame]
            ) / 0.02
            self.highpass.reset(row, error, velocity - reference_velocity)
            self._last_time[row] = float(sim.data.time)

    def _observe(self, row, time, qpos):
        if not self._collecting or not math.isclose(
            time - self._last_time[row], 0.005, abs_tol=1e-8
        ):
            raise ValueError("spectral observer requires consecutive actual 5ms steps")
        self._last_time[row] = time
        self.spectral_samples[row] += 1
        if self.spectral_samples[row] > 4:
            raise ValueError("more than four physics samples per control transition")
        frame = self._step_frame[row]
        alpha = self.spectral_samples[row] / 4
        reference = (1 - alpha) * self._reference_features[
            frame
        ] + alpha * self._reference_features[frame + 1]
        high = self.highpass.observe(row, self._features(qpos) - reference)
        self._spectral_sum[row] += [np.mean((high[:4] / 0.005) ** 2), (high[4] / 0.001) ** 2]

    @torch.no_grad()
    def step(self, actions):
        self.spectral_samples.fill(0)
        self._spectral_sum.fill(0)
        self._step_frame[:] = self.f.numpy()
        self._collecting = True
        try:
            result = super().step(actions)
        finally:
            self._collecting = False
        if not np.all(self.spectral_samples == 4):
            raise ValueError("exactly four physics samples per control transition required")
        cost = self._spectral_sum / 4
        penalty = torch.tensor(
            0.02 * (self.ankle_hf_weight * cost[:, 0] + self.root_hf_weight * cost[:, 1]),
            dtype=torch.float32,
        )
        transition = result.info["supermimic"]
        transition["spectral_cost"] = torch.tensor(cost, dtype=torch.float32)
        transition["spectral_penalty"] = penalty
        transition["r"] = transition["r"] - penalty
        done = transition["done"]
        self.episode_return[~done] -= penalty[~done]
        if bool(done.any()):
            # Previous costs already reduced episode_return on earlier steps;
            # auto-reset has saved its final return and cleared done rows.
            transition["ep_ret_norm"] -= (
                penalty[done] / transition["ep_len"] / self.supermimic_mdp.max_step_reward
            )
        self.state = result.replace(reward=result.reward - penalty)
        return self.state
