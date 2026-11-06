"""Headless explicit motor-PD MuJoCo adapter, independent of UniLab/MDP."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .bundle import DeploymentBundle
from .controller import SensorState, finite_vector


def pd_torque(q, dq, target, kp, kd, effort) -> tuple[np.ndarray, int]:
    values = [np.asarray(v, dtype=float) for v in (q, dq, target, kp, kd, effort)]
    if any(v.shape != values[0].shape or not np.isfinite(v).all() for v in values):
        raise ValueError("PD arguments must have equal finite vector shapes")
    q, dq, target, kp, kd, effort = values
    if np.any(kp < 0) or np.any(kd < 0) or np.any(effort <= 0):
        raise ValueError("invalid PD gains or effort limits")
    desired = kp * (target - q) - kd * dq
    applied = np.clip(desired, -effort, effort)
    return applied, int(np.count_nonzero(applied != desired))


class NativeSimulator:
    """200Hz physical servo, 50Hz held policy targets. Root state is diagnostics only."""

    def __init__(self, bundle: DeploymentBundle, scene: Path, *, friction: float = 1.0,
                 mass_scale: float = 1.0, step_observer=None):
        if step_observer is not None and not callable(step_observer):
            raise ValueError("step_observer must be callable")
        self._step_observer = step_observer
        self.bundle, self.scene = bundle, Path(scene).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        model = self.model
        names = bundle.metadata["joint_names"]
        if model.nu != 29 or model.nq != 36 or model.nv != 35:
            raise ValueError("G1 mapping requires exactly 29 actuated joints and one free root")
        self.joints = np.array([model.joint(name).id for name in names])
        self.qadr, self.vadr = model.jnt_qposadr[self.joints], model.jnt_dofadr[self.joints]
        mapping = {int(j): [] for j in self.joints}
        for actuator in range(model.nu):
            joint = int(model.actuator_trnid[actuator, 0])
            if joint in mapping:
                mapping[joint].append(actuator)
        if any(len(v) != 1 for v in mapping.values()):
            raise ValueError("motor-to-joint mapping must be one-to-one")
        self.actuators = np.array([mapping[int(j)][0] for j in self.joints])
        self.pelvis = model.body("pelvis").id
        self.imu_sensors = None
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat") >= 0:
            quat_sensor, gyro_sensor = model.sensor("imu_quat"), model.sensor("imu_gyro")
            site = int(model.sensor_objid[quat_sensor.id])
            if model.sensor_objtype[quat_sensor.id] != mujoco.mjtObj.mjOBJ_SITE or model.site_bodyid[site] != self.pelvis:
                raise ValueError("primary IMU must be attached to pelvis for this policy")
            if model.sensor_objid[gyro_sensor.id] != site:
                raise ValueError("quaternion and gyro must use the same pelvis IMU")
            self.imu_sensors = (quat_sensor.id, gyro_sensor.id)
        self.kp = np.array(bundle.metadata["kp"])
        self.kd = np.array(bundle.metadata["kd"])
        self.effort = np.array(bundle.metadata["effort_limits"])
        # Convert position/motor actuators alike to unity-gear torque motors.
        model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[:] = 0
        model.actuator_gainprm[:, 0] = 1
        model.actuator_biasprm[:] = 0
        model.actuator_gear[:] = 0
        model.actuator_gear[:, 0] = 1
        for index, actuator in enumerate(self.actuators):
            model.actuator_ctrlrange[actuator] = [-self.effort[index], self.effort[index]]
            model.actuator_forcerange[actuator] = [-self.effort[index], self.effort[index]]
        model.actuator_ctrllimited[:] = 1
        model.actuator_forcelimited[:] = 1
        model.opt.timestep = 0.005
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        if not math.isfinite(friction) or not 0 < friction <= 3:
            raise ValueError("friction must be finite in (0,3]")
        if not math.isfinite(mass_scale) or not 0.5 <= mass_scale <= 1.5:
            raise ValueError("mass scale must be finite in [0.5,1.5]")
        model.geom_friction[:, 0] = friction
        model.body_mass[1:] *= mass_scale
        model.body_inertia[1:] *= mass_scale
        self.data = mujoco.MjData(model)
        if mass_scale != 1.0:
            mujoco.mj_setConst(model, self.data)
        self.saturated = 0
        self.torque_samples = 0
        self.max_torque_ratio = 0.0

    def reset(self, mode: str) -> None:
        if mode not in ("reference", "reference_zero_velocity", "standing"):
            raise ValueError("unknown simulation initialization mode")
        mujoco.mj_resetData(self.model, self.data)
        ref, data = self.bundle.reference, self.data
        data.qpos[:3] = ref["root_p"][0]
        data.qpos[3:7] = ref["root_quat"][0]
        data.qpos[self.qadr] = ref["q"][0]
        if mode == "reference":
            data.qvel[:3] = ref["root_v"][0]
            data.qvel[3:6] = ref["root_R"][0].T @ ref["root_w"][0]
            data.qvel[self.vadr] = ref["qd"][0]
        elif mode == "standing":
            data.qpos[2] = 0.76
            yaw = math.atan2(ref["root_R"][0, 1, 0], ref["root_R"][0, 0, 0])
            data.qpos[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
            data.qpos[self.qadr] = self.bundle.metadata["default_joint_positions"]
        mujoco.mj_forward(self.model, data)
        self._emit_physics_sample()

    def _emit_physics_sample(self) -> None:
        if self._step_observer is not None:
            self._step_observer(
                float(self.data.time),
                self.qpos(),
                self.data.qvel[self.vadr].copy(),
                self.data.qvel[:3].copy(),
            )

    def sensor(self) -> SensorState:
        if self.imu_sensors is None:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                    self.pelvis, velocity, 1)
            quat, gyro = self.data.qpos[3:7].copy(), velocity[:3].copy()
        else:
            quat = self.data.sensor(self.imu_sensors[0]).data.copy()
            gyro = self.data.sensor(self.imu_sensors[1]).data.copy()
        return SensorState(self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy(),
                           quat, gyro, float(self.data.time))

    def advance(self, target: np.ndarray, *, kp=None, kd=None, duration: float = 0.02) -> None:
        target = finite_vector(target, 29, "motor target")
        kp = self.kp if kp is None else finite_vector(kp, 29, "kp")
        kd = self.kd if kd is None else finite_vector(kd, 29, "kd")
        steps = round(duration / self.model.opt.timestep)
        if steps < 1 or not math.isclose(steps * self.model.opt.timestep, duration, abs_tol=1e-8):
            raise ValueError("servo duration must be a positive multiple of physics dt")
        for _ in range(steps):
            torque, clipped = pd_torque(self.data.qpos[self.qadr], self.data.qvel[self.vadr],
                                       target, kp, kd, self.effort)
            self.saturated += clipped
            self.torque_samples += 29
            self.max_torque_ratio = max(self.max_torque_ratio, float(np.max(np.abs(torque)/self.effort)))
            self.data.ctrl[self.actuators] = torque
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise ValueError("nonfinite physics state")
            self._emit_physics_sample()
        # mj_step integrates qpos after calculating xmat/sensors. Publish all
        # observation fields from the same finished timestep rather than a mix.
        mujoco.mj_forward(self.model, self.data)

    def qpos(self) -> np.ndarray:
        return np.concatenate((self.data.qpos[:7], self.data.qpos[self.qadr]))

    def failure(self, frame: int | None = None) -> str | None:
        if self.data.qpos[2] < 0.35:
            return "root height below 0.35m"
        if self.data.xmat[self.pelvis].reshape(3, 3)[2, 2] < math.cos(1.2):
            return "pelvis tilt exceeds 1.2rad"
        if frame is not None and abs(self.data.qpos[2] - self.bundle.reference["root_p"][frame, 2]) > 0.25:
            return "reference height deviation exceeds 0.25m"
        return None
