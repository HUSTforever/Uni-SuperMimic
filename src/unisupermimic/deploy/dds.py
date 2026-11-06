"""Unitree HG wire adapter restricted to local simulation endpoints."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from .bundle import G1_JOINT_NAMES
from .controller import SensorState, finite_vector, quaternion_matrix

# Pinned SDK G1 29-DoF PR indices: legs 0..11, waist 12..14, arms 15..28.
SDK_MOTOR_INDEX = {name: index for index, name in enumerate(G1_JOINT_NAMES)}


def validate_sim_endpoint(interface: str, domain: int) -> None:
    if interface != "lo" or type(domain) is not int or not 1 <= domain <= 200:
        raise ValueError("simulation transport requires interface='lo' and domain in 1..200")


def initialize_dds(domain: int, interface: str = "lo") -> None:
    validate_sim_endpoint(interface, domain)
    from unitree_sdk2py.core import channel

    if getattr(channel.ChannelFactory, "_ChannelFactory__initialized", False):
        raise ValueError("DDS must initialize in a fresh dedicated simulation process")
    # SDK interface selection alone leaves discovery defaults enabled. Pin both
    # data and discovery to loopback, independent of machine-wide DDS settings.
    channel.ChannelConfigHasInterface = '''<CycloneDDS><Domain Id="any">
      <General><Interfaces><NetworkInterface name="lo" multicast="false"/>
      </Interfaces><AllowMulticast>false</AllowMulticast></General>
      <Discovery><ParticipantIndex>auto</ParticipantIndex><Peers>
      <Peer Address="127.0.0.1"/></Peers></Discovery>
      </Domain></CycloneDDS>'''
    channel.ChannelFactoryInitialize(domain, "lo")


class TickClock:
    """Unwrap local simulator uint32 millisecond ticks; receive age is separate."""

    def __init__(self):
        self.last: int | None = None
        self.elapsed = 0.0

    def update(self, tick: int) -> float:
        if not isinstance(tick, int) or not 0 <= tick < 2**32:
            raise ValueError("invalid uint32 simulation tick")
        if self.last is not None:
            delta = (tick - self.last) % 2**32
            if delta >= 2**31:
                raise ValueError("out-of-order simulation tick")
            if delta > 100:
                raise ValueError("simulation tick gap exceeds 100ms")
            self.elapsed += delta / 1000
        self.last = tick
        return self.elapsed


class CommandWatchdog:
    def __init__(self, timeout: float = 0.1):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("watchdog timeout must be finite and positive")
        self.timeout, self.last, self.latched = timeout, None, False

    def received(self, now: float) -> None:
        if not math.isfinite(now) or (self.last is not None and now < self.last):
            raise ValueError("invalid monotonic command time")
        if not self.latched:
            self.last = now

    def expired(self, now: float) -> bool:
        if self.last is not None and now - self.last > self.timeout:
            self.latched = True
        return self.latched


def _mapping(metadata: dict) -> list[int]:
    names = metadata["joint_names"]
    if len(names) != 29 or set(names) != set(SDK_MOTOR_INDEX):
        raise ValueError("invalid G1 29-DoF SDK motor mapping")
    return [SDK_MOTOR_INDEX[name] for name in names]


def encode_command(target: np.ndarray, metadata: dict, *, damping: bool = False):
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    target = finite_vector(target, 29, "command target")
    if np.any(target < np.array(metadata["joint_lower"])-1e-6) or np.any(target > np.array(metadata["joint_upper"])+1e-6):
        raise ValueError("command target exceeds joint range")
    msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr = 0
    msg.mode_machine = 0  # Local simulator mode; this is not a hardware mode claim.
    for i, sdk in enumerate(_mapping(metadata)):
        motor = msg.motor_cmd[sdk]
        motor.mode, motor.q, motor.dq, motor.tau = 1, float(target[i]), 0., 0.
        motor.kp = 0. if damping else float(metadata["kp"][i])
        motor.kd = max(2., float(metadata["kd"][i])) if damping else float(metadata["kd"][i])
    msg.crc = CRC().Crc(msg)
    return msg


@dataclass
class MotorCommand:
    target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


def decode_command(msg, metadata: dict) -> MotorCommand:
    from unitree_sdk2py.utils.crc import CRC

    if CRC().Crc(msg) != msg.crc:
        raise ValueError("LowCmd CRC mismatch")
    if msg.mode_pr != 0 or msg.mode_machine != 0:
        raise ValueError("local G1 simulator only accepts PR control in mode zero")
    motors = [msg.motor_cmd[index] for index in _mapping(metadata)]
    if any(m.mode != 1 or m.dq != 0 or m.tau != 0 for m in motors):
        raise ValueError("unsupported motor mode, desired velocity or feedforward torque")
    if any(m.mode != 0 for m in msg.motor_cmd[29:]):
        raise ValueError("unused motor slots must remain disabled")
    target = finite_vector([m.q for m in motors], 29, "LowCmd positions")
    kp = finite_vector([m.kp for m in motors], 29, "LowCmd kp")
    kd = finite_vector([m.kd for m in motors], 29, "LowCmd kd")
    if (np.any(target < np.array(metadata["joint_lower"])-1e-5)
            or np.any(target > np.array(metadata["joint_upper"])+1e-5)
            or np.any(kp < 0) or np.any(kp > np.array(metadata["kp"])+1e-4)
            or np.any(kd < 0) or np.any(kd > np.maximum(2, metadata["kd"])+1e-4)):
        raise ValueError("LowCmd exceeds position/gain limits")
    return MotorCommand(target, kp, kd)


def decode_state(msg, clock: TickClock) -> SensorState:
    from unitree_sdk2py.utils.crc import CRC

    if CRC().Crc(msg) != msg.crc:
        raise ValueError("LowState CRC mismatch")
    if msg.mode_pr != 0:
        raise ValueError("LowState requires G1 PR motor coordinates")
    q = finite_vector([m.q for m in msg.motor_state[:29]], 29, "LowState q")
    dq = finite_vector([m.dq for m in msg.motor_state[:29]], 29, "LowState dq")
    quat = finite_vector(msg.imu_state.quaternion, 4, "LowState pelvis quaternion")
    gyro = finite_vector(msg.imu_state.gyroscope, 3, "LowState pelvis gyro")
    quaternion_matrix(quat)
    return SensorState(q, dq, quat, gyro, clock.update(msg.tick))


class DDSClient:
    """Latest valid state, fresh-sample waits and actual SDK2 command publication."""

    def __init__(self, metadata: dict, domain: int):
        initialize_dds(domain)
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

        self.meta, self.clock = metadata, TickClock()
        self.condition = threading.Condition()
        self.latest, self.received_at, self.error = None, 0., None
        self.received_count, self.command_count = 0, 0
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._receive)

    def _receive(self, msg):
        with self.condition:
            try:
                self.latest = decode_state(msg, self.clock)
                self.received_at = time.monotonic()
                self.received_count += 1
            except (ValueError, RuntimeError) as exc:
                self.error = str(exc)
            self.condition.notify_all()

    def state_at(self, sample_time: float, *, timeout: float = 0.1) -> SensorState:
        deadline = time.monotonic()+timeout
        with self.condition:
            while True:
                if self.error:
                    raise ValueError(f"DDS state fault: {self.error}")
                now = time.monotonic()
                if self.latest is not None and self.latest.sample_time + 1e-6 >= sample_time:
                    if now-self.received_at > 0.1:
                        raise ValueError("stale DDS sensor state")
                    return self.latest
                remaining = deadline-now
                if remaining <= 0:
                    raise ValueError("DDS sensor timeout")
                self.condition.wait(remaining)

    def send(self, target: np.ndarray, *, damping: bool = False) -> None:
        msg = encode_command(target, self.meta, damping=damping)
        if not self.publisher.Write(msg):
            raise ValueError("DDS command write failed")
        self.command_count += 1

    def close(self):
        self.subscriber.Close()
        self.publisher.Close()
