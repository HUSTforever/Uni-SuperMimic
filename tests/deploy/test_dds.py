import numpy as np
import pytest


@pytest.mark.parametrize("interface,domain", [("eth0",17),(None,17),("lo",0),("lo",-1),("lo",True)])
def test_sim_endpoint_rejects_hardware_routes_before_sdk_initialization(interface, domain):
    from unisupermimic.deploy.dds import validate_sim_endpoint

    with pytest.raises(ValueError):
        validate_sim_endpoint(interface, domain)


def test_tick_wrap_and_out_of_order_samples():
    from unisupermimic.deploy.dds import TickClock

    clock = TickClock()
    assert clock.update(2**32-3) == 0
    assert clock.update(2) == pytest.approx(0.005)
    with pytest.raises(ValueError, match="order|tick"):
        clock.update(1)


def test_lowcmd_mapping_crc_and_damping(bundle):
    pytest.importorskip("unitree_sdk2py")
    from unitree_sdk2py.utils.crc import CRC

    from unisupermimic.deploy.dds import decode_command, encode_command

    target = np.array(bundle.metadata["default_joint_positions"])
    msg = encode_command(target, bundle.metadata)
    assert CRC().Crc(msg) == msg.crc
    assert msg.mode_pr == 0
    assert msg.motor_cmd[15].q == pytest.approx(target[15])
    assert msg.motor_cmd[22].q == pytest.approx(target[22])
    assert msg.motor_cmd[29].mode == 0
    command = decode_command(msg, bundle.metadata)
    np.testing.assert_allclose(command.target, target)
    damped = encode_command(target, bundle.metadata, damping=True)
    stop = decode_command(damped, bundle.metadata)
    assert np.all(stop.kp == 0)
    assert np.all(stop.kd > 0)
    msg.motor_cmd[0].q += 1
    with pytest.raises(ValueError, match="CRC"):
        decode_command(msg, bundle.metadata)


def test_lowstate_decode_uses_pelvis_and_validates_crc():
    pytest.importorskip("unitree_sdk2py")
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.utils.crc import CRC

    from unisupermimic.deploy.dds import TickClock, decode_state

    msg = unitree_hg_msg_dds__LowState_()
    msg.imu_state.quaternion = [1.,0.,0.,0.]
    msg.imu_state.gyroscope = [1.,2.,3.]
    msg.motor_state[14].q = 0.6
    msg.crc = CRC().Crc(msg)
    state = decode_state(msg, TickClock())
    np.testing.assert_allclose(state.quaternion, [1,0,0,0])
    np.testing.assert_allclose(state.gyro, [1,2,3])
    assert state.q[14] == pytest.approx(0.6)
    msg.tick += 5
    with pytest.raises(ValueError, match="CRC"):
        decode_state(msg, TickClock())


def test_watchdog_latches_when_controller_stops_and_never_rearms():
    from unisupermimic.deploy.dds import CommandWatchdog

    watchdog = CommandWatchdog(timeout=0.1)
    watchdog.received(1.0)
    assert not watchdog.expired(1.05)
    assert watchdog.expired(1.101)
    watchdog.received(1.11)
    assert watchdog.expired(1.12)


@pytest.mark.slow
def test_actual_publisher_process_exit_triggers_simulator_watchdog(bundle, tmp_path):
    pytest.importorskip("unitree_sdk2py")
    from pathlib import Path

    from unisupermimic.deploy.dds_sim import run_watchdog_probe

    scene = Path(__file__).resolve().parents[2] / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"
    report = run_watchdog_probe(bundle, scene, tmp_path / "probe", domain=44)
    assert report["publisher_exit_code"] == 0
    assert report["watchdog_triggered"] is True
    assert report["damping_latched"] is True
    assert report["max_torque_limit_ratio"] <= 1
