import copy

import numpy as np


def setup_case(bundle):
    from unisupermimic.deploy.controller import SensorState

    meta = copy.deepcopy(bundle.metadata)
    meta["sequence"] = dict(fast_finish=dict(exit_seconds=1.0, hold_seconds=12.0))
    stand = np.array(meta["default_joint_positions"]) + np.array(meta["context_offset"])
    state = SensorState(stand.copy(), np.zeros(29), np.array([1.0, 0, 0, 0]), np.zeros(3), 1.0)
    return meta, state, stand + 0.03


def test_completed_finish_keeps_last_active_target(bundle):
    from unisupermimic.deploy.finish import completion_command

    meta, state, last = setup_case(bundle)
    target, damping, assessment = completion_command(meta, state, last, completed=True)
    assert not damping and assessment["mode"] == "standing_hold"
    np.testing.assert_allclose(target, last)
    assert not np.allclose(target, state.q)


def test_fault_and_legacy_finish_still_use_damping(bundle):
    from unisupermimic.deploy.finish import completion_command

    meta, state, last = setup_case(bundle)
    assert completion_command(meta, state, last, completed=False)[1]
    meta.pop("sequence")
    assert completion_command(meta, state, last, completed=True)[1]


def test_invalid_or_unsettled_sensor_does_not_claim_standing(bundle):
    from unisupermimic.deploy.finish import completion_command

    meta, state, last = setup_case(bundle)
    state.dq[:] = 1.0
    assert completion_command(meta, state, last, completed=True)[1]
    state.q[0] = float("nan")
    target, damping, assessment = completion_command(meta, state, last, completed=True)
    assert damping and np.isfinite(target).all() and assessment["mode"] == "damping"
