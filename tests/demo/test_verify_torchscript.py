"""Unit tests for ``demo/verify_torchscript.py``."""

from __future__ import annotations

import numpy as np
import pytest
from demo import verify_torchscript

ROLLOUT_DIM = 7 + 29


def _rollout(frames: int = 4) -> np.ndarray:
    return np.zeros((frames, ROLLOUT_DIM), dtype=np.float64)


def test_validate_rollouts_returns_max_abs_diff() -> None:
    eager = _rollout()
    scripted = eager.copy()
    scripted[1, 2] = 5e-4
    scripted[3, 4] = -4e-4

    diff = verify_torchscript.validate_rollouts(eager, scripted)

    assert diff == pytest.approx(5e-4, rel=0.0, abs=1e-12)


def test_validate_rollouts_allows_diff_equal_to_atol() -> None:
    eager = _rollout()
    scripted = eager.copy()
    scripted[0, 0] = 1e-3

    assert verify_torchscript.validate_rollouts(eager, scripted) == pytest.approx(1e-3)


def test_validate_rollouts_raises_on_diff_above_atol() -> None:
    eager = _rollout()
    scripted = eager.copy()
    scripted[0, 0] = 2e-3

    with pytest.raises(ValueError, match="mismatch"):
        verify_torchscript.validate_rollouts(eager, scripted)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bad_side", ["eager", "scripted"])
def test_validate_rollouts_rejects_non_finite_values(bad_value: float, bad_side: str) -> None:
    eager = _rollout()
    scripted = eager.copy()
    target = eager if bad_side == "eager" else scripted
    target[1, 2] = bad_value

    with pytest.raises(ValueError, match="finite"):
        verify_torchscript.validate_rollouts(eager, scripted)


@pytest.mark.parametrize(
    ("eager", "scripted", "message"),
    [
        (_rollout(4), _rollout(3), "shape"),
        (_rollout(4), np.zeros((4, 35)), "shape"),
        (_rollout(4)[:, :35], _rollout(4)[:, :35], "shape"),
        (np.zeros((ROLLOUT_DIM,)), np.zeros((ROLLOUT_DIM,)), "shape"),
        (_rollout(1), _rollout(1), "at least 2"),
        (_rollout(0), _rollout(0), "at least 2"),
    ],
)
def test_validate_rollouts_rejects_bad_shapes(
    eager: np.ndarray, scripted: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        verify_torchscript.validate_rollouts(eager, scripted)


@pytest.mark.parametrize("atol", [0.0, -1e-3, np.nan, np.inf, -np.inf])
def test_validate_rollouts_rejects_invalid_atol(atol: float) -> None:
    with pytest.raises(ValueError, match="atol"):
        verify_torchscript.validate_rollouts(_rollout(), _rollout(), atol=atol)


def test_equal_short_rollouts_do_not_establish_requested_duration():
    from demo.verify_torchscript import validate_rollouts

    with pytest.raises(ValueError, match="expected"):
        validate_rollouts(np.zeros((3, 36)), np.zeros((3, 36)), expected_frames=100)
