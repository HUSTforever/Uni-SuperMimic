"""Tests for the GVHMR -> GMR retarget script's Y-up ground correction."""

from __future__ import annotations

import numpy as np
import pytest
from demo.retarget_gvhmr import ground_yup

FEET = [7, 8, 10, 11]


def _synthetic_motion() -> tuple[np.ndarray, np.ndarray]:
    frames = 4
    joints = np.zeros((frames, 17, 3), dtype=np.float64)
    joints[:, :, 0] = np.linspace(-0.5, 0.5, 17)
    joints[:, :, 2] = 2.0 + np.linspace(-0.2, 0.2, 17)
    joints[:, :, 1] = np.linspace(0.3, 1.6, 17)
    # Feet sit slightly above the floor at ~0.1 m, with a different minimum
    # per frame so the correction has to be frame-wise.
    foot_y = np.array(
        [
            [0.08, 0.10, 0.13, 0.09],
            [0.11, 0.09, 0.12, 0.10],
            [0.07, 0.12, 0.14, 0.08],
            [0.10, 0.13, 0.15, 0.11],
        ],
        dtype=np.float64,
    )
    joints[:, FEET, 1] = foot_y
    translations = np.array(
        [
            [0.0, 0.90, 2.0],
            [0.1, 0.92, 2.1],
            [0.2, 0.94, 2.2],
            [0.3, 0.96, 2.3],
        ],
        dtype=np.float64,
    )
    return joints, translations


def test_ground_yup_zeroes_lowest_foot_and_preserves_x_z() -> None:
    joints, translations = _synthetic_motion()
    original_joints = joints.copy()
    original_translations = translations.copy()
    expected_offset = joints[:, FEET, 1].min(axis=1)

    shifted_joints, shifted_translations, offset = ground_yup(joints, translations)

    np.testing.assert_allclose(offset, expected_offset)
    np.testing.assert_allclose(shifted_joints[:, FEET, 1].min(axis=1), 0.0, atol=1e-12)
    np.testing.assert_array_equal(shifted_joints[:, :, 0], original_joints[:, :, 0])
    np.testing.assert_array_equal(shifted_joints[:, :, 2], original_joints[:, :, 2])
    np.testing.assert_allclose(
        shifted_translations[:, 1], original_translations[:, 1] - expected_offset
    )
    np.testing.assert_array_equal(shifted_translations[:, 0], original_translations[:, 0])
    np.testing.assert_array_equal(shifted_translations[:, 2], original_translations[:, 2])
    # The caller's arrays must never be modified in place.
    np.testing.assert_array_equal(joints, original_joints)
    np.testing.assert_array_equal(translations, original_translations)


def test_ground_yup_keeps_float32_dtype() -> None:
    joints, translations = _synthetic_motion()
    shifted_joints, shifted_translations, offset = ground_yup(
        joints.astype(np.float32), translations.astype(np.float32)
    )

    assert shifted_joints.dtype == np.float32
    assert shifted_translations.dtype == np.float32
    assert offset.dtype == np.float32


@pytest.mark.parametrize(
    ("joints_shape", "translations_shape", "value"),
    [
        ((2, 11, 3), (2, 3), 0.0),
        ((2, 17, 2), (2, 3), 0.0),
        ((2, 17, 3), (2, 2), 0.0),
        ((2, 17, 3), (3, 3), 0.0),
        ((17, 3), (1, 3), 0.0),
        ((2, 17, 3), (2, 3), np.nan),
        ((2, 17, 3), (2, 3), np.inf),
    ],
)
def test_ground_yup_rejects_invalid_input(
    joints_shape: tuple[int, ...],
    translations_shape: tuple[int, ...],
    value: float,
) -> None:
    joints = np.full(joints_shape, value, dtype=np.float64)
    translations = np.zeros(translations_shape, dtype=np.float64)
    if np.isfinite(value):
        translations.flat[0] = value

    with pytest.raises(ValueError):
        ground_yup(joints, translations)


class _FakeTensor:
    def __init__(self, array: np.ndarray, dtype: object = None, device: object = "cpu") -> None:
        self.array = np.asarray(array)
        self.dtype = dtype if dtype is not None else self.array.dtype
        self.device = device

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array


@pytest.mark.parametrize("filename", ["qpos.npz", "qpos.NPZ"])
def test_main_grounds_y_up_tensor_and_writes_sidecar(tmp_path: Path, monkeypatch, filename) -> None:
    import json
    import sys
    import types

    from demo import retarget_gvhmr

    joints, translations = _synthetic_motion()
    joints = joints.astype(np.float32)
    original_translations = translations.copy()
    fake_tensor = _FakeTensor(joints, dtype=np.float32, device="cuda:0")
    smplx_data = {
        "trans": translations.copy(),
        "pose_body": np.ones((4, 63), dtype=np.float32),
        "mocap_frame_rate": np.array(30.0),
    }
    calls = {}

    fake_torch = types.ModuleType("torch")
    fake_torch.as_tensor = lambda array, dtype=None, device=None: _FakeTensor(
        array, dtype=dtype, device=device
    )
    gmr_module = types.ModuleType("general_motion_retargeting")

    class FakeGMR:
        def __init__(self, **kwargs) -> None:
            calls["gmr"] = kwargs

        def retarget(self, frame):
            return np.full(36, frame, dtype=np.float64)

    gmr_module.GeneralMotionRetargeting = FakeGMR
    smpl_module = types.ModuleType("general_motion_retargeting.utils.smpl")

    def fake_load(pred, body_models):
        calls["load"] = (pred, body_models)
        output = types.SimpleNamespace(joints=fake_tensor)
        calls["output"] = output
        return smplx_data, object(), output, 1.7

    def fake_frames(data, body_model, output, tgt_fps):
        calls["frames"] = (data["mocap_frame_rate"].item(), tgt_fps)
        return [0.0, 0.0, 0.0, 0.0], tgt_fps

    smpl_module.load_gvhmr_pred_file = fake_load
    smpl_module.get_gvhmr_data_offline_fast = fake_frames
    utils_module = types.ModuleType("general_motion_retargeting.utils")
    utils_module.smpl = smpl_module
    gmr_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "general_motion_retargeting", gmr_module)
    monkeypatch.setitem(sys.modules, "general_motion_retargeting.utils", utils_module)
    monkeypatch.setitem(sys.modules, "general_motion_retargeting.utils.smpl", smpl_module)

    pred = tmp_path / "pred.pt"
    pred.write_bytes(b"prediction")
    gmr_root = tmp_path / "gmr"
    (gmr_root / "assets" / "body_models").mkdir(parents=True)
    output = tmp_path / "out" / filename

    retarget_gvhmr.main(
        [
            "--gvhmr_pred_file",
            str(pred),
            "--output",
            str(output),
            "--gmr-root",
            str(gmr_root),
            "--src-fps",
            "30",
            "--ground-human",
        ]
    )

    grounded_tensor = calls["output"].joints
    grounded = grounded_tensor.array
    expected_offset = joints[:, FEET, 1].min(axis=1)
    np.testing.assert_allclose(grounded[:, FEET, 1].min(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(
        smplx_data["trans"][:, 1],
        original_translations[:, 1] - expected_offset,
    )
    assert grounded_tensor.dtype == np.float32
    assert grounded_tensor.device == "cuda:0"
    assert calls["load"][1] == gmr_root / "assets" / "body_models"
    assert calls["frames"] == (30.0, 30.0)
    assert calls["gmr"] == {
        "actual_human_height": 1.7,
        "src_human": "smplx",
        "tgt_robot": "unitree_g1",
        "verbose": False,
    }
    with np.load(output) as saved:
        assert saved["qpos"].shape == (4, 36)
        assert saved["fps"].tolist() == [30.0]
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["ground_axis"] == "y"
    assert metadata["src_fps"] == 30.0
    assert metadata["human_height"] == 1.7
    assert metadata["source_sha256"]
    assert metadata["root_y_range"] == [
        float(original_translations[:, 1].min()),
        float(original_translations[:, 1].max()),
    ]


@pytest.mark.parametrize("suffix", [".npz", ".json"])
def test_retarget_preserves_existing_outputs_before_loading_dependencies(tmp_path, suffix):
    from demo.retarget_gvhmr import main

    output = tmp_path / "qpos.npz"
    protected = output.with_suffix(suffix)
    protected.write_bytes(b"preserve existing result")
    with pytest.raises(FileExistsError):
        main(
            [
                "--gvhmr_pred_file",
                str(tmp_path / "pred.pt"),
                "--output",
                str(output),
                "--gmr-root",
                str(tmp_path / "gmr"),
            ]
        )
    assert protected.read_bytes() == b"preserve existing result"


def test_explicit_flight_mode_preserves_vertical_takeoff():
    joints, translations = _synthetic_motion()
    joints[:, FEET, 1] = 0
    translations[:, 1] = 0.9
    joints[1:3, :, 1] += 0.25
    translations[1:3, 1] += 0.25
    shifted, trans, offset = ground_yup(joints, translations, preserve_flight=True)
    np.testing.assert_allclose(shifted[:, FEET, 1].min(1), [0, 0.25, 0.25, 0])
    np.testing.assert_allclose(trans[:, 1], [0.9, 1.15, 1.15, 0.9])
    np.testing.assert_allclose(offset, 0)
