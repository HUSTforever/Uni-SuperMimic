from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "frames,fps,expected", [(2039, 50.0, 2039), (2049, 50.0, 2049), (38, 30.0, 62)]
)
def test_reference_sample_count_respects_integer_endpoints(tmp_path, frames, fps, expected):
    from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
    from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import default_joint_pos_list

    raw = np.zeros((frames, 36))
    raw[:, 0] = np.linspace(0, 0.2, frames)
    raw[:, 2] = 0.76
    raw[:, 6] = 1
    raw[:, 7:] = default_joint_pos_list()
    if fps == 50:
        # Identity conversion must not smear a valid, sharp shoulder movement.
        raw[1043, 22] = 0.5
    path = tmp_path / "motion.csv"
    np.savetxt(path, raw, delimiter=",")
    scene = (
        Path(__file__).resolve().parents[2]
        / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"
    )
    motion = MotionLib([str(path)], G1Kinematics(str(scene), "cpu"), "cpu", src_fps=fps)
    assert motion.num_frames == expected
    if fps == 50:
        assert float(motion.root_p[-1, 0]) == pytest.approx(0.2, abs=2e-6)
        np.testing.assert_array_equal(motion.q.numpy(), raw[:, 7:].astype(np.float32))
