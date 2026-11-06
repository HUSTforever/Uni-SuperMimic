import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def packer():
    path = (
        Path(__file__).resolve().parents[2]
        / "finetune_projects/g1_continuity_20261029/pack_motion_bundle.py"
    )
    spec = importlib.util.spec_from_file_location("tested_projection_packer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared(bundle, tmp_path, *, over_limit=False):
    ref = bundle.reference
    poses = np.c_[ref["root_p"], ref["root_quat"], ref["q"]]
    if over_limit:
        poses[:, 7 + 14] = bundle.metadata["joint_upper"][14] + 0.01
    array = tmp_path / "prepared.npz"
    np.savez(array, qpos=poses)
    metadata = tmp_path / "prepared.json"
    metadata.write_text(
        json.dumps(
            dict(
                preparation_eligible=True,
                prepared_array_sha256=hashlib.sha256(array.read_bytes()).hexdigest(),
            )
        )
    )
    return array, metadata, poses


def test_prepared_array_must_match_the_checked_provenance(adaptation_bundle, tmp_path):
    array, metadata, poses = prepared(adaptation_bundle, tmp_path)
    poses[:, 22:] += 0.05
    np.savez(array, qpos=poses)
    with pytest.raises(ValueError, match="hash|digest|fingerprint"):
        packer().pack(adaptation_bundle, array, metadata, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_matching_hash_cannot_bypass_joint_limits(adaptation_bundle, tmp_path):
    array, metadata, _ = prepared(adaptation_bundle, tmp_path, over_limit=True)
    with pytest.raises(ValueError, match="limit"):
        packer().pack(adaptation_bundle, array, metadata, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_missing_preparation_hash_is_rejected(adaptation_bundle, tmp_path):
    array, metadata, _ = prepared(adaptation_bundle, tmp_path)
    metadata.write_text(json.dumps(dict(preparation_eligible=True)))
    with pytest.raises(ValueError, match="hash|digest|fingerprint"):
        packer().pack(adaptation_bundle, array, metadata, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
