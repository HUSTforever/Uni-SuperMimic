"""Create a simulation-only training bundle with doubled ankle damping (kd)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "finetune_projects/g1_continuous_20261029/data/sequence_v2"
OUTPUT_DIR = PROJECT_DIR / "data/damping_kd200"

BUNDLE_FILES = ("manifest.json", "policy.ts", "reference.npz", "sequence.csv", "sequence.json")
ANKLE_KD_FACTOR = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_damping_profile() -> dict[str, Any]:
    if OUTPUT_DIR.exists() or OUTPUT_DIR.is_symlink():
        raise FileExistsError(f"output directory already exists: {OUTPUT_DIR}")

    source_manifest_path = SOURCE_DIR / "manifest.json"
    source_manifest_sha256 = _sha256(source_manifest_path)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["metadata"]

    joint_names = metadata["joint_names"]
    ankle_indices = [index for index, name in enumerate(joint_names) if "ankle" in name]
    if len(ankle_indices) != 4:
        raise ValueError(f"expected four ankle joints, found {len(ankle_indices)}")

    old_kd = list(metadata["kd"])
    new_kd = list(old_kd)
    for index in ankle_indices:
        new_kd[index] = old_kd[index] * ANKLE_KD_FACTOR
    metadata["kd"] = new_kd
    metadata["servo_profile"] = {
        "name": "ankle_kd200",
        "source_manifest_sha256": source_manifest_sha256,
        "ankle_kd_factor": ANKLE_KD_FACTOR,
        "simulation_only": True,
    }
    metadata["accepted_for_delivery"] = False
    metadata["hardware_ready"] = False
    metadata["candidate_status"] = "unvalidated"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    for name in BUNDLE_FILES:
        shutil.copyfile(SOURCE_DIR / name, OUTPUT_DIR / name)
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "source_dir": str(SOURCE_DIR.relative_to(REPO_ROOT)),
        "output_dir": str(OUTPUT_DIR.relative_to(REPO_ROOT)),
        "old_kd": old_kd,
        "new_kd": new_kd,
    }


def main() -> None:
    print(json.dumps(prepare_damping_profile(), indent=2))


if __name__ == "__main__":
    main()
