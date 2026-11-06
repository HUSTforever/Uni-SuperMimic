"""Report camera-relative rotation diagnostics from two SMPL root estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def rotation(vectors):
    zero = torch.zeros_like(vectors[:, 0])
    x, y, z = vectors.T
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).reshape(-1, 3, 3)
    return torch.linalg.matrix_exp(skew)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    args = parser.parse_args()
    data = torch.load(args.prediction, map_location="cpu", weights_only=True)
    world = rotation(data["smpl_params_global"]["global_orient"])
    camera = rotation(data["smpl_params_incam"]["global_orient"])
    world_to_camera = camera @ world.transpose(-1, -2)
    relative = world_to_camera @ world_to_camera[:1].transpose(-1, -2)
    cosine = ((torch.diagonal(relative, dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angles = torch.rad2deg(torch.arccos(cosine))
    print(
        json.dumps(
            {
                "prediction": str(args.prediction),
                "sha256": hashlib.sha256(args.prediction.read_bytes()).hexdigest(),
                "frames": len(angles),
                "relative_rotation_max_degrees": angles.max().item(),
                "relative_rotation_median_degrees": angles.median().item(),
                "interpretation": "Diagnostic only. Compare estimated camera motion against the actual background and source/model heading; a stable policy score does not verify correct motion recovery.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
