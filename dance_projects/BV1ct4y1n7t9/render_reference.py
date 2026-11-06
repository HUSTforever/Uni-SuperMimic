"""Render only the kinematic reference; this is explicitly NOT a policy rollout."""

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import torch

from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.g1.supermimic.render import (
    label_font,
    reference_qpos,
)

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
SCENE = ROOT / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"


def main():
    import numpy as np
    from PIL import Image, ImageDraw

    active = json.loads((PROJECT / "source.json").read_text())
    motion = MotionLib(
        [str(PROJECT / active.get("active_motion", "motion/ji_ni_tai_mei.csv"))],
        G1Kinematics(str(SCENE), "cpu"),
        "cpu",
    )
    rows = reference_qpos(motion, torch.arange(motion.num_frames)).numpy()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 540
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.1, 135, -12
    renderer = mujoco.Renderer(model, height=540, width=960)
    target = PROJECT / "artifacts/reference_kinematic_static.mp4"
    try:
        with imageio.get_writer(
            str(target), fps=50, codec="libx264", quality=8, macro_block_size=None
        ) as writer:
            for index, row in enumerate(rows):
                data.qpos[:] = row
                mujoco.mj_kinematics(model, data)
                camera.lookat[:] = [row[0], row[1], 0.75]
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 36), fill="white")
                draw.text(
                    (12, 5),
                    f"G1 29DoF REFERENCE (kinematic, not policy) | t={index / 50:.2f}s",
                    fill="black",
                    font=label_font(),
                )
                writer.append_data(np.asarray(frame))
    finally:
        renderer.close()
    print("reference frames", len(rows), "->", target)


if __name__ == "__main__":
    main()
