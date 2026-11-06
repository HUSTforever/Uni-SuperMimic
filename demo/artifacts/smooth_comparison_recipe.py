"""Rebuild the presentation comparison; physics metrics use the unretimed NPZs."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from unisupermimic.tasks.motion_tracking.g1.supermimic.render import label_font

ROOT = Path(__file__).resolve().parent
before = imageio.get_reader(str(ROOT / "smooth_baseline/policy_vs_reference.mp4"))
after = imageio.get_reader(str(ROOT / "smooth_final/policy_vs_reference.mp4"))
n_before, n_after = before.count_frames(), after.count_frames()
try:
    with imageio.get_writer(
        str(ROOT / "smooth_before_after.mp4"),
        fps=50,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for index in range(n_after):
            phase = index / (n_after - 1)
            first = Image.fromarray(before.get_data(round(phase * (n_before - 1)))).convert("RGB")
            second = Image.fromarray(after.get_data(index)).convert("RGB")
            for frame, text in [
                (first, f"BEFORE | native 5.06s | phase {phase:.0%}"),
                (second, f"SMOOTH | native 6.32s | phase {phase:.0%}"),
            ]:
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, frame.width, 36), fill=(250, 250, 250))
                draw.text((12, 5), text, fill=(15, 15, 15), font=label_font())
            canvas = Image.new(
                "RGB", (first.width + second.width, first.height + 32), color=(30, 30, 30)
            )
            canvas.paste(first, (0, 32))
            canvas.paste(second, (first.width, 32))
            ImageDraw.Draw(canvas).text(
                (12, 4),
                "Phase-aligned comparison; each panel labels its original simulated duration. Metrics use native 50 Hz trajectories.",
                fill="white",
                font=label_font(),
            )
            writer.append_data(np.asarray(canvas))
finally:
    before.close()
    after.close()
