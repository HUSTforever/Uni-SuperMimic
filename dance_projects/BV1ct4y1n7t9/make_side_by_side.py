"""Compose source and genuine policy at the declared 0.8x source tempo."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", type=Path, default=PROJECT / "artifacts/final_static/policy_only.mp4"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "artifacts/final_static/source_vs_policy.mp4"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = imageio.get_reader(str(PROJECT / "clips/ji_ni_tai_mei_fullbody.mp4"))
    policy = imageio.get_reader(str(args.policy))
    source_count = source.count_frames()
    policy_count = policy.count_frames()
    policy_fps = policy.get_meta_data()["fps"]
    source_fps = source.get_meta_data()["fps"]
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    silent = args.output.with_name(args.output.stem + "_silent.mp4")
    try:
        with imageio.get_writer(
            str(silent), fps=policy_fps, codec="libx264", quality=8, macro_block_size=None
        ) as writer:
            for index in range(policy_count):
                src_index = min(round(index * source_fps / policy_fps / 1.25), source_count - 1)
                left = Image.fromarray(source.get_data(src_index)).resize((960, 540))
                right = Image.fromarray(policy.get_data(index)).resize((960, 540))
                canvas = Image.new("RGB", (1920, 580), "white")
                canvas.paste(left, (0, 40))
                canvas.paste(right, (960, 40))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (12, 8),
                    f"SOURCE | original {13.4 + src_index / source_fps:.2f}s | tempo 0.8x",
                    fill="black",
                    font=font,
                )
                draw.text(
                    (972, 8),
                    f"G1 29DoF | physical policy | {index / policy_fps:.2f}s | 50 Hz",
                    fill="black",
                    font=font,
                )
                writer.append_data(np.asarray(canvas))
    finally:
        source.close()
        policy.close()
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(silent),
        "-ss",
        "13.4",
        "-i",
        str(PROJECT / "inputs/source.mp4"),
        "-filter_complex",
        "[1:a]atrim=duration=16.8,asetpts=PTS-STARTPTS,atempo=0.8[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-t",
        str(policy_count / policy_fps),
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    subprocess.run(cmd, check=True)
    silent.unlink()
    print(args.output)


if __name__ == "__main__":
    main()
