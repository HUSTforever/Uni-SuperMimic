"""Render the raw DDS 24-28 s window and build an unsmoothed side-by-side comparison."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
WORKSPACE = PROJECT.parents[1]
SCENE = WORKSPACE / "deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
BASELINE_NPZ = (
    WORKSPACE
    / "finetune_projects/g1_continuous_20261029"
    / "evaluation/dds_ground008_200_initial/trajectory.npz"
)
EXPERIMENTAL_NPZ = PROJECT / "evaluation/dds_run002_250_probe/trajectory.npz"
ARTIFACTS = PROJECT / "artifacts"
PREFIX = "run002_250_dds_24_28"
FFMPEG = Path("/home/ytkj/cx/ros2/ros2-jazzy/bin/ffmpeg")
FFPROBE = FFMPEG.with_name("ffprobe")
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

SLICE_START = 1200
SLICE_STOP = 1400
FRAME_COUNT = SLICE_STOP - SLICE_START
FPS = 50
RENDER_WIDTH, RENDER_HEIGHT = 960, 540
COMPARE_WIDTH, COMPARE_HEIGHT = 1920, 540
LEFT_LABEL = "Baseline"
RIGHT_LABEL = "Experimental kd2 ft250"
EXPECTED_NPZ_FRAMES = 1799


def sha256(path: Path) -> str:
    """Hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_window(path: Path) -> np.ndarray:
    """Return the raw qpos rows [1200:1400]; no resampling or smoothing is applied."""
    with np.load(path) as archive:
        if "qpos" not in archive:
            raise KeyError(f"{path} has no 'qpos' key: {sorted(archive.keys())}")
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"{path}: qpos must be 2-D, got {qpos.shape}")
    if qpos.shape[0] != EXPECTED_NPZ_FRAMES:
        raise ValueError(f"{path}: expected {EXPECTED_NPZ_FRAMES} frames, got {qpos.shape[0]}")
    window = qpos[SLICE_START:SLICE_STOP]
    if window.shape[0] != FRAME_COUNT:
        raise ValueError(f"{path}: window has {window.shape[0]} frames, expected {FRAME_COUNT}")
    if not np.isfinite(window).all():
        raise ValueError(f"{path}: window contains non-finite qpos values")
    return window


def relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE).as_posix()


def parse_rate(value: str) -> float:
    numerator, _, denominator = value.partition("/")
    return float(numerator) / float(denominator or 1)


def probe(path: Path) -> dict:
    """Measure the encoded stream with ffprobe, decoding every frame for nb_read_frames."""
    command = [
        str(FFPROBE),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": parse_rate(stream["avg_frame_rate"]),
        "frame_count": int(stream["nb_read_frames"]),
        "duration_s": round(float(payload["format"]["duration"]), 6),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_video(name: str, measured: dict, width: int, height: int) -> None:
    require(measured["width"] == width, f"{name}: width {measured['width']} != {width}")
    require(measured["height"] == height, f"{name}: height {measured['height']} != {height}")
    require(measured["frame_count"] == FRAME_COUNT, f"{name}: frames {measured['frame_count']}")
    require(measured["fps"] == FPS, f"{name}: fps {measured['fps']} != {FPS}")
    require(
        abs(measured["duration_s"] - FRAME_COUNT / FPS) <= 0.02,
        f"{name}: duration {measured['duration_s']} is not {FRAME_COUNT / FPS:.2f}s",
    )


def drawtext(label: str, x: int) -> str:
    return (
        f"drawtext=fontfile={FONT}:text='{label}':x={x}:y=20:"
        "fontsize=30:fontcolor=white:borderw=2:bordercolor=black"
    )


def compose(baseline_video: Path, experimental_video: Path, output: Path) -> None:
    """Stack both 960x540 renders and label them with ffmpeg drawtext."""
    filtergraph = (
        f"[0:v][1:v]hstack=inputs=2,"
        f"{drawtext(LEFT_LABEL, 24)},"
        f"{drawtext(RIGHT_LABEL, RENDER_WIDTH + 24)}"
    )
    command = [
        str(FFMPEG),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(baseline_video),
        "-i",
        str(experimental_video),
        "-filter_complex",
        filtergraph,
        "-frames:v",
        str(FRAME_COUNT),
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MUJOCO_GL"] = "egl"

    from unisupermimic.deploy.run_sim2sim import render_trajectory

    for path in (SCENE, BASELINE_NPZ, EXPERIMENTAL_NPZ, FONT):
        require(path.is_file(), f"missing required input: {path}")
    require(FFMPEG.is_file(), f"missing ffmpeg: {FFMPEG}")

    baseline_sha = sha256(BASELINE_NPZ)
    experimental_sha = sha256(EXPERIMENTAL_NPZ)
    baseline_qpos = load_window(BASELINE_NPZ)
    experimental_qpos = load_window(EXPERIMENTAL_NPZ)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    baseline_video = ARTIFACTS / f"{PREFIX}_baseline.mp4"
    experimental_video = ARTIFACTS / f"{PREFIX}_experimental.mp4"
    compare_video = ARTIFACTS / f"{PREFIX}_compare.mp4"
    recipe_path = ARTIFACTS / f"{PREFIX}_recipe.json"

    # render_trajectory replays the stored physics states verbatim at 50 fps.
    render_trajectory(SCENE, baseline_qpos, baseline_video)
    render_trajectory(SCENE, experimental_qpos, experimental_video)
    baseline_probe = probe(baseline_video)
    experimental_probe = probe(experimental_video)
    check_video("baseline render", baseline_probe, RENDER_WIDTH, RENDER_HEIGHT)
    check_video("experimental render", experimental_probe, RENDER_WIDTH, RENDER_HEIGHT)

    compose(baseline_video, experimental_video, compare_video)
    compare_probe = probe(compare_video)
    check_video("compare", compare_probe, COMPARE_WIDTH, COMPARE_HEIGHT)

    recipe = {
        "description": "Raw DDS physics qpos replay for seconds 24-28, baseline vs experimental",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "qpos_key": "qpos",
        "slice": [SLICE_START, SLICE_STOP],
        "slice_semantics": "numpy qpos[1200:1400], stop exclusive, no resampling",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "duration_s": FRAME_COUNT / FPS,
        "trajectory_filtering": "none",
        "sources": {
            "baseline": {
                "npz": relative(BASELINE_NPZ),
                "sha256": baseline_sha,
                "npz_frame_count": EXPECTED_NPZ_FRAMES,
                "slice": [SLICE_START, SLICE_STOP],
                "frame_count": FRAME_COUNT,
                "render": relative(baseline_video),
            },
            "experimental": {
                "npz": relative(EXPERIMENTAL_NPZ),
                "sha256": experimental_sha,
                "npz_frame_count": EXPECTED_NPZ_FRAMES,
                "slice": [SLICE_START, SLICE_STOP],
                "frame_count": FRAME_COUNT,
                "render": relative(experimental_video),
            },
        },
        "scene": {"path": relative(SCENE), "sha256": sha256(SCENE)},
        "comparison": {
            "path": relative(compare_video),
            "width": COMPARE_WIDTH,
            "height": COMPARE_HEIGHT,
            "fps": FPS,
            "frame_count": FRAME_COUNT,
            "duration_s": FRAME_COUNT / FPS,
            "labels": {"left": LEFT_LABEL, "right": RIGHT_LABEL},
        },
        "ffprobe": {
            "baseline": baseline_probe,
            "experimental": experimental_probe,
            "compare": compare_probe,
        },
        "experimental_not_accepted": True,
        "hardware_ready": False,
    }
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")

    print(f"baseline      {relative(baseline_video)}  {baseline_probe}")
    print(f"experimental  {relative(experimental_video)}  {experimental_probe}")
    print(f"compare       {relative(compare_video)}  {compare_probe}")
    print(f"recipe        {relative(recipe_path)}")


if __name__ == "__main__":
    main()
