"""Tests for CFR video normalization before GVHMR."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from demo.prepare_video import prepare_video


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(path: Path) -> Path:
    frames = [np.full((64, 64, 3), 30 + i * 10, dtype=np.uint8) for i in range(12)]
    imageio.mimwrite(path, frames, fps=12, codec="libx264", quality=5)
    return path


def _decode(path: Path) -> tuple[dict, int]:
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        frames = sum(1 for _ in reader)
    finally:
        reader.close()
    return metadata, frames


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return _write_source(tmp_path / "source.mp4")


def test_prepare_video_trims_resamples_and_records_hashes(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "prepared.mp4"

    result = prepare_video(source, output, start=0.25, end=0.75, fps=30)

    metadata, frames = _decode(output)
    assert metadata["fps"] == pytest.approx(30.0)
    assert frames == 15

    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["source"] == str(source.resolve())
    assert sidecar["source_sha256"] == _sha256(source)
    assert sidecar["output_sha256"] == _sha256(output)
    assert sidecar["start"] == 0.25
    assert sidecar["end"] == 0.75
    assert sidecar["fps"] == 30.0
    assert result == sidecar


@pytest.mark.parametrize(
    "options",
    [
        {"start": -0.1},
        {"start": math.nan},
        {"end": math.inf},
        {"start": 0.5, "end": 0.5},
        {"start": 0.75, "end": 0.25},
        {"fps": 0},
        {"fps": math.inf},
    ],
)
def test_invalid_ranges_are_rejected_without_creating_output(
    source: Path, tmp_path: Path, options: dict
) -> None:
    output = tmp_path / "out.mp4"

    with pytest.raises(ValueError):
        prepare_video(source, output, **options)

    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_existing_output_is_rejected_without_touching_it(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.mp4"
    output.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        prepare_video(source, output)

    assert output.read_bytes() == b"sentinel"
    assert not output.with_suffix(".json").exists()


def test_same_source_and_output_is_rejected_without_modifying_input(
    source: Path,
) -> None:
    before = _sha256(source)

    with pytest.raises(ValueError):
        prepare_video(source, source)

    assert _sha256(source) == before


def test_missing_source_is_rejected_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "out.mp4"

    with pytest.raises(FileNotFoundError):
        prepare_video(tmp_path / "missing.mp4", output)

    assert not output.exists()
    assert not output.with_suffix(".json").exists()


@pytest.mark.parametrize("start,end", [(2, 3), (0.5, 2)])
def test_out_of_source_ranges_cannot_be_reported_as_success(source, tmp_path, start, end):
    output = tmp_path / "outside.mp4"
    with pytest.raises(ValueError, match="duration|range"):
        prepare_video(source, output, start=start, end=end)
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_existing_sidecar_is_preserved(source, tmp_path):
    output = tmp_path / "new.mp4"
    metadata = output.with_suffix(".json")
    metadata.write_text("preserve existing metadata")
    with pytest.raises(FileExistsError):
        prepare_video(source, output)
    assert metadata.read_text() == "preserve existing metadata"
    assert not output.exists()


def test_vfr_content_timestamps_survive_cfr_conversion(tmp_path):
    import subprocess

    import imageio_ffmpeg
    from PIL import Image

    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    paths = []
    for i, colour in enumerate(colours):
        path = tmp_path / f"frame{i}.png"
        Image.new("RGB", (64, 64), colour).save(path)
        paths.append(path)
    concat = tmp_path / "frames.txt"
    concat.write_text(
        "".join(f"file '{p}'\nduration {d}\n" for p, d in zip(paths, [0.12, 0.20, 0.28, 0.40]))
        + f"file '{paths[-1]}'\n"
    )
    source = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "cfr.mp4"
    result = prepare_video(source, output, end=1, fps=30)
    reader = imageio.get_reader(output)
    try:
        assert reader.count_frames() == 30
        green = reader.get_data(6).mean((0, 1))
        blue = reader.get_data(15).mean((0, 1))
        assert green[1] > 200 and green[0] < 40
        assert blue[2] > 200 and blue[0] < 40
    finally:
        reader.close()
    assert result["frames"] == 30
    assert result["actual_duration_seconds"] == 1
