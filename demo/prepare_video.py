#!/usr/bin/env python3
"""Normalize a source video to CFR H.264 before GVHMR (which assumes 30 fps)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_video(
    source: Path,
    output: Path,
    *,
    start: float = 0,
    end: float | None = None,
    fps: float = 30,
) -> dict[str, Any]:
    """Write ``output`` as CFR ``fps`` H.264 and return its metadata sidecar."""

    source_path = Path(source)
    output_path = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(f"source video does not exist: {source_path}")
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise ValueError("source and output must be different paths")
    sidecar = output_path.with_suffix(".json")
    if output_path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if not math.isfinite(start) or start < 0:
        raise ValueError("start must be a finite value >= 0")
    if end is not None and (not math.isfinite(end) or end <= start):
        raise ValueError("end must be finite and greater than start")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite value > 0")

    # Imported lazily so importing this module does not require a video backend.
    import imageio_ffmpeg

    probe = imageio_ffmpeg.read_frames(str(source_path))
    try:
        source_metadata = next(probe)
    finally:
        probe.close()
    # VFR container headers can under-report duration; decode the video stream
    # timestamps rather than trusting average fps or the container's duration.
    source_frames, source_duration = imageio_ffmpeg.count_frames_and_secs(str(source_path))
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise ValueError("source video duration is unknown or empty")
    if start >= source_duration or (end is not None and end > source_duration + 1e-6):
        raise ValueError(f"requested range exceeds source duration {source_duration:.6g}s")

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-i",
        str(source_path),
    ]
    if end is not None:
        command += ["-t", str(end - start)]
    command += [
        "-vf",
        f"fps={fps}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True)
        frames, _ = imageio_ffmpeg.count_frames_and_secs(str(output_path))
        requested_duration = (source_duration if end is None else end) - start
        expected_frames = round(requested_duration * fps)
        if frames < 2 or abs(frames - expected_frames) > max(2, round(fps * 0.1)):
            raise ValueError(
                f"decoded video coverage is incomplete: expected about {expected_frames} frames, got {frames}"
            )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    metadata = {
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "start": float(start),
        "end": None if end is None else float(end),
        "fps": float(fps),
        "source_duration_seconds": source_duration,
        "source_reported_duration_seconds": float(source_metadata.get("duration", 0)),
        "source_frames": source_frames,
        "frames": frames,
        "actual_duration_seconds": frames / float(fps),
        "actual_source_start": float(start),
        "actual_source_end": min(
            source_duration if end is None else end, float(start) + frames / float(fps)
        ),
        "output_sha256": _sha256(output_path),
    }
    with sidecar.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)
    metadata = prepare_video(args.source, args.output, start=args.start, end=args.end, fps=args.fps)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
