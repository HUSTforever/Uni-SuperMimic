#!/usr/bin/env python3
"""Unified staged demo pipeline: video or GMR NPZ -> CSV -> train/eval/export.

Stages (only those applicable to the selected inputs are recorded):

``gvhmr``     video -> ``<output>/upstream/<video stem>/hmr4d_results.pt``
``retarget``  GVHMR prediction -> G1 qpos NPZ (GMR, ``--ground-human``)
``convert``   qpos NPZ -> SuperMimic 36-column CSV
``train``     optional fresh SuperMimic run (``<output>/run``)
``evaluate``  physical rollout -> ``evaluation.json`` + comparison video
``export``    checkpoint -> ``policy.ts``
``verify``    eager vs TorchScript closed-loop equivalence

Every subprocess runs from the repository root with a clean proxy environment.
The manifest is atomically rewritten before and after each stage so a failed
run always leaves its commands, status and diagnostics behind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SOCKS_SCHEMES = ("socks://", "socks4://", "socks5://")


class PipelineError(RuntimeError):
    """Raised for an invalid invocation, preflight failure or bad stage output."""


def clean_env() -> dict[str, str]:
    """Return the environment with socks ``ALL_PROXY`` values removed.

    A SOCKS proxy makes ``httpx`` (used by GVHMR/Hugging Face) fail.  Other
    proxy settings are left exactly as the caller supplied them; no proxy is
    injected.
    """

    env = dict(os.environ)
    for key in ("ALL_PROXY", "all_proxy"):
        value = env.get(key)
        if value is not None and value.strip().lower().startswith(SOCKS_SCHEMES):
            env.pop(key, None)
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _format_number(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--npz", type=Path, help="existing GMR qpos NPZ (skips GVHMR/GMR)")
    inputs.add_argument("--video", type=Path, help="source video (runs GVHMR + GMR first)")
    parser.add_argument(
        "--source-start", type=float, default=0.0, help="video trim start in seconds"
    )
    parser.add_argument("--source-end", type=float, default=None, help="video trim end in seconds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="kick_rave_smooth")
    parser.add_argument("--mocap-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-iterations", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--smooth-seconds", type=float, default=0.07)
    parser.add_argument("--slowdown", type=float, default=1.25)
    parser.add_argument("--face-yaw-deg", type=float, default=315.0)
    parser.add_argument("--action-smoothing", type=float, default=0.35)
    parser.add_argument("--backend", choices=("mujoco", "mjwarp"), default="mujoco")
    parser.add_argument("--camera", choices=("static", "simplevo"), default="static")
    parser.add_argument(
        "--flip-test", action="store_true", help="average GVHMR inference with mirrored input"
    )
    parser.add_argument(
        "--contact-mode",
        choices=("support", "preserve-flight"),
        default="support",
        help="support approximates one grounded foot; preserve-flight keeps vertical takeoff and requires reliable world height",
    )
    parser.add_argument("--device", default=None)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if NAME_PATTERN.fullmatch(args.name) is None:
        parser.error("--name may contain only letters, digits, '_' and '-'")
    if args.train_iterations < 0:
        parser.error("--train-iterations must be >= 0")
    if args.num_envs <= 0:
        parser.error("--num-envs must be > 0")
    if not math.isfinite(args.smooth_seconds) or args.smooth_seconds < 0:
        parser.error("--smooth-seconds must be a finite value >= 0")
    if not math.isfinite(args.slowdown) or args.slowdown <= 0:
        parser.error("--slowdown must be a finite value > 0")
    if not math.isfinite(args.face_yaw_deg):
        parser.error("--face-yaw-deg must be finite")
    if not math.isfinite(args.action_smoothing) or not 0 < args.action_smoothing <= 1:
        parser.error("--action-smoothing must be in (0, 1]")
    if not math.isfinite(args.source_start) or args.source_start < 0:
        parser.error("--source-start must be a finite value >= 0")
    if args.source_end is not None:
        if not math.isfinite(args.source_end):
            parser.error("--source-end must be finite")
        if args.source_end <= args.source_start:
            parser.error("--source-end must be greater than --source-start")
    if args.npz is not None and (args.source_start != 0 or args.source_end is not None):
        parser.error("--source-start/--source-end are only valid with --video")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PipelineError(f"missing {label}: {path}")


def _preflight_mocap(mocap_root: Path) -> None:
    _require_file(mocap_root / ".venv" / "bin" / "gvhmr", "gvhmr executable")
    _require_file(mocap_root / ".venv-gmr" / "bin" / "python", "GMR python")
    body_models = mocap_root / "GMR" / "assets" / "body_models"
    if not body_models.is_dir():
        raise PipelineError(f"missing GMR body models directory: {body_models}")


def _preflight_output(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise PipelineError(f"--output-dir is not a directory: {output_dir}")
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        raise PipelineError(
            f"refusing to overwrite existing manifest {manifest_path}; choose a new --output-dir"
        )
    if (output_dir / "run").exists():
        raise PipelineError(
            f"refusing to overwrite existing run {output_dir / 'run'}; choose a new --output-dir"
        )
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise PipelineError(
            f"refusing non-empty output directory: {output_dir}; choose a new --output-dir"
        )


def _evaluation_completed(path: Path) -> None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read evaluation result {path}: {exc}") from exc
    if not isinstance(result, dict) or result.get("completed") is not True:
        completed = result.get("completed") if isinstance(result, dict) else None
        raise PipelineError(
            f"evaluation did not complete successfully (completed={completed!r}): {path}"
        )


def _run_stage(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    name: str,
    command: Sequence[object],
    outputs: Sequence[Path],
    required: Sequence[Path] = (),
    validate: Callable[[], None] | None = None,
) -> None:
    stage: dict[str, Any] = {
        "name": name,
        "command": [str(part) for part in command],
        "status": "running",
        "error": None,
        "outputs": [],
    }
    manifest["stages"].append(stage)
    _atomic_write_json(manifest_path, manifest)
    try:
        subprocess.run(
            [str(part) for part in command],
            cwd=ROOT,
            env=clean_env(),
            check=True,
        )
        if validate is not None:
            validate()
        for path in required:
            if not path.is_file():
                raise PipelineError(f"stage {name} did not produce {path}")
        stage["outputs"] = [str(path) for path in outputs]
        stage["status"] = "completed"
        _atomic_write_json(manifest_path, manifest)
    except Exception as exc:
        stage["status"] = "failed"
        stage["error"] = f"{type(exc).__name__}: {exc}"
        stage["outputs"] = [str(path) for path in outputs if path.exists()]
        manifest["status"] = "failed"
        manifest["exception"] = stage["error"]
        _atomic_write_json(manifest_path, manifest)
        raise


def _print_artifacts(
    manifest_path: Path,
    csv_path: Path,
    checkpoint: Path,
    evaluation_path: Path,
    video_path: Path,
    policy_path: Path,
) -> None:
    print("status: completed")
    for label, path in (
        ("manifest", manifest_path),
        ("csv", csv_path),
        ("checkpoint", checkpoint),
        ("evaluation", evaluation_path),
        ("video", video_path),
        ("policy", policy_path),
    ):
        print(f"{label}: {path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    output_dir = args.output_dir.resolve()
    npz = args.npz.resolve() if args.npz is not None else None
    video = args.video.resolve() if args.video is not None else None
    mocap_root = (
        args.mocap_root.resolve()
        if args.mocap_root is not None
        else (ROOT.parent / "mocap").resolve()
    )
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None

    # Preflight every input before creating the output directory.
    if npz is not None:
        _require_file(npz, "input NPZ")
    if video is not None:
        _require_file(video, "input video")
        _preflight_mocap(mocap_root)
    if args.train_iterations == 0:
        if checkpoint is None:
            raise PipelineError("--checkpoint is required when --train-iterations is 0")
        _require_file(checkpoint, "checkpoint")
    elif checkpoint is not None:
        _require_file(checkpoint, "checkpoint")
    _preflight_output(output_dir)

    settings = {
        "name": args.name,
        "npz": str(npz) if npz is not None else None,
        "video": str(video) if video is not None else None,
        "source_start": args.source_start,
        "source_end": args.source_end,
        "output_dir": str(output_dir),
        "mocap_root": str(mocap_root),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "train_iterations": args.train_iterations,
        "num_envs": args.num_envs,
        "smooth_seconds": args.smooth_seconds,
        "slowdown": args.slowdown,
        "face_yaw_deg": args.face_yaw_deg,
        "action_smoothing": args.action_smoothing,
        "backend": args.backend,
        "camera": args.camera,
        "flip_test": args.flip_test,
        "contact_mode": args.contact_mode,
        "device": args.device,
    }
    input_sha256: dict[str, str] = {}
    if npz is not None:
        input_sha256["npz"] = _sha256(npz)
    if video is not None:
        input_sha256["video"] = _sha256(video)
    if checkpoint is not None:
        input_sha256["checkpoint"] = _sha256(checkpoint)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "settings": settings,
        "input_sha256": input_sha256,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "stages": [],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    _atomic_write_json(manifest_path, manifest)

    csv_path = output_dir / f"{args.name}.csv"
    evaluation_path = output_dir / "evaluation.json"
    video_path = output_dir / "policy_vs_reference.mp4"
    policy_path = output_dir / "policy.ts"

    try:
        if video is not None:
            upstream = output_dir / "upstream"
            prepared = output_dir / "prepared" / f"{video.stem}.mp4"
            qpos = output_dir / "g1_qpos.npz"
            prepare_command: list[object] = [
                sys.executable,
                ROOT / "demo" / "prepare_video.py",
                video,
                prepared,
                "--fps",
                "30",
                "--start",
                _format_number(args.source_start),
            ]
            if args.source_end is not None:
                prepare_command += ["--end", _format_number(args.source_end)]
            _run_stage(
                manifest,
                manifest_path,
                name="prepare_video",
                command=prepare_command,
                outputs=(prepared, prepared.with_suffix(".json")),
                required=(prepared, prepared.with_suffix(".json")),
            )
            prediction = upstream / video.stem / "hmr4d_results.pt"
            gvhmr_command: list[object] = [
                mocap_root / ".venv" / "bin" / "gvhmr",
                "demo",
                prepared,
            ]
            if args.camera == "static":
                gvhmr_command.append("-s")
            else:
                gvhmr_command += ["--camera", "simplevo"]
            gvhmr_command += ["-o", upstream]
            if args.flip_test:
                gvhmr_command.append("--flip-test")
            _run_stage(
                manifest,
                manifest_path,
                name="gvhmr",
                command=gvhmr_command,
                outputs=(prediction,),
                required=(prediction,),
            )
            _run_stage(
                manifest,
                manifest_path,
                name="retarget",
                command=[
                    mocap_root / ".venv-gmr" / "bin" / "python",
                    ROOT / "demo" / "retarget_gvhmr.py",
                    "--gvhmr_pred_file",
                    prediction,
                    "--output",
                    qpos,
                    "--gmr-root",
                    mocap_root / "GMR",
                    "--src-fps",
                    "30",
                    "--ground-human",
                    *(["--preserve-flight"] if args.contact_mode == "preserve-flight" else []),
                ],
                outputs=(qpos,),
                required=(qpos,),
            )
        else:
            assert npz is not None
            qpos = npz

        convert_command: list[object] = [
            sys.executable,
            ROOT / "demo" / "convert_gmr_to_csv.py",
            qpos,
            csv_path,
            "--ground-clamp",
            "--smooth-seconds",
            _format_number(args.smooth_seconds),
            "--slowdown",
            _format_number(args.slowdown),
            "--face-yaw-deg",
            _format_number(args.face_yaw_deg),
        ]
        gmr_model = mocap_root / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
        if args.contact_mode == "preserve-flight":
            convert_command.append("--preserve-flight")
        if gmr_model.is_file():
            convert_command += ["--gmr-model", gmr_model]
        _run_stage(
            manifest,
            manifest_path,
            name="convert",
            command=convert_command,
            outputs=(csv_path, csv_path.with_suffix(".json")),
            required=(csv_path, csv_path.with_suffix(".json")),
        )

        if args.train_iterations > 0:
            run_dir = output_dir / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            trained_checkpoint = run_dir / f"ckpt_{args.train_iterations:06d}.pt"
            overrides = [
                f"training.log_dir={run_dir}",
                f"env.commands.motion.motion_files=[{csv_path}]",
                f"algo.num_envs={args.num_envs}",
                f"algo.iterations={args.train_iterations}",
                f"algo.action_smoothing={_format_number(args.action_smoothing)}",
                "algo.wm_only_iters=100",
                "algo.wm_batches=4",
                "algo.wm_batch=1024",
                "algo.pi_batch=1024",
                "algo.buffer_steps=600",
                "algo.lr=0.0001",
                "algo.ckpt_every=250",
                "algo.log_every=50",
            ]
            if checkpoint is not None:
                overrides.append(f"training.init_checkpoint={checkpoint}")
            _run_stage(
                manifest,
                manifest_path,
                name="train",
                command=[
                    sys.executable,
                    "-c",
                    "from unisupermimic.scripts.train_supermimic import hydra_main; hydra_main()",
                    *overrides,
                ],
                outputs=(run_dir, trained_checkpoint),
                required=(trained_checkpoint,),
            )
            checkpoint = trained_checkpoint
            manifest["checkpoint"] = str(checkpoint)
            _atomic_write_json(manifest_path, manifest)
        else:
            assert checkpoint is not None
            _require_file(checkpoint, "checkpoint")

        evaluate_command: list[object] = [
            sys.executable,
            ROOT / "demo" / "evaluate_motion.py",
            "--ckpt",
            checkpoint,
            "--motion-files",
            csv_path,
            "--backend",
            args.backend,
            "--output",
            evaluation_path,
            "--video",
            video_path,
            "--fps",
            "50",
        ]
        if args.device is not None:
            evaluate_command += ["--device", args.device]
        _run_stage(
            manifest,
            manifest_path,
            name="evaluate",
            command=evaluate_command,
            outputs=(evaluation_path, video_path),
            required=(evaluation_path, video_path),
            validate=lambda: _evaluation_completed(evaluation_path),
        )

        export_command: list[object] = [
            sys.executable,
            ROOT / "demo" / "export_policy.py",
            "--ckpt",
            checkpoint,
            "--output",
            policy_path,
        ]
        _run_stage(
            manifest,
            manifest_path,
            name="export",
            command=export_command,
            outputs=(policy_path, policy_path.with_name(policy_path.name + ".json")),
            required=(policy_path, policy_path.with_suffix(policy_path.suffix + ".json")),
        )

        _run_stage(
            manifest,
            manifest_path,
            name="verify",
            command=[
                sys.executable,
                ROOT / "demo" / "verify_torchscript.py",
                "--ckpt",
                checkpoint,
                "--ts",
                policy_path,
                "--clip",
                args.name,
                "--seconds",
                "2",
                "--full-clip",
                "--motion-files",
                csv_path,
            ],
            outputs=(),
        )

        manifest["status"] = "completed"
        _atomic_write_json(manifest_path, manifest)
    except Exception as exc:
        if manifest["status"] != "failed":
            manifest["status"] = "failed"
            manifest["exception"] = f"{type(exc).__name__}: {exc}"
            _atomic_write_json(manifest_path, manifest)
        raise

    _print_artifacts(manifest_path, csv_path, checkpoint, evaluation_path, video_path, policy_path)


if __name__ == "__main__":
    main()
