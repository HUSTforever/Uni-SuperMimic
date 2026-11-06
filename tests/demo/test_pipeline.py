"""Tests for the unified staged demo pipeline.

The heavyweight stages (GVHMR, GMR, training, MuJoCo rollout, export) are
replaced at the subprocess boundary.  The stub writes the small artifacts the
real command would produce, from the paths found in the command line, so the
pipeline's manifest and argument propagation are exercised without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from demo import pipeline


def _write_npz(path: Path) -> None:
    qpos = np.zeros((4, 7 + 29), dtype=np.float64)
    qpos[:, 3] = 1.0
    np.savez(path, qpos=qpos, fps=np.array([30.0]))


def _write_checkpoint(path: Path) -> None:
    path.write_bytes(b"checkpoint")


def _make_mocap(root: Path) -> Path:
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "gvhmr").write_text("#!/bin/sh\n")
    (root / ".venv-gmr" / "bin").mkdir(parents=True)
    (root / ".venv-gmr" / "bin" / "python").write_text("#!/bin/sh\n")
    (root / "GMR" / "assets" / "body_models").mkdir(parents=True)
    gmr_model = root / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    gmr_model.parent.mkdir(parents=True)
    gmr_model.write_text("<mujoco/>\n")
    return root


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _has(command: list[str], needle: str) -> bool:
    return any(needle in part for part in command)


class StubRunner:
    """Fake ``subprocess.run`` that produces the requested small artifacts."""

    def __init__(
        self,
        *,
        evaluation_completed: bool = True,
        fail_marker: str | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.evaluation_completed = evaluation_completed
        self.fail_marker = fail_marker

    def __call__(self, command, *, cwd, env, check):
        assert isinstance(command, list)
        assert check is True
        assert not any(isinstance(part, Path) for part in command)
        assert cwd == pipeline.ROOT
        assert isinstance(env, dict)
        command = [str(part) for part in command]
        self.calls.append(command)
        text = " ".join(command)
        if self.fail_marker is not None and self.fail_marker in text:
            raise subprocess.CalledProcessError(1, command)

        executable = Path(command[0]).name
        if executable == "gvhmr":
            video = Path(command[2])
            upstream = Path(_arg(command, "-o"))
            prediction = upstream / video.stem / "hmr4d_results.pt"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            prediction.write_bytes(b"gvhmr")
        elif len(command) > 1 and Path(command[1]).name == "prepare_video.py":
            source = Path(command[2])
            prepared = Path(command[3])
            assert source.is_file()
            assert _arg(command, "--fps") == "30"
            prepared.parent.mkdir(parents=True, exist_ok=True)
            prepared.write_bytes(b"prepared")
            prepared.with_suffix(".json").write_text("{}\n")
        elif len(command) > 1 and Path(command[1]).name == "retarget_gvhmr.py":
            assert Path(_arg(command, "--gvhmr_pred_file")).is_file()
            assert Path(_arg(command, "--gmr-root")).is_dir()
            output = Path(_arg(command, "--output"))
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_npz(output)
        elif len(command) > 1 and Path(command[1]).name == "convert_gmr_to_csv.py":
            csv_path = Path(command[3])
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(csv_path, np.zeros((4, 36)), delimiter=",")
            csv_path.with_suffix(".json").write_text("{}\n")
        elif len(command) > 1 and command[1] == "-c":
            overrides = command[2:]
            log_dir = Path(
                next(
                    item.split("=", 1)[1]
                    for item in overrides
                    if item.startswith("training.log_dir=")
                )
            )
            iterations = int(
                next(
                    item.split("=", 1)[1]
                    for item in overrides
                    if item.startswith("algo.iterations=")
                )
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"ckpt_{iterations:06d}.pt").write_bytes(b"trained")
        elif len(command) > 1 and Path(command[1]).name == "evaluate_motion.py":
            output = Path(_arg(command, "--output"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps({"completed": self.evaluation_completed, "backend": "stub"}) + "\n"
            )
            video = Path(_arg(command, "--video"))
            video.write_bytes(b"video")
        elif len(command) > 1 and Path(command[1]).name == "export_policy.py":
            output = Path(_arg(command, "--output"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"torchscript")
            output.with_suffix(output.suffix + ".json").write_text("{}")
        elif len(command) > 1 and Path(command[1]).name == "verify_torchscript.py":
            assert Path(_arg(command, "--ckpt")).is_file()
            assert Path(_arg(command, "--ts")).is_file()
        return subprocess.CompletedProcess(command, 0)


@pytest.fixture
def npz_input(tmp_path: Path) -> Path:
    path = tmp_path / "motion.npz"
    _write_npz(path)
    return path


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "init.pt"
    _write_checkpoint(path)
    return path


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubRunner:
    runner = StubRunner()
    monkeypatch.setattr(pipeline.subprocess, "run", runner)
    return runner


def _invoke(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["pipeline.py", *args])


def _read_manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def test_npz_without_training_runs_stages_in_order_and_uses_checkpoint(
    tmp_path: Path,
    npz_input: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--npz",
        str(npz_input),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--name",
        "my_clip",
        "--mocap-root",
        str(tmp_path / "mocap"),
        "--backend",
        "mjwarp",
        "--device",
        "cpu",
    )

    pipeline.main()

    manifest = _read_manifest(output)
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "completed"
    assert [stage["name"] for stage in manifest["stages"]] == [
        "convert",
        "evaluate",
        "export",
        "verify",
    ]
    assert manifest["input_sha256"]["npz"] == hashlib.sha256(npz_input.read_bytes()).hexdigest()
    assert (
        manifest["input_sha256"]["checkpoint"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert manifest["settings"]["name"] == "my_clip"
    assert manifest["settings"]["backend"] == "mjwarp"
    assert manifest["settings"]["train_iterations"] == 0

    assert not any("gvhmr" in call[0] for call in stub.calls)
    assert not any(_has(call, "retarget_to_g1.py") for call in stub.calls)
    assert not any(len(call) > 1 and call[1] == "-c" for call in stub.calls)

    csv_path = output / "my_clip.csv"
    convert = next(c for c in stub.calls if _has(c, "convert_gmr_to_csv.py"))
    assert convert[2] == str(npz_input)
    assert convert[3] == str(csv_path)
    assert "--ground-clamp" in convert
    assert _arg(convert, "--smooth-seconds") == "0.07"
    assert _arg(convert, "--slowdown") == "1.25"
    assert _arg(convert, "--face-yaw-deg") == "315"
    assert "--gmr-model" not in convert

    evaluate = next(c for c in stub.calls if _has(c, "evaluate_motion.py"))
    assert _arg(evaluate, "--ckpt") == str(checkpoint)
    assert _arg(evaluate, "--motion-files") == str(csv_path)
    assert _arg(evaluate, "--backend") == "mjwarp"
    assert _arg(evaluate, "--output") == str(output / "evaluation.json")
    assert _arg(evaluate, "--video") == str(output / "policy_vs_reference.mp4")
    assert _arg(evaluate, "--fps") == "50"
    assert _arg(evaluate, "--device") == "cpu"

    export = next(c for c in stub.calls if _has(c, "export_policy.py"))
    assert _arg(export, "--ckpt") == str(checkpoint)
    assert _arg(export, "--output") == str(output / "policy.ts")

    verify = next(c for c in stub.calls if _has(c, "verify_torchscript.py"))
    assert _arg(verify, "--ckpt") == str(checkpoint)
    assert _arg(verify, "--ts") == str(output / "policy.ts")
    assert _arg(verify, "--clip") == "my_clip"
    assert _arg(verify, "--seconds") == "2"
    assert _arg(verify, "--motion-files") == str(csv_path)

    assert manifest["stages"][0]["outputs"] == [
        str(csv_path),
        str(csv_path.with_suffix(".json")),
    ]
    assert manifest["stages"][1]["outputs"] == [
        str(output / "evaluation.json"),
        str(output / "policy_vs_reference.mp4"),
    ]


def test_video_uses_video_stem_for_prediction_and_propagates_conversion_options(
    tmp_path: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "dance_take.mp4"
    video.write_bytes(b"video")
    mocap = _make_mocap(tmp_path / "mocap")
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--video",
        str(video),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--mocap-root",
        str(mocap),
        "--name",
        "renamed_clip",
        "--smooth-seconds",
        "0.12",
        "--slowdown",
        "1.5",
        "--face-yaw-deg",
        "270",
    )

    pipeline.main()

    manifest = _read_manifest(output)
    assert [stage["name"] for stage in manifest["stages"]] == [
        "prepare_video",
        "gvhmr",
        "retarget",
        "convert",
        "evaluate",
        "export",
        "verify",
    ]
    prediction = output / "upstream" / "dance_take" / "hmr4d_results.pt"
    prepared = output / "prepared" / "dance_take.mp4"
    assert prediction.is_file()
    assert prepared.is_file()
    assert not (output / "upstream" / "renamed_clip").exists()

    prepare = next(c for c in stub.calls if _has(c, "prepare_video.py"))
    assert prepare[0] == sys.executable
    assert Path(prepare[1]) == pipeline.ROOT / "demo" / "prepare_video.py"
    assert Path(prepare[2]) == video
    assert Path(prepare[3]) == prepared
    assert _arg(prepare, "--fps") == "30"
    assert _arg(prepare, "--start") == "0"
    assert "--end" not in prepare
    assert manifest["settings"]["source_start"] == 0
    assert manifest["settings"]["source_end"] is None

    gvhmr = next(c for c in stub.calls if Path(c[0]).name == "gvhmr")
    assert Path(gvhmr[2]) == prepared
    assert _arg(gvhmr, "-o") == str(output / "upstream")
    assert "-s" in gvhmr
    assert "--camera" not in gvhmr
    assert manifest["settings"]["camera"] == "static"

    retarget = next(c for c in stub.calls if _has(c, "retarget_gvhmr.py"))
    assert Path(retarget[0]) == mocap / ".venv-gmr" / "bin" / "python"
    assert Path(retarget[1]) == pipeline.ROOT / "demo" / "retarget_gvhmr.py"
    assert _arg(retarget, "--gvhmr_pred_file") == str(prediction)
    assert _arg(retarget, "--output") == str(output / "g1_qpos.npz")
    assert _arg(retarget, "--gmr-root") == str(mocap / "GMR")
    assert _arg(retarget, "--src-fps") == "30"
    assert "--ground-human" in retarget
    assert not any(_has(call, "retarget_to_g1.py") for call in stub.calls)

    convert = next(c for c in stub.calls if _has(c, "convert_gmr_to_csv.py"))
    assert _arg(convert, "--gmr-model") == str(
        mocap / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    )
    assert _arg(convert, "--smooth-seconds") == "0.12"
    assert _arg(convert, "--slowdown") == "1.5"
    assert _arg(convert, "--face-yaw-deg") == "270"

    retarget_stage = manifest["stages"][2]
    assert retarget_stage["outputs"] == [str(output / "g1_qpos.npz")]


def test_video_simplevo_camera_propagates_without_static_flag(
    tmp_path: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "moving_take.mp4"
    video.write_bytes(b"video")
    mocap = _make_mocap(tmp_path / "mocap")
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--video",
        str(video),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--mocap-root",
        str(mocap),
        "--camera",
        "simplevo",
        "--flip-test",
    )

    pipeline.main()

    manifest = _read_manifest(output)
    assert manifest["settings"]["camera"] == "simplevo"
    prepared = output / "prepared" / "moving_take.mp4"
    assert prepared.is_file()
    gvhmr = next(c for c in stub.calls if Path(c[0]).name == "gvhmr")
    assert Path(gvhmr[2]) == prepared
    assert _arg(gvhmr, "--camera") == "simplevo"
    assert "-s" not in gvhmr
    assert gvhmr.count("--camera") == 1
    assert "--flip-test" in gvhmr

    retarget = next(c for c in stub.calls if _has(c, "retarget_gvhmr.py"))
    assert _arg(retarget, "--gmr-root") == str(mocap / "GMR")
    assert _arg(retarget, "--src-fps") == "30"
    assert "--ground-human" in retarget


def test_video_source_trim_propagates_to_prepare_video(
    tmp_path: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "long_take.mp4"
    video.write_bytes(b"video")
    mocap = _make_mocap(tmp_path / "mocap")
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--video",
        str(video),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--mocap-root",
        str(mocap),
        "--source-start",
        "0.25",
        "--source-end",
        "0.75",
    )

    pipeline.main()

    manifest = _read_manifest(output)
    assert [stage["name"] for stage in manifest["stages"]][:2] == ["prepare_video", "gvhmr"]
    prepared = output / "prepared" / "long_take.mp4"
    prepare = next(c for c in stub.calls if _has(c, "prepare_video.py"))
    assert Path(prepare[2]) == video
    assert Path(prepare[3]) == prepared
    assert _arg(prepare, "--start") == "0.25"
    assert _arg(prepare, "--end") == "0.75"
    assert manifest["settings"]["source_start"] == 0.25
    assert manifest["settings"]["source_end"] == 0.75
    assert manifest["stages"][0]["outputs"] == [
        str(prepared),
        str(prepared.with_suffix(".json")),
    ]
    assert Path(next(c for c in stub.calls if Path(c[0]).name == "gvhmr")[2]) == prepared


@pytest.mark.parametrize(
    "extra",
    [
        ["--source-start", "0.25"],
        ["--source-end", "0.75"],
        ["--source-start", "-0.1"],
        ["--source-start", "nan"],
        ["--source-start", "0.5", "--source-end", "0.25"],
        ["--source-end", "inf"],
    ],
)
def test_npz_rejects_source_time_range_without_creating_output(
    tmp_path: Path,
    npz_input: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
) -> None:
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--npz",
        str(npz_input),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        *extra,
    )

    with pytest.raises(SystemExit):
        pipeline.main()

    assert not output.exists()
    assert stub.calls == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--source-start", "-1"],
        ["--source-start", "0.5", "--source-end", "0.5"],
        ["--source-end", "nan"],
    ],
)
def test_video_rejects_invalid_source_time_range_without_creating_output(
    tmp_path: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
) -> None:
    video = tmp_path / "dance.mp4"
    video.write_bytes(b"video")
    mocap = _make_mocap(tmp_path / "mocap")
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--video",
        str(video),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--mocap-root",
        str(mocap),
        *extra,
    )

    with pytest.raises(SystemExit):
        pipeline.main()

    assert not output.exists()
    assert stub.calls == []


def test_video_preflight_requires_gmr_body_models(
    tmp_path: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "dance.mp4"
    video.write_bytes(b"video")
    mocap = tmp_path / "mocap"
    (mocap / ".venv" / "bin").mkdir(parents=True)
    (mocap / ".venv" / "bin" / "gvhmr").write_text("#!/bin/sh\n")
    (mocap / ".venv-gmr" / "bin").mkdir(parents=True)
    (mocap / ".venv-gmr" / "bin" / "python").write_text("#!/bin/sh\n")
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--video",
        str(video),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--mocap-root",
        str(mocap),
    )

    with pytest.raises(pipeline.PipelineError, match="body_models"):
        pipeline.main()

    assert not output.exists()
    assert stub.calls == []


def test_training_uses_fresh_run_checkpoint_and_hydra_overrides(
    tmp_path: Path,
    npz_input: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--npz",
        str(npz_input),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--mocap-root",
        str(tmp_path / "mocap"),
        "--name",
        "trained_clip",
        "--train-iterations",
        "7",
        "--num-envs",
        "16",
        "--action-smoothing",
        "0.4",
    )

    pipeline.main()

    csv_path = output / "trained_clip.csv"
    run_dir = output / "run"
    trained_checkpoint = run_dir / "ckpt_000007.pt"
    manifest = _read_manifest(output)
    assert [stage["name"] for stage in manifest["stages"]] == [
        "convert",
        "train",
        "evaluate",
        "export",
        "verify",
    ]

    train = next(c for c in stub.calls if len(c) > 1 and c[1] == "-c")
    assert "from unisupermimic.scripts.train_supermimic import hydra_main; hydra_main()" in train
    overrides = train[2:]
    assert f"training.log_dir={run_dir}" in overrides
    assert f"env.commands.motion.motion_files=[{csv_path}]" in overrides
    assert "algo.num_envs=16" in overrides
    assert "algo.iterations=7" in overrides
    assert "algo.action_smoothing=0.4" in overrides
    assert "algo.wm_only_iters=100" in overrides
    assert "algo.wm_batches=4" in overrides
    assert "algo.wm_batch=1024" in overrides
    assert "algo.pi_batch=1024" in overrides
    assert "algo.buffer_steps=600" in overrides
    assert "algo.lr=0.0001" in overrides
    assert "algo.ckpt_every=250" in overrides
    assert "algo.log_every=50" in overrides
    assert f"training.init_checkpoint={checkpoint}" in overrides

    evaluate = next(c for c in stub.calls if _has(c, "evaluate_motion.py"))
    assert _arg(evaluate, "--ckpt") == str(trained_checkpoint)
    export = next(c for c in stub.calls if _has(c, "export_policy.py"))
    assert _arg(export, "--ckpt") == str(trained_checkpoint)
    verify = next(c for c in stub.calls if _has(c, "verify_torchscript.py"))
    assert _arg(verify, "--ckpt") == str(trained_checkpoint)


def test_subprocess_failure_is_recorded_and_reraised(
    tmp_path: Path,
    npz_input: Path,
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StubRunner(fail_marker="convert_gmr_to_csv.py")
    monkeypatch.setattr(pipeline.subprocess, "run", runner)
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--npz",
        str(npz_input),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
    )

    with pytest.raises(subprocess.CalledProcessError):
        pipeline.main()

    manifest = _read_manifest(output)
    assert manifest["status"] == "failed"
    assert "convert_gmr_to_csv.py" in manifest["exception"]
    assert [stage["name"] for stage in manifest["stages"]] == ["convert"]
    assert manifest["stages"][0]["status"] == "failed"
    assert manifest["stages"][0]["error"]


def test_incomplete_evaluation_fails_before_export(
    tmp_path: Path,
    npz_input: Path,
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StubRunner(evaluation_completed=False)
    monkeypatch.setattr(pipeline.subprocess, "run", runner)
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--npz",
        str(npz_input),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
    )

    with pytest.raises(pipeline.PipelineError, match="completed"):
        pipeline.main()

    manifest = _read_manifest(output)
    assert manifest["status"] == "failed"
    assert manifest["stages"][-1]["name"] == "evaluate"
    assert manifest["stages"][-1]["status"] == "failed"
    assert "completed" in manifest["stages"][-1]["error"]
    assert (output / "evaluation.json").is_file()
    assert not any(_has(call, "export_policy.py") for call in runner.calls)
    assert not (output / "policy.ts").exists()


def test_existing_manifest_is_not_overwritten(
    tmp_path: Path,
    npz_input: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    manifest_path = output / "manifest.json"
    original = json.dumps({"schema_version": 1, "status": "completed", "sentinel": 1})
    manifest_path.write_text(original, encoding="utf-8")
    _invoke(
        monkeypatch,
        "--npz",
        str(npz_input),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
    )

    with pytest.raises(pipeline.PipelineError, match="manifest"):
        pipeline.main()

    assert manifest_path.read_text(encoding="utf-8") == original
    assert stub.calls == []


def test_existing_run_is_not_overwritten(
    tmp_path: Path,
    npz_input: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out"
    run_dir = output / "run"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "existing.pt"
    sentinel.write_bytes(b"existing")
    _invoke(
        monkeypatch,
        "--npz",
        str(npz_input),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
    )

    with pytest.raises(pipeline.PipelineError, match="run"):
        pipeline.main()

    assert sentinel.read_bytes() == b"existing"
    assert not (output / "manifest.json").exists()
    assert stub.calls == []


def test_invalid_name_does_not_create_output(
    tmp_path: Path,
    npz_input: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--npz",
        str(npz_input),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--name",
        "bad name!",
    )

    with pytest.raises(SystemExit):
        pipeline.main()

    assert not output.exists()
    assert stub.calls == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--smooth-seconds", "-0.1"],
        ["--slowdown", "0"],
        ["--train-iterations", "-1"],
        ["--num-envs", "0"],
        ["--action-smoothing", "0"],
        ["--action-smoothing", "1.1"],
    ],
)
def test_invalid_numeric_options_do_not_create_output(
    tmp_path: Path,
    npz_input: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
) -> None:
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--npz",
        str(npz_input),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        *extra,
    )

    with pytest.raises(SystemExit):
        pipeline.main()

    assert not output.exists()
    assert stub.calls == []


def test_zero_training_requires_existing_checkpoint(
    tmp_path: Path,
    npz_input: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out"
    _invoke(monkeypatch, "--npz", str(npz_input), "--output-dir", str(output))

    with pytest.raises(pipeline.PipelineError, match="checkpoint"):
        pipeline.main()

    assert not output.exists()
    assert stub.calls == []


def test_video_preflight_missing_environment_does_not_create_output(
    tmp_path: Path,
    checkpoint: Path,
    stub: StubRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "dance.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "out"
    _invoke(
        monkeypatch,
        "--video",
        str(video),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--mocap-root",
        str(tmp_path / "missing_mocap"),
    )

    with pytest.raises(pipeline.PipelineError, match="gvhmr"):
        pipeline.main()

    assert not output.exists()
    assert stub.calls == []


def test_clean_env_removes_only_socks_proxy_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7897")
    monkeypatch.setenv("all_proxy", "socks://127.0.0.1:7897")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7898")

    env = pipeline.clean_env()

    assert "ALL_PROXY" not in env
    assert "all_proxy" not in env
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert env["HTTP_PROXY"] == "http://127.0.0.1:7898"

    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:7897")
    assert pipeline.clean_env()["ALL_PROXY"] == "http://127.0.0.1:7897"


def test_pipeline_preserves_existing_motion_without_manifest(tmp_path, npz_input, checkpoint, stub):
    output = tmp_path / "previous_results"
    output.mkdir()
    original = output / "kick_rave_smooth.csv"
    original.write_text("preserve this motion")
    with pytest.raises(pipeline.PipelineError, match="non-empty"):
        pipeline.main(
            ["--npz", str(npz_input), "--checkpoint", str(checkpoint), "--output-dir", str(output)]
        )
    assert original.read_text() == "preserve this motion"
    assert not stub.calls


def test_pipeline_fails_if_export_interface_metadata_is_missing(
    tmp_path, npz_input, checkpoint, monkeypatch
):
    runner = StubRunner()

    def without_metadata(command, **kwargs):
        result = runner(command, **kwargs)
        if len(command) > 1 and Path(command[1]).name == "export_policy.py":
            output = Path(_arg(command, "--output"))
            output.with_suffix(output.suffix + ".json").unlink()
        return result

    monkeypatch.setattr(pipeline.subprocess, "run", without_metadata)
    output = tmp_path / "out"
    with pytest.raises(pipeline.PipelineError, match="did not produce"):
        pipeline.main(
            ["--npz", str(npz_input), "--checkpoint", str(checkpoint), "--output-dir", str(output)]
        )
    assert _read_manifest(output)["status"] == "failed"


def test_video_flight_mode_reaches_human_grounding_and_robot_conversion(tmp_path, checkpoint, stub):
    video = tmp_path / "jump.mp4"
    video.write_bytes(b"video")
    mocap = _make_mocap(tmp_path / "mocap")
    output = tmp_path / "out"
    pipeline.main(
        [
            "--video",
            str(video),
            "--output-dir",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--mocap-root",
            str(mocap),
            "--contact-mode",
            "preserve-flight",
        ]
    )
    retarget = next(c for c in stub.calls if _has(c, "retarget_gvhmr.py"))
    convert = next(c for c in stub.calls if _has(c, "convert_gmr_to_csv.py"))
    assert "--preserve-flight" in retarget
    assert "--preserve-flight" in convert
    assert _read_manifest(output)["settings"]["contact_mode"] == "preserve-flight"
