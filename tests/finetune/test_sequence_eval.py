import numpy as np
import pytest


def test_sequence_summary_requires_all_phases_and_stationary_final_window():
    from unisupermimic.finetune.sequence_eval import summarize_sequence

    meta = {
        "sequence": {
            "segments": {
                "standing": [0, 2],
                "entry": [2, 4],
                "dance": [4, 6],
                "exit": [6, 8],
                "settle": [8, 60],
            }
        }
    }
    qpos = np.zeros((61, 36))
    qpos[:, 2] = 0.75
    qpos[:, 3] = 1
    ref = {"q": np.zeros((61, 29))}
    good = summarize_sequence(meta, ref, qpos, np.zeros((61, 29)), np.zeros((61, 3)), 60)
    assert good["complete"] and good["final_hold"]["passed"]
    moving = np.ones((61, 29))
    bad = summarize_sequence(meta, ref, qpos, moving, np.zeros((61, 3)), 60)
    assert not bad["complete"] and not bad["final_hold"]["passed"]
    early = summarize_sequence(meta, ref, qpos[:6], moving[:6], np.zeros((6, 3)), 5)
    assert not early["complete"] and not early["phases"]["exit"]["completed"]
    missing = summarize_sequence(meta, ref, qpos[:60], np.zeros((60, 29)), np.zeros((60, 3)), 60)
    assert not missing["complete"] and not missing["phases"]["settle"]["completed"]


@pytest.mark.slow
def test_actual_dds_sequence_starts_at_rest_and_does_not_fake_legacy_entry(
    adaptation_bundle, experiment_inputs, tmp_path
):
    pytest.importorskip("unitree_sdk2py")
    import json
    import subprocess
    import sys

    from unisupermimic.deploy.bundle import DeploymentBundle
    from unisupermimic.finetune.sequence import build_sequence

    build_sequence(
        adaptation_bundle,
        experiment_inputs["scene"],
        tmp_path / "sequence",
        head_seconds=0.2,
        entry_seconds=0.4,
        exit_seconds=0.4,
        tail_seconds=1.0,
    )
    b = DeploymentBundle.load(tmp_path / "sequence")
    # SDK ChannelFactory is process-global: isolate each actual CLI run.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unisupermimic.deploy.run_sim2sim",
            "--bundle",
            str(b.directory),
            "--scene",
            str(experiment_inputs["scene"]),
            "--output",
            str(tmp_path / "dds"),
            "--transport",
            "dds",
            "--init",
            "sequence",
            "--domain",
            "47",
            "--no-video",
        ],
        capture_output=True,
        timeout=20,
    )
    result = json.loads((tmp_path / "dds/report.json").read_text())
    assert result["initial_velocity_norm"] == 0
    assert result["initialization"] == "sequence"
    assert result["tracking"]["steps"] > 0
    assert "sequence" in result
    assert not result["all_attempted_phases_passed"] or result["sequence"]["complete"]
