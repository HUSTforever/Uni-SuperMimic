import json

import pytest


def test_real_short_training_creates_fresh_run_and_preserves_source(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.finetune.train import run_training

    output = tmp_path / "training"
    before = experiment_inputs["checkpoint_path"].read_bytes()
    report = run_training(
        adaptation_bundle,
        experiment_inputs["scene"],
        experiment_inputs["checkpoint_path"],
        output,
        iterations=2,
        num_envs=2,
        wm_warmup=1,
        batch=4,
        wm_batches=1,
        device="cpu",
        compile=False,
        horizon=4,
        checkpoint_every=1,
        action_rate_weight=-0.3,
        balance_weight=1.0,
        ground_slip_weight=1,
        ankle_rate_weight=0.2,
        wm_ankle_velocity_weight=4,
        wm_root_velocity_weight=2,
        focus_intervals=[(20, 30)],
    )
    assert report["completed"] and report["iteration"] == 2
    assert (output / "ckpt_000000.pt").is_file()
    assert (output / "ckpt_000002.pt").is_file()
    assert experiment_inputs["checkpoint_path"].read_bytes() == before
    config = json.loads((output / "finetune_config.json").read_text())
    assert config["method"] == "simulation_rl_warm_start"
    assert config["policy_normalization_frozen"] is True
    assert config["action_rate_weight"] == -0.3
    assert config["balance_weight"] == 1.0
    assert config["ground_slip_weight"] == 1
    assert config["ankle_rate_weight"] == 0.2
    assert config["wm_ankle_velocity_weight"] == 4
    assert config["wm_root_velocity_weight"] == 2
    assert config["ankle_rate_cost"] == dict(
        command_velocity_scale_rad_s=2.0,
        measured_velocity_scale_rad_s=2.0,
        measured_weight=0.5,
    )
    assert config["balance_cost_coefficients"]["ground_slip"] == 1.0
    assert config["focus_intervals"] == [[20, 30]]
    with pytest.raises(FileExistsError):
        run_training(
            adaptation_bundle,
            experiment_inputs["scene"],
            experiment_inputs["checkpoint_path"],
            output,
            iterations=2,
            num_envs=2,
            device="cpu",
        )


@pytest.mark.parametrize("name", ["wm_ankle_velocity_weight", "wm_root_velocity_weight"])
@pytest.mark.parametrize("value", [0.5, float("nan")])
def test_invalid_wm_weights_do_not_create_a_training_directory(
    adaptation_bundle, experiment_inputs, tmp_path, name, value
):
    from unisupermimic.finetune.train import run_training

    output = tmp_path / "invalid_weight"
    with pytest.raises(ValueError, match="weight"):
        run_training(
            adaptation_bundle,
            experiment_inputs["scene"],
            experiment_inputs["checkpoint_path"],
            output,
            iterations=2,
            num_envs=2,
            wm_warmup=1,
            device="cpu",
            **{name: value},
        )
    assert not output.exists()
