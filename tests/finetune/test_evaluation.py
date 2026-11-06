import json


def test_ranking_prefers_coverage_then_worst_case_then_tracking():
    from unisupermimic.finetune.evaluate import candidate_rank

    def case(steps, done, error):
        return dict(
            tracking=dict(steps=steps, expected_steps=100, completed=done),
            joint_tracking_rmse_rad=error,
        )

    assert candidate_rank([case(100, True, 0.2), case(90, False, 0.1)]) > candidate_rank(
        [case(99, False, 0.01), case(99, False, 0.01)]
    )
    assert candidate_rank([case(100, True, 0.2), case(95, False, 0.2)]) > candidate_rank(
        [case(100, True, 0.01), case(90, False, 0.01)]
    )
    assert candidate_rank([case(100, True, 0.1)]) > candidate_rank([case(100, True, 0.2)])


def test_native_matrix_keeps_all_predeclared_scenarios(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.finetune.evaluate import evaluate_matrix

    target = tmp_path / "matrix"
    summary = evaluate_matrix(adaptation_bundle, experiment_inputs["scene"], target)
    assert set(summary["cases"]) == {"reference", "delayed_5ms", "zero_velocity"}
    assert summary["hardware_ready"] is False
    assert json.loads((target / "summary.json").read_text())["rank"] == list(summary["rank"])
    for name, report in summary["cases"].items():
        assert report["tracking"]["expected_steps"] == 99
        assert report["tracking"]["steps"] <= 99
        assert report["standing_entry_verified"] is False
