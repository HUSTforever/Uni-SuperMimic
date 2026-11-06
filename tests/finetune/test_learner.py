import torch


def test_imagined_projection_has_correct_targets_and_unsaturated_gradients():
    from unisupermimic.finetune.learner import project_raw_action

    action = torch.tensor([[0.2, 10.0]], requires_grad=True)
    applied = project_raw_action(
        action,
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([-1.0, -1.0]),
        torch.tensor([1.0, 1.0]),
    )
    torch.testing.assert_close(applied, torch.tensor([[0.2, 1.6]]))
    applied.sum().backward()
    torch.testing.assert_close(action.grad, torch.tensor([[1.0, 0.0]]))


def test_world_model_warmup_preserves_all_policy_parameters_and_normalization(
    adaptation_bundle, experiment_inputs, tmp_path
):
    from unisupermimic.finetune.collector import DeploymentCollector
    from unisupermimic.finetune.learner import DeploymentLearner
    from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig

    c = DeploymentCollector(
        adaptation_bundle, experiment_inputs["scene"], num_envs=2, randomize=False
    )
    cfg = SuperMimicConfig(
        iterations=2,
        buffer_steps=16,
        horizon=2,
        goal_frames=10,
        hist=10,
        z_dim=4,
        wm_hidden=(8,),
        policy_hidden=(8,),
        critic_hidden=(8,),
        wm_window=2,
        wm_batches=1,
        wm_batch=4,
        pi_batch=4,
        warmup_steps=10,
        wm_only_iters=2,
        log_backend="no_print",
        amp="none",
        compile=False,
        log_every=1,
        ckpt_every=2,
        action_smoothing=0.35,
    )
    learner = DeploymentLearner(c, cfg, tmp_path / "warmup", device="cpu")
    learner.initialize_weights(experiment_inputs["checkpoint_path"])
    pi_before = {k: v.clone() for k, v in learner.pi.state_dict().items()}
    wm_before = {k: v.clone() for k, v in learner.wm.named_parameters()}
    assert learner.iteration == 0 and not learner.opt_pi.state
    learner.train()
    for key, value in learner.pi.state_dict().items():
        torch.testing.assert_close(value, pi_before[key], atol=0, rtol=0)
    assert any(
        not torch.equal(value, wm_before[key]) for key, value in learner.wm.named_parameters()
    )
