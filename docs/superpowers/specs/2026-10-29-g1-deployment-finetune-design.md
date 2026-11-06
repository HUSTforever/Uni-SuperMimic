# G1 deployment adaptation fine-tuning experiment

## User request and existing capability

The user explicitly authorizes trying fine-tuning and summarizing the method. The repository already supports `training.init_checkpoint` / `SuperMimicLearner.initialize_weights`: warm-start policy and learned world model in a new run, reset optimizers/counters/replay, optionally load a saved critic. The current dance run_002 itself used that capability. It lacks a deployment-adaptation recipe with measured DDS before/after results.

## Approach

Continuing the unchanged old training environment is cheap but does not target the measured gap. Replacing the entire project with an official PPO tracker would change network/data contracts and cannot reuse this checkpoint directly. Select a local experimental MuJoCo collector that reuses the *same NativeSimulator explicit PD and official G1 model* as deployment and feeds the existing SuperMimic learner via its injected MDP/env protocol. Keep policy architecture (99D x10 history,730D goal,29raw actions,beta0.35), motion CSV and reference phase unchanged.

This is simulation RL weight fine-tuning, not LoRA and not on-robot learning. No robot communication is authorized; actual DDS evaluation remains loopback/nonzero-domain only.

## Experiment

- Start from `dance_projects/BV1ct4y1n7t9/run_002/ckpt_001000.pt` and trusted bundle_v3 reference/controller constants.
- Separate project `finetune_projects/g1_deployment_20261029/`; refuse existing run/output directories. Keep every old model/result.
- CPU MuJoCo collectors, GPU world model/policy updates. Begin with64envs, lr5e-5,1000iterations,150world-model-only iterations,4world-model batches of512,policy batch512,600replay steps. Record actual throughput and adjust batch/count only if required by measured resource usage.
- Warm-start wm/pi, keep normalization statistics, fresh optimizer/replay/counters; critic initializes anew when numbered source checkpoint lacks critic.
- Explicit PD at200Hz and policy50Hz, joint target clipping and effort saturation exactly as deployment. Nominal context offsets zero. Friction per episode in[0.7,1.1], small pose/velocity reset perturbations. Mix reference and zero initial velocity; mix zero/one5ms delayed command substep. Preserve previous applied raw action after clipping. Delay uses previous held target only after the first episode step.
- Delay adds hidden actuator memory that the original world model does not explicitly parameterize; monitor model error and report this limitation. Do not claim an exact DDS timing model.
- Differentiable policy imagination projects raw actions to the same target ranges as collection/deployment, preserving gradients through unsaturated ranges. Ordinary project training behavior stays unchanged.
- Existing adaptive-style sampling can be approximated by explicit uniform/hard-region sampling in this experimental collector; identify the rule exactly and use deterministic seeds. Hard interval14–18s is from observed DDS failures, not unseen validation data.

## Acceptance and reporting

- Prove collector transitions contain pre-reset state/reward/final frame and post-reset current state, including row-specific resets; no off-by-one or fabricated completion.
- Prove identical nominal NativeSimulator state transitions for no delay, correct old/new target duration for5ms delay, finite actions, applied-action feedback and correct sensor/world-state convention.
- Tests show actor is not updated during wm-only phase and warm-start leaves optimizer/counters fresh. Preserve source checkpoint SHA.
- Evaluate source weights and multiple saved candidates with the same motion, model, physical limits, reference-reset, zero-velocity and delayed deterministic scenarios. Select by completion and tracking quality, then run repeated actual DDS on source and selected candidate without changing controller parameters.
- Export candidates as explicitly unvalidated evaluation bundles with real checkpoint/reference provenance; do not fabricate `accepted_for_delivery`. Hardware readiness remains false even if tracking passes.
- Save training logs, all candidates, raw trajectories, reports, a real50fps video and a concise Chinese method summary covering commands/hyperparameters/results/limitations. Report failed experiments and distinguish simulation tracking from standing entry/exit.

## Worktree and scope decisions

Continue in the current codex/g1-continuous-pipeline checkout: required prior changes are staged and uncommitted and models are local. Do not alter unrelated demo/live_viewer.py or root train_supermimic.log. No merge/push, invented Git author or physical hardware actions. User authorization covers this concrete experiment and documentation; routine implementation choices are owned by Codex.

## Builder-check rulings

- Collector hard termination combines original reference-relative gates with absolute pelvis tilt1.2rad and height0.35m; differentiable survival includes matching sigmoid gates.
- Freeze policy observation/goal normalization for the entire adaptation run by setting their update caps to zero. WM normalization retains its statistics and continues updating. WM-only phase changes no policy state_dict tensor, including normalization buffers.
- Reset sampling:25%firstframe,25%uniformwithin14–18s hardinterval,50%uniformwholeclip; zero-velocity reset25%,delay0/5ms equally likely. Small fixtures with no hardinterval use the last available interval; training references remain unchanged.
- Native candidate ranking: maximize completed-scenario count,then worst completed fraction,then minimize mean joint RMSE across fixed reference/no-delay,reference/5ms-delay,zero-velocity/no-delay cases. Evaluate all250/500/750/1000 checkpoints. Actual DDS:3interleaved repeats source0/candidate0/source1/candidate1/source2/candidate2,unchanged controller/model/limits. Full improvement requires3/3candidate completions with no nominal native tracking regression>25%RMSE; otherwise report partial/inconclusive/regression honestly. Standing/exit remain separate.
- Use source run_002/last.pt for the full experiment after verifying all pi/wm tensors and iteration1000 are exactly equal to ckpt_001000.pt. This preserves its existing critic as well; initialize_weights still discards optimizer/counters/replay. The numbered checkpoint was used for the8-iteration GPU smoke test.
