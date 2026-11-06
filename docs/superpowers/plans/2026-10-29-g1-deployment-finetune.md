# G1 Deployment Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Run an actual simulation RL fine-tuning experiment from the current dance checkpoint and document measured deployment before/after results.

**Architecture:** An experimental CPU collector uses the exact official-model NativeSimulator/PD implementation and the existing injected SuperMimic MDP/learner protocol. A learner subclass projects imagined actions to deployment limits. Explicit candidate exports and unchanged loopback DDS validate saved policies.

**Tech Stack:** Python,MuJoCo,NumPy,PyTorch CUDA,existing SuperMimic learned world model,SDK2 loopback evaluation.

**Spec:** docs/superpowers/specs/2026-10-29-g1-deployment-finetune-design.md

## Global Constraints

- No physical robot actions; DDS lo and nonzero domain only.
- Preserve source checkpoint,motion,all old outputs; new directories only.
- Same99x10/730/29 architecture and internalbeta0.35; no output-video smoothing.
- Candidate evaluation bundles must not pretend delivery/hardware acceptance.
- Separate tracking,zero-velocity,latency,standing entry and exit outcomes.

## Review Focus

- Episode ends must store final pre-reset transition before scattering reset state.
-5ms delay must use old target for exactly one of four substeps and reset actuator memory per row.
- Imagination/collection/deployment must agree on clipped targets and previous applied action.
- Warm-start must preserve source weights but reset counters/optimizers; source files must remain unchanged.
- Candidate selection must not cherry-pick DDS scheduling; compare repeated source/candidate runs and keep failures.

### Task1: Deployment collector and learner bridge

**Files:** Create src/unisupermimic/finetune/{__init__,collector,learner}.py and tests/finetune/.

**Interfaces:** DeploymentCollector exposes num_envs,device,supermimic_mdp,state,init_state,step,reset,close. It returns TorchEnvState with current/transition contracts. DeploymentLearner keeps source architecture and overrides imagined action projection only.

- [x] Write and observe RED tests for nominal transitions,delay,limits,pre/post reset boundaries,world/body gyro and invalid inputs.
- [x] Implement collector/learner and run focused GREEN plus normal regression suite.

### Task2: Reproducible experiment and candidate exporter

**Files:** Create src/unisupermimic/finetune/{train,export_candidate,evaluate}.py and finetune_projects/g1_deployment_20261029/{train.sh,README.md}; ignore generated runs/media.

**Interfaces:** CLI records complete source hashes/config/hyperparameters and produces numbered checkpoints; explicit candidate bundle reuses trusted reference but binds new policy/checkpoint and marks candidate status.

- [x] Write RED tests for candidate provenance,graph parity,overwrite guards and checkpoint/config compatibility.
- [x] Implement and smoke-test real CPU collector/GPU learner with a short run before full training.

### Task3: Actual fine-tuning and matched evaluation

**Files:** New generated run_001/ and evaluation directories under the experiment project.

- [x] Record source baselines with official explicit-PD/limits; train1000iterations with150WMwarmup and save250-step checkpoints.
- [x] Compare candidates under identical native/reference,zero-velocity and5ms-delay conditions, then repeat unchanged actual DDS source/candidate trials.
- [x] Render actual selected-policy50fps trajectory; preserve unsuccessful results and flag standing/exit separately.

### Task4: Review,verification and Chinese method summary

- [x] Fresh independent review; reproduce and fix substantive findings with regression tests.
- [x] Run final normal suite,focused real-training/DDS integration checks and changed-file lint; verify source hashes and media decoding.
- [x] Summarize existing fine-tuning mechanism,new recipe,actual gains/regressions,limitations and commands; leave hardware_ready=false.
