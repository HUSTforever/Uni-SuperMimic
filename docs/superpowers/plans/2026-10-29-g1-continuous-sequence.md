# G1 Continuous Sequence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Complete actual standing–dance–standing sim2sim with continuous physical motion and explicit stability evidence.
**Architecture:** New kinematic sequence bundle,existing deployment-physics RL warm-start,new explicit sequence evaluation mode for native/DDS without legacy PD transitions.
**Tech Stack:** NumPy,TorchSO3,MuJoCo,PyTorch,SDK2loopback.
**Spec:** docs/superpowers/specs/2026-10-29-g1-continuous-sequence-design.md

## Global Constraints
- No real robot actions,lo/nonzerodomain only; preserve all old artifacts.
- Source dance poses unchanged; record actual motion with no video smoothing.
- Zero-velocity stationary start and physical completion of every phase/final hold.

## Review Focus
- Endpoint joins,quaternion convention,source slice/frame indexing.
- Sequence mode must bypass legacy fixedPD entry/exit but not skip physical standing phases.
- Final stability uses actual observed velocities/tilt and cannot pass on early stop.
- New model/reference provenance explicit; no fabricated acceptance.

### Task1: Sequence preparation
- [x] RED tests for preserved source slice,quintic endpoint derivatives,unit rotations,stationary head/tail and overwrite guards.
- [x] Implement new sequence builder and generate a separate bundle.

### Task2: Sequence evaluation and sampling
- [x] RED tests for sequence phase/finalhold reporting and zero-velocity initialization.
- [x] Add explicit sequence native/DDS mode,retain legacy behavior; bias fine-tune reset sampling to new transitions.

### Task3: Actual iterative training/verification
- [x] Warm-start a new1500iteration run,compare saved checkpoints and preserve failed trials.
- [x] Repeat full-sequence actual DDS3times; test delay/perturbations and smoothness vs prior dance.

### Task4: Review and delivery
- [x] Independent review and relevant regression suite/lint; inspect real50fps fullsequencevideo.
- [x] Document stable cases,remaininglimits,commands and new candidate paths. Keep hardware_ready=false.
