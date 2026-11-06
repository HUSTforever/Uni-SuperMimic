# G1 Deployment Sim2sim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Run the existing G1 29-DoF exported dance policy through an independently validated deployment controller and local simulation transport.

**Architecture:** Export an immutable reference/controller bundle offline. A standalone controller consumes motor/IMU observations; direct MuJoCo and loopback SDK2/DDS adapters use that same controller. Keep diagnostic reference-reset tracking distinct from standing entry/exit and robustness checks.

**Tech Stack:** Python, NumPy, TorchScript, MuJoCo, official Unitree SDK2 Python on local loopback.

**Spec:** `docs/superpowers/specs/2026-10-29-g1-deployment-sim2sim-design.md`

## Global Constraints

- No hardware connection or robot actuation is authorized by this scope.
- Runtime must not call the training environment or MDP.
- DDS uses `lo` and nonzero domain only; no physical interfaces/domain zero/robot service changes.
- Preserve all old artifacts and unrelated files; no invented Git identity.
- Report reference-reset tracking, standing entry/exit and robustness independently.

## Review Focus

- Mismatched model/reference/gains, corrupted hash or wrong motor mapping must fail before control.
- Pelvis vs torso IMU, quaternion convention, heading alignment and body gyro errors must be caught by independent fixtures/oracle comparison.
- Histories and previous applied action must advance exactly once per 20ms reference frame; end clamping must not index another clip.
- Dropped/stale/out-of-order samples and invalid model output must latch a fault without another tracking command.
- A clip started at reference pose/velocity cannot be reported as a successful standing-to-dance deployment.

### Task 1: Bundle and independent policy interface

**Files:** Create `src/unisupermimic/deploy/{__init__,bundle,controller}.py`; test `tests/deploy/test_{bundle,controller}.py`.

**Interfaces:** `DeploymentBundle.load(directory)` supplies validated metadata, arrays and TorchScript path. `SensorState` supplies q/dq, pelvis quaternion wxyz, body gyro, sample time. `PolicyController` produces a 29-element nominal position target from sensor/reference state and maintains raw-action/history state.

- [x] Write failing tests for input equivalence, history ordering, target scaling, corruption, malformed state and terminal frames.
- [x] Observe RED, implement minimal bundle export/load and controller, then GREEN.
- [x] Compare independent observations/goals against the training oracle in tests only.

### Task 2: Independent native MuJoCo closed loop

**Files:** Create `src/unisupermimic/deploy/{mujoco_sim,run_sim2sim}.py`; test `tests/deploy/test_sim2sim.py`.

**Interfaces:** Explicit joint-name mapping, torque PD, finite/effort checks, controller step, report/NPZ/video artifacts. Native physics adapter exposes SensorState and accepts targets/gains; reference root state is evaluation/reset only.

- [x] Write and observe failing tests for mapping, torque saturation, no hidden root input, failure/completion reporting and no artifact overwrite.
- [x] Implement native runner; run actual 1048-step exported-policy tracking and separately test entry/exit.
- [x] Save timing/continuity/limit metrics and actual 50fps policy video in a new output directory.

### Task 3: SDK2/DDS loopback

**Files:** Create `src/unisupermimic/deploy/{dds,dds_sim}.py`; test `tests/deploy/test_dds.py`; add simulation dependency/setup instructions.

**Interfaces:** SDK2 HG LowState/LowCmd conversion with CRC, explicit natural-to-SDK mapping, loopback endpoint validation, sample age/sequence handling and bounded stop. Same controller as Task 2.

- [x] Write and observe failing tests for domain/interface rejection, state/message conversion, bad CRC/output and timeout/fault behavior.
- [x] Integrate installed SDK or install local official dependency in project environment; do not modify upstream repos.
- [x] Run actual two-process loopback simulation with published sensor states and commanded torques; record exact model/SDK identities, timing, entry/tracking/exit status.

### Task 4: Review and reproducible delivery

**Files:** Create `deploy/README.md`, `deploy/.gitignore`, curated result report and ledger; update relevant project docs only as needed.

- [x] Document commands, accepted model, input contract, result limitations and remaining hardware prerequisites.
- [x] Run full normal pytest suite and changed-file lint; fix substantive failures with RED→GREEN regressions.
- [x] Fresh independent code review focused on runtime data boundary and misleading readiness claims; resolve Critical/Important findings.
- [x] Verify artifact hashes, actual video decoding, true completion/failure fields and no live simulator processes before delivery.
