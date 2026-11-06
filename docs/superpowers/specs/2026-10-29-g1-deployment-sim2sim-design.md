# G1 deployment-program sim2sim

## Intent and authorization

The user asked whether the current dance can run on hardware, then said “继续” after the proposed next step: drive a G1 29-DoF simulator using the deployment program and exported policy before hardware testing. Implement that next step locally. No hardware connection or robot actuation is authorized by this scope.

## Selected approach

1. Reusing the training evaluation environment is already available, but cannot validate the deployment observation/action bridge.
2. An independent NumPy/Torch runtime with native MuJoCo provides deterministic diagnosis of that bridge.
3. The same runtime connected through Unitree SDK2/DDS on loopback validates the communication boundary as well. This is the recommended route accepted by “继续”; include the native runner as the diagnostic baseline, then exercise DDS if the local SDK can run.

Use existing Python, TorchScript, MuJoCo and the locally available official Unitree SDK. Do not migrate old assets or change previous training outputs. Work on the current `codex/g1-continuous-pipeline` branch because the validated pipeline and artifacts are uncommitted there; copying only HEAD to a worktree would omit dependencies. Do not commit with an invented Git identity.

## Architecture

- Offline bundle exporter: reads the selected `.ts`, `.ts.json`, CSV/metadata and training robot constants; precomputes reference goals at 50 Hz with the existing MotionLib. Stores root poses, velocities, joints and goals in a non-pickle NPZ, a manifest with hashes, and explicit controller gains/effort limits/joint limits. The runtime must not call the training environment or MDP.
- Independent controller: validates the complete bundle before running; constructs 99 raw observation features from ordered motor state, pelvis rotation/body gyro, previous applied raw action and the current reference orientation; keeps 10 observations with newest last; feeds precomputed 730D goals to TorchScript. No external duplicate smoothing: the policy already includes beta=0.35.
- Joint target: `default_q + context_offset + action_scale * raw_action`, with context_offset zero for nominal deployment. Explicit named SDK motor mapping; PR control, zero desired velocity/feedforward torque, exported training kp/kd.
- Native diagnostic runner: direct MuJoCo physics loop with explicit torque PD and physical effort limits, no RL environment imports. Reference-state initialization is a named diagnostic mode, never evidence of successful standing entry. A separate zero-velocity/standing-entry mode measures the actual transition.
- DDS runner: Unitree HG LowState/LowCmd on `lo` and a nonzero domain only. Refuse physical network interfaces and domain zero in this implementation. No MotionSwitcher client, robot-network setup or onboard service changes. Use actual SDK serialization and CRC. Where a local headless simulator adapter is necessary, label it accurately; distinguish it from the unmodified official simulator and record model/SDK identities.
- Logs: policy inputs/outputs, joint targets and measured trajectory, inference timing, clipping and failures. Physics root position is permitted for offline evaluation only; online policy input uses only motor/IMU fields and reference data.

## Startup, end and faults

Validate finite values, shapes, quaternion norm, monotonic simulation/sample time and fresh state before producing commands. Invalid/stale state, excessive tilt or model errors latch a fault; send bounded damping/stop when transport is available. Sequence entry, tracking, exit and completion explicitly. Do not claim hardware-ready or full deployment success when only reference-reset tracking passes. Start/stop transitions and timing/noise robustness have their own results.

## Acceptance

- Independently constructed observations and histories agree with the training oracle for nontrivial rotations, gyro, joints, previous actions and end-of-reference clamping.
- Mutated or incompatible artifacts fail before stepping/communication. NaN, invalid mapping, wrong dimensions, stale/out-of-order state and frame exhaustion cannot silently continue.
- Native closed-loop run uses only the exported model/bundle and standalone controller, completes the selected 1048 tracking steps or records the failure truthfully; actual trajectory, 50 fps video and report are saved in a new directory.
- Actual DDS loopback run uses the same controller, LowState/LowCmd messages and CRC; report communication count, sample timing, simulator identity, tracking completion and startup/exit status separately. No fabricated result or substitution with training rollout.
- Run regression suite, relevant lint and an independent code review; preserve old artifacts and unrelated files.

## Limits

Passing local sim2sim does not validate hardware calibration, firmware, actuator response, physical emergency stop, thermal limits or robustness on real floors. No hardware enable flag is added. If standing entry or a robustness scenario fails, retain diagnostics and identify that blocker explicitly.

## Builder-check decisions

- Bind to the accepted report: compare actual checkpoint SHA and CSV SHA against RESULTS.json; compare TorchScript state_dict exactly with checkpoint pi and sidecar against export_metadata(checkpoint). Bundle hashing alone is insufficient provenance.
- Physics/LowState 200 Hz (5 ms), policy 50 Hz (20 ms). The local adapter defines LowState.tick as uint32 simulation milliseconds; unwrap rollover, while freshness uses local monotonic reception time. This local tick convention is not a hardware firmware assertion. Reference/history advance once per expected 20ms sample; excessive gaps fail rather than skip history.
- Simulator has its own 100ms LowCmd watchdog that latches damping if command publication stops. Test this with an actual stopped controller process.
- HG LowState.imu_state must be pelvis for this bundle. Perform one initial yaw alignment to reference, then keep it fixed; do not apply torso compensation to a pelvis message. Validate the selected model's IMU sensor attachment explicitly.
- Initialize all ten history frames from the first actual observation. Clip targets to physical joint ranges and feed back the raw action represented by the actual clipped target. Log clipping; do not add a second action smoother.
