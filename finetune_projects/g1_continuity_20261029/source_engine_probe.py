"""Read-only closed-loop probe on the original MJWarp model and actuator path.

Default recording uses pre-reset 50Hz transitions. The optional pinned-backend
observer also records every 5ms graph replay; its sampling equivalence is checked
separately. Passing this probe alone never establishes complete acceptance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.run_sim2sim import write_json
from unisupermimic.finetune.continuity_metrics import summarize_continuity
from unisupermimic.finetune.learner import project_raw_action
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import make_eval_env
from unisupermimic.tasks.motion_tracking.g1.supermimic.render import flattened_qpos


@torch.inference_mode()
def run(bundle_path, output, physics_record=False, motion_csv=None):
    bundle = DeploymentBundle.load(bundle_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    csv = Path(
        motion_csv
        if motion_csv is not None
        else "finetune_projects/g1_continuous_20261029/data/sequence_v2/sequence.csv"
    ).resolve()
    env = make_eval_env(
        Path("runs/demo_kick_smooth_001"), "mjwarp", motion_files=[str(csv)], device="cuda"
    )
    try:
        mdp = env.supermimic_mdp
        for key in ("q", "qd", "root_p", "root_R", "root_v", "root_w"):
            np.testing.assert_allclose(
                getattr(mdp.motion, key).cpu().numpy(), bundle.reference[key], rtol=0, atol=2e-5
            )
        device = env.device
        policy = torch.jit.load(str(bundle.policy_path), map_location=device).eval()
        scales = torch.tensor(bundle.metadata["action_scales"], device=device)
        lower = torch.tensor(bundle.metadata["joint_lower"], device=device)
        upper = torch.tensor(bundle.metadata["joint_upper"], device=device)
        _, reset = env.reset_to_frames(mdp.motion.clip_start.clone())
        current = reset[SUPERMIMIC_STATE_KEY]
        physics = []
        original_steps = env.backend._execute_device_steps
        model = mujoco.MjModel.from_xml_path(
            "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"
        )
        joints = np.array([model.joint(name).id for name in bundle.metadata["joint_names"]])
        qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]

        def snapshot():
            # Pinned-backend diagnostic only: read raw device fields, never
            # sensordata/tracked bodies (those can trigger a forward pass).
            env.backend._warp.synchronize()
            views = env.backend.get_state_views(("qpos", "qvel", "time"))
            raw_q = views["qpos"][0].cpu().numpy().copy()
            raw_v = views["qvel"][0].cpu().numpy().copy()
            raw_time = float(views["time"].flatten()[0].cpu())
            physics.append((np.r_[raw_q[:7], raw_q[qadr]], raw_v[vadr], raw_v[:3], raw_time))

        def observed_steps(count):
            # Preserve the original step_tensor control upload, force lifetime,
            # time/cache updates and graph replay. Only read between replays.
            for _ in range(count):
                original_steps(1)
                snapshot()

        if physics_record:
            assert abs(env.backend._sim_dt - 0.005) < 1e-9
            snapshot()
            env.backend._execute_device_steps = observed_steps
        states = [current["s"][0].cpu()]
        frames = [0]
        history = None
        terminated = done = False
        n = len(bundle.reference["q"])
        for _ in range(n - 1):
            obs = mdp.obs(
                unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"]
            )
            history = (
                obs[:, None].repeat(1, 10, 1)
                if history is None
                else torch.cat((history[:, 1:], obs[:, None]), 1)
            )
            action = project_raw_action(
                policy(history, mdp.goal(current["f"], 10)), mdp.default_q, scales, lower, upper
            )
            state = env.step(action)
            transition = state.info[SUPERMIMIC_TRANSITION_KEY]
            states.append(transition["s1"][0].cpu())
            frames.append(int(transition["f"][0]) + 1)
            done = bool(transition["done"][0])
            terminated = bool(transition["term"][0])
            if done:
                break
            current = state.info[SUPERMIMIC_STATE_KEY]
        qpos = flattened_qpos(torch.stack(states)).numpy()
        frame_ids = np.asarray(frames)
        np.savez_compressed(output / "trajectory.npz", qpos=qpos, reference_frame=frame_ids)
        report = dict(
            backend="mjwarp",
            model="src/unisupermimic/assets/robots/g1/scene_supermimic.xml",
            bundle=str(Path(bundle_path).resolve()),
            motion_csv=str(csv),
            recorded_hz=50,
            recorded_samples=len(qpos),
            expected_samples=n,
            last_reference_frame=int(frame_ids[-1]),
            terminated=terminated,
            done=done,
            physical_complete=bool(not terminated and len(qpos) == n and frame_ids[-1] == n - 1),
            joint_rmse_rad=float(
                np.sqrt(np.mean((qpos[:, 7:] - bundle.reference["q"][frame_ids]) ** 2))
            ),
            continuity_200hz_evaluated=False,
            goal_complete_claimed=False,
            hardware_ready=False,
        )
        write_json(output / "report.json", report)
        if physics_record:
            arrays = dict(
                sample_time=np.arange(len(physics)) * 0.005,
                qpos=np.asarray([p[0] for p in physics]),
                joint_velocity=np.asarray([p[1] for p in physics]),
                root_velocity=np.asarray([p[2] for p in physics]),
                device_time=np.asarray([p[3] for p in physics]),
            )
            np.savez_compressed(output / "physics_trajectory.npz", **arrays)
            continuity = summarize_continuity(
                bundle.metadata,
                arrays["qpos"],
                arrays["joint_velocity"],
                arrays["root_velocity"],
                arrays["sample_time"],
            )
            report.update(
                continuity=continuity,
                recorded_physics_samples=len(physics),
                physics_clock="step_index * verified backend sim_dt=0.005; raw device_time retained",
                continuity_200hz_evaluated=True,
                max_device_clock_drift=float(
                    np.abs(arrays["device_time"] - arrays["sample_time"]).max()
                ),
            )
            write_json(output / "report.json", report)
        print({key: value for key, value in report.items() if key != "continuity"}, flush=True)
        return report
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics-record", action="store_true")
    parser.add_argument("--motion-csv", type=Path, help="Use the matching CSV for a retimed bundle")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.bundle, args.output, args.physics_record, args.motion_csv)
