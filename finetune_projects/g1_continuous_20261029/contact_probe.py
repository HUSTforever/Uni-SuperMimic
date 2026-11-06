"""One-off solver-contact probe; not a policy observation or stability certificate."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.deploy.bundle import DeploymentBundle
from unisupermimic.deploy.controller import PolicyController
from unisupermimic.deploy.mujoco_sim import NativeSimulator

torch.set_num_threads(1)
b = DeploymentBundle.load(Path(sys.argv[1]))
scene = Path("deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
sim = NativeSimulator(b, scene)
sim.reset("reference_zero_velocity")
pi = PolicyController(b)
m, d = sim.model, sim.data
feet = {m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")}
sole = {
    i
    for i in range(m.ngeom)
    if m.geom_bodyid[i] in feet
    and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
    and m.geom_contype[i]
}
floor = m.geom("floor").id
rows = []
failed = None
for frame in range(len(b.reference["q"]) - 1):
    contacts = []
    for contact_id in range(d.ncon):
        c = d.contact[contact_id]
        pair = {int(c.geom1), int(c.geom2)}
        if floor not in pair or not (pair & sole):
            continue
        geom = next(iter(pair & sole))
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, contact_id, force)
        if force[0] <= 1:
            continue
        spatial = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, geom, spatial, 0)
        velocity = spatial[3:] + np.cross(spatial[:3], c.pos - d.geom_xpos[geom])
        contacts.append((float(force[0]), float(np.dot(velocity[:2], velocity[:2]))))
    rows.append(
        dict(
            time=d.time,
            normal_force=sum(x[0] for x in contacts),
            weighted_speed2=sum(x[0] * x[1] for x in contacts),
            contacts=len(contacts),
        )
    )
    sim.advance(pi.step(sim.sensor(), frame))
    failed = sim.failure(frame + 1)
    if failed:
        break
selected = [r for r in rows if 5 - 1e-6 <= r["time"] <= 27.96 + 1e-6]
forces = sum(r["normal_force"] for r in selected)
result = dict(
    bundle=str(b.directory),
    physics_hz=200,
    sample_hz=50,
    criterion="floor/sole contact normal force >1N at completed policy boundaries; solver diagnostics only",
    completed=len(rows) == 1798 and failed is None,
    failure=failed,
    force_weighted_contact_speed_rms_m_s=float(
        np.sqrt(sum(r["weighted_speed2"] for r in selected) / forces)
    )
    if forces
    else None,
    no_loaded_sole_contact_fraction=sum(r["contacts"] == 0 for r in selected) / len(selected),
    mean_normal_force_N=forces / len(selected),
    samples=rows,
)
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
print({k: v for k, v in result.items() if k != "samples"})
