import numpy as np
import pytest


@pytest.mark.parametrize("mass_scale", [1.0, 1.05])
def test_mass_scaling_updates_cached_subtree_mass_and_com(
    adaptation_bundle, experiment_inputs, mass_scale
):
    from unisupermimic.deploy.mujoco_sim import NativeSimulator

    sim = NativeSimulator(adaptation_bundle, experiment_inputs["scene"], mass_scale=mass_scale)
    sim.reset("reference")
    ids = []
    for body in range(sim.model.nbody):
        ancestor = body
        while ancestor and ancestor != sim.pelvis:
            ancestor = int(sim.model.body_parentid[ancestor])
        if ancestor == sim.pelvis:
            ids.append(body)
    masses = sim.model.body_mass[ids]
    assert sim.model.body_subtreemass[sim.pelvis] == pytest.approx(masses.sum(), abs=1e-12)
    expected = (masses[:, None] * sim.data.xipos[ids]).sum(0) / masses.sum()
    np.testing.assert_allclose(sim.data.subtree_com[sim.pelvis], expected, atol=1e-12, rtol=0)
