"""Real-physics smoke/regression check for #627 (requires atomic database)."""

import numpy as np
import pytest

pytest.importorskip("xraydb")

from pyrite.materials.channeling.axial import axial_profile  # noqa: E402
from pyrite.montecarlo.channeling import (  # noqa: E402
    AxialEnergyMCWFOptions,
    IncidentEnergyEnsemble,
    axial_energy_mcwf,
)
from pyrite.montecarlo.channeling.axial_diagnostic import AxialDiagnosticConfig  # noqa: E402
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator  # noqa: E402
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF  # noqa: E402
from pyrite.montecarlo.channeling.mcwf import mcwf_stream_key  # noqa: E402
from pyrite.montecarlo.channeling.states import gaussian  # noqa: E402


def test_each_mono_reference_has_energy_dependent_dissipator_and_matching_stream():
    energies = (8e6, 12e6)
    beam = IncidentEnergyEnsemble(10e6, energy_nodes_eV=energies, node_weights=(1.0, 1.0))
    cfg = AxialDiagnosticConfig(
        mode="mcwf",
        kinetic_energy_eV=10e6,
        mechanisms=("tds",),
        shape=(12, 12),
        depths_ang=(0.0, 2.0),
        step_ang=1.0,
    )
    result = axial_energy_mcwf(
        beam,
        cfg,
        options=AxialEnergyMCWFOptions(n_electrons=2, seed=627),
        electron_energies_eV=energies,
    )
    profile = axial_profile(cfg.crystal, cfg.axis, cfg.temperature_K)
    assert result.energies_eV.tolist() == list(energies)
    assert result.bin_counts.tolist() == [1, 1]
    assert result.density_per_ang2.shape == (2, 12, 12)
    diss = [
        axial_dissipator(profile, e, cfg.cells, cfg.shape, mechanisms=cfg.mechanisms)
        for e in energies
    ]
    assert diss[0].hbar_v_eV_ang < diss[1].hbar_v_eV_ang
    for eid, e in enumerate(energies):
        run = AxialMCWF(diss[eid], e, profile.on_grid(cfg.cells, cfg.shape)).run(
            gaussian(waist_ang=0.5),
            cfg.depths_ang,
            1,
            seed=int(mcwf_stream_key(627 ^ 0x6272, eid)),
            step_ang=cfg.step_ang,
            tilt_rad=cfg.tilt_rad,
        )
        np.testing.assert_allclose(result.surviving_by_bin[eid], run.surviving, atol=0)
        np.testing.assert_allclose(
            result.density_by_bin_per_ang2[eid], run.observables, rtol=1e-12, atol=1e-13
        )
