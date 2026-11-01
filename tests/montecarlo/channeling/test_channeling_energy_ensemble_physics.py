"""Small physical-solver cross-checks for energy ensembles (issue #626)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xraydb")  # Required by the existing material-scattering stack.

from pyrite.materials.channeling import planar_profile, transverse_states  # noqa: E402
from pyrite.montecarlo.channeling import (  # noqa: E402
    IncidentEnergyEnsemble,
    planar_energy_diagnostic,
    planar_solver_factory,
)


def test_energy_specific_eigenstates_scattering_and_weighted_physical_density():
    """Independent transverse eigenvalues, kernel and full propagator reference."""
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    e1, e2 = 17e6, 20e6
    factory = planar_solver_factory(profile, n_cells=2, max_order=5)
    left, right = factory(e1), factory(e2)
    assert right.gamma0 > left.gamma0
    assert right.kernel.hbar_v_eV_ang > left.kernel.hbar_v_eV_ang
    for solver in (left, right):
        # Eigenvalues recomputed at the exact Bloch k grid used by this solver.
        states = transverse_states(
            profile,
            solver.gamma0,
            solver.kappa[0],
            n_plane_waves=5,
        )
        np.testing.assert_allclose(solver.basis.energies[0], states.energies_eV, atol=1e-11)
    assert np.max(np.abs(left.basis.energies - right.basis.energies)) > 1e-4
    assert not np.array_equal(left.kernel.momentum_kernel, right.kernel.momentum_kernel)

    coords = np.linspace(0.0, profile.period_ang, 17)
    kw = dict(spatial_x_ang=coords, record_every=1, divergence_rad=0.001)
    direct = [factory(e).propagate(100.0, 100.0, **kw) for e in (e1, e2)]
    ensemble = IncidentEnergyEnsemble(e1, energy_nodes_eV=(e1, e2), node_weights=(1, 3))
    averaged = planar_energy_diagnostic(ensemble, factory, 100.0, 100.0, **kw)
    for attr in (
        "trace",
        "bound_population",
        "free_flux",
        "overflow_flux",
        "spatial_density_per_ang",
    ):
        expected = 0.25 * getattr(direct[0], attr) + 0.75 * getattr(direct[1], attr)
        np.testing.assert_allclose(getattr(averaged, attr), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        averaged.balance_defect,
        0.25 * direct[0].balance_defect + 0.75 * direct[1].balance_defect,
        atol=1e-12,
    )
    n_bands = min(result.band_populations.shape[-1] for result in direct)
    np.testing.assert_allclose(
        averaged.band_populations,
        0.25 * direct[0].band_populations[..., :n_bands]
        + 0.75 * direct[1].band_populations[..., :n_bands],
        rtol=1e-12,
        atol=1e-12,
    )
