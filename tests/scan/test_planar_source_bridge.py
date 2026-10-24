"""Quantum source -> Bloch density -> propagation, without Case/MC handoff (#680)."""

import json
from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite.montecarlo.channeling import states
from pyrite.montecarlo.channeling.density import DensityMatrixSolver
from pyrite.montecarlo.channeling.handoff import zone_solver
from pyrite.planar_source import project_planar_source, propagate_planar_source

ENERGY_EV = 17e6


def _setup(**numerics):
    return pr.QuantumSetup(
        region=pr.QuantumRegion(kind="planar", plane_hkl=(1, 1, 0)),
        dynamics=pr.QuantumDynamics(kind="channeling"),
        numerics=pr.QuantumNumerics(basis_size=4, bloch_cells=2, step_ang=20.0, **numerics),
    )


def _project(spec, *, impact="averaged", setup=None, **kwargs):
    options = dict(crystal="silicon", kinetic_energy_eV=ENERGY_EV, incidence_rad=0.0)
    options.update(kwargs)
    return project_planar_source(
        spec, setup=_setup() if setup is None else setup, impact=impact, **options,
    )


def test_projected_entry_is_identical_to_existing_solver_reference():
    spec = pr.TransverseStateSpec.recipe("gaussian", waist_ang=0.8, dimension=1)
    projected = _project(spec, impact="fixed")
    profile, cached = zone_solver("silicon", (1, 1, 0), 293.15, ENERGY_EV, 4, 2,
                                  projected.margin_eV)
    solver = DensityMatrixSolver(profile, ENERGY_EV, cached.kernel,
                                 margin_eV=projected.margin_eV)
    plane, outside = states.planar_entry_density(
        solver, spec.to_runtime(), incidence_rad=0.0, impact="fixed"
    )
    rho, free, outside = solver.entry_from_plane(plane, outside)
    np.testing.assert_array_equal(projected.plane_density, plane)
    np.testing.assert_array_equal(projected.subspace_density, rho)
    np.testing.assert_array_equal(projected.free_population, free)
    assert projected.outside_population == outside
    assert projected.total_population == pytest.approx(1.0, abs=2e-3)
    assert projected.state_digest == spec.digest

    # Returning a fresh mutable initial state must not mutate the stored source.
    with pytest.raises(ValueError):
        projected.subspace_density[...] = 0
    writable, entry = projected.initial
    assert writable.flags.writeable
    writable[...] = 0
    assert np.any(projected.subspace_density != 0)
    assert entry == pytest.approx(float(free.sum()) + outside)


def test_diagnostic_propagation_uses_exact_projected_initial():
    spec = pr.TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
    setup = _setup()
    projection = setup.project_state(spec, crystal="silicon", kinetic_energy_eV=ENERGY_EV)
    actual = propagate_planar_source(projection, depth_ang=40.0)
    profile, cached = zone_solver("silicon", (1, 1, 0), 293.15, ENERGY_EV, 4, 2,
                                  projection.margin_eV)
    reference_solver = DensityMatrixSolver(profile, ENERGY_EV, cached.kernel,
                                           margin_eV=projection.margin_eV)
    expected = reference_solver.propagate(40.0, 20.0, initial=projection.initial)
    np.testing.assert_array_equal(actual.final_density, expected.final_density)
    np.testing.assert_array_equal(actual.trace, expected.trace)
    assert projection.total_population == pytest.approx(1.0, abs=2e-3)


def test_coherent_vs_mixed_specs_keep_distinct_phase_identity():
    plus = pr.TransverseStateSpec.recipe("hermite_gauss", m=0, waist_ang=1.0)
    minus = pr.TransverseStateSpec.recipe("hermite_gauss", m=1, waist_ang=1.0)
    coherent = pr.TransverseStateSpec.superpose([(1.0, plus), (1.0j, minus)])
    mixed = pr.TransverseStateSpec.mixture([(1.0, plus), (1.0, minus)])
    assert coherent.digest != mixed.digest
    c = _project(coherent, impact="fixed")
    m = _project(mixed, impact="fixed")
    assert c.total_population == pytest.approx(1.0, abs=2e-3)
    assert m.total_population == pytest.approx(1.0, abs=2e-3)
    assert not np.allclose(c.plane_density, m.plane_density)
    assert json.loads(json.dumps(coherent.to_dict()))["kind"] == "superposition"


def test_averaged_impact_forgets_translation():
    base = pr.TransverseStateSpec.recipe("gaussian", waist_ang=0.8, dimension=1)
    shifted = base.translated((0.4,))
    a = _project(base)
    b = _project(shifted)
    np.testing.assert_allclose(a.plane_density, b.plane_density, atol=1e-8)
    # At fixed impact, the shift is physically observable.
    a_fixed = _project(base, impact="fixed")
    b_fixed = _project(shifted, impact="fixed")
    assert not np.allclose(a_fixed.plane_density, b_fixed.plane_density)


def test_refusal_before_case_creation_or_propagation():
    spec = pr.TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
    with pytest.raises(ValueError, match="planar source diagnostics"):
        _project(spec, setup=pr.QuantumSetup())
    # Match #624's early axial admission, even when diagnostics are requested.
    axial = pr.QuantumSetup(region=pr.QuantumRegion(kind="axial", axis_uvw=(0, 0, 1)))
    with pytest.raises(ValueError, match="#623"):
        _project(spec, setup=axial)
    with pytest.raises(ValueError, match="calibrated"):
        _project(pr.TransverseStateSpec.recipe(
            "aperture_probe", kinetic_energy_eV=30_000.0, semiangle_rad=0.01
        ))
    with pytest.raises(ValueError, match="impact"):
        _project(spec, impact="silently-reinterpret")
    with pytest.raises(ValueError, match="incidence_rad"):
        _project(spec, incidence_rad=float("nan"))
    capped = replace(_setup(), region=pr.QuantumRegion(
        kind="planar", plane_hkl=(1, 1, 0), zone_max_ang=20.0))
    projection = _project(spec, setup=capped)
    with pytest.raises(ValueError, match="zone_max_ang"):
        propagate_planar_source(projection, depth_ang=21.0)
    with pytest.raises(ValueError, match="nonnegative"):
        propagate_planar_source(projection, depth_ang=10.0, stopping_eV_per_ang=-1)
    # Explicit production injection remains refused until handoff is validated.
    beam = pr.Beam(energy_keV=17000.0, transverse=pr.Spot(),
                   longitudinal=pr.PointBunch(), state=spec)
    with pytest.raises(ValueError, match="not yet.*transport"):
        beam.to_spec()
