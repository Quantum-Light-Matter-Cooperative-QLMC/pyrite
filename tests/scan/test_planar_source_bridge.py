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
from pyrite.units import ureg

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
        spec,
        setup=_setup() if setup is None else setup,
        impact=impact,
        **options,
    )


def test_projected_entry_is_identical_to_existing_solver_reference():
    spec = pr.TransverseStateSpec.recipe("gaussian", waist_ang=0.8, dimension=1)
    projected = _project(spec, impact="fixed")
    profile, cached = zone_solver(
        "silicon", (1, 1, 0), 293.15, ENERGY_EV, 4, 2, projected.margin_eV
    )
    solver = DensityMatrixSolver(profile, ENERGY_EV, cached.kernel, margin_eV=projected.margin_eV)
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
    profile, cached = zone_solver(
        "silicon", (1, 1, 0), 293.15, ENERGY_EV, 4, 2, projection.margin_eV
    )
    reference_solver = DensityMatrixSolver(
        profile, ENERGY_EV, cached.kernel, margin_eV=projection.margin_eV
    )
    expected = reference_solver.propagate(40.0, 20.0, initial=projection.initial)
    np.testing.assert_array_equal(actual.final_density, expected.final_density)
    np.testing.assert_array_equal(actual.trace, expected.trace)
    assert projection.total_population == pytest.approx(1.0, abs=2e-3)


def test_projection_configuration_accepts_quantities_at_the_diagnostic_boundary():
    spec = pr.TransverseStateSpec.recipe("gaussian", waist_ang=0.8, dimension=1)
    projected = _project(
        spec,
        kinetic_energy_eV=17 * ureg.MeV,
        incidence_rad=0.1 * ureg.degree,
        temperature_K=ureg.Quantity(20, "degC"),
        margin_eV=0.1 * ureg.keV,
    )
    assert projected.kinetic_energy_eV == ENERGY_EV
    assert projected.temperature_K == pytest.approx(293.15)
    assert projected.incidence_rad == pytest.approx(np.deg2rad(0.1))
    assert projected.margin_eV == 100.0


def test_projection_rejects_wrong_dimension_before_solver_work():
    spec = pr.TransverseStateSpec.recipe("gaussian", waist_ang=0.8, dimension=1)
    with pytest.raises(ValueError, match="project_planar_source.kinetic_energy_eV.*energy.*eV"):
        _project(spec, kinetic_energy_eV=1 * ureg.mm)


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
        _project(
            pr.TransverseStateSpec.recipe(
                "aperture_probe", kinetic_energy_eV=30_000.0, semiangle_rad=0.01
            )
        )
    with pytest.raises(ValueError, match="impact"):
        _project(spec, impact="silently-reinterpret")
    with pytest.raises(ValueError, match="incidence_rad"):
        _project(spec, incidence_rad=float("nan"))
    capped = replace(
        _setup(), region=pr.QuantumRegion(kind="planar", plane_hkl=(1, 1, 0), zone_max_ang=20.0)
    )
    projection = _project(spec, setup=capped)
    with pytest.raises(ValueError, match="zone_max_ang"):
        propagate_planar_source(projection, depth_ang=21.0)
    with pytest.raises(ValueError, match="nonnegative"):
        propagate_planar_source(projection, depth_ang=10.0, stopping_eV_per_ang=-1)
    # Explicit production injection remains refused until handoff is validated.
    beam = pr.Beam(
        energy_keV=17000.0,
        transverse=pr.beams.Spot(),
        longitudinal=pr.beams.PointBunch(),
        state=spec,
    )
    with pytest.raises(ValueError, match="not yet.*transport"):
        beam.to_spec()


def test_scoring_matches_existing_emitter_from_the_same_projected_state():
    from pyrite.montecarlo.channeling import channeling_spectrum
    from pyrite.planar_source import _solver_for, score_planar_source

    low = pr.TransverseStateSpec.recipe("hermite_gauss", m=0, waist_ang=1.0)
    high = pr.TransverseStateSpec.recipe("hermite_gauss", m=1, waist_ang=1.0)
    for state, impact in (
        (low, "averaged"),
        (pr.TransverseStateSpec.superpose([(1.0, low), (1.0j, high)]), "fixed"),
        (pr.TransverseStateSpec.mixture([(1.0, low), (2.0, high)]), "fixed"),
    ):
        proj = _project(state, impact=impact)
        grid = np.linspace(5e3, 60e3, 25)
        new = score_planar_source(
            proj,
            depth_ang=80.0,
            photon_energy_eV=grid,
            theta_rad=0.0,
            absorption=False,
            sample_ang=40.0,
        )
        legacy = channeling_spectrum(
            _solver_for(proj),
            80.0,
            grid,
            0.0,
            incidence_rad=proj.incidence_rad,
            entry_state=state.to_runtime(),
            entry_impact=impact,
            dz_ang=proj.setup.numerics.step_ang,
            sample_ang=40.0,
            absorption=False,
        )
        np.testing.assert_allclose(new.spectrum, legacy.spectrum, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(new.line_rates, legacy.line_rates, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(
            new.bound_population, legacy.bound_population, rtol=1e-11, atol=1e-13
        )
        assert new.spectrum.shape == grid.shape
        assert np.all(np.isfinite(new.spectrum))


def test_preprojected_cr_input_validates_population_and_exclusivity():
    from pyrite.montecarlo.channeling import channeling_spectrum
    from pyrite.planar_source import _solver_for, score_planar_source

    source = pr.TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
    proj = _project(source)
    solver = _solver_for(proj)
    grid = np.linspace(5e3, 60e3, 15)
    kwargs = dict(dz_ang=20.0, sample_ang=20.0, absorption=False)
    with pytest.raises(ValueError, match="entry_initial conflicts"):
        channeling_spectrum(
            solver,
            20.0,
            grid,
            0.0,
            entry_initial=proj.initial,
            entry_state=source.to_runtime(),
            **kwargs,
        )
    with pytest.raises(ValueError, match="entry_initial conflicts"):
        channeling_spectrum(
            solver, 20.0, grid, 0.0, entry_initial=proj.initial, incidence_rad=0.001, **kwargs
        )
    with pytest.raises(ValueError, match="normalized"):
        channeling_spectrum(
            solver, 20.0, grid, 0.0, entry_initial=(proj.initial[0], -1.0), **kwargs
        )
    with pytest.raises(ValueError, match="shape"):
        channeling_spectrum(solver, 20.0, grid, 0.0, entry_initial=(np.ones((2, 2)), 0.0), **kwargs)
    with pytest.raises(ValueError, match="Hermitian"):
        rho = proj.initial[0]
        rho[0, 0, 1] += 0.25j
        channeling_spectrum(
            solver, 20.0, grid, 0.0, entry_initial=(rho, proj.entry_population), **kwargs
        )
    with pytest.raises(ValueError, match="positive"):
        score_planar_source(proj, depth_ang=20.0, photon_energy_eV=[5e3, 5e3], theta_rad=0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        score_planar_source(
            proj, depth_ang=20.0, photon_energy_eV=grid, theta_rad=0.0, aperture_rad=-0.1
        )
    with pytest.raises(ValueError, match="zone_max_ang"):
        capped = replace(
            proj,
            setup=replace(
                proj.setup,
                region=pr.QuantumRegion(kind="planar", plane_hkl=(1, 1, 0), zone_max_ang=10.0),
            ),
        )
        score_planar_source(capped, depth_ang=20.0, photon_energy_eV=grid, theta_rad=0.0)


def test_two_dimensional_source_requires_explicit_in_plane_angular_model():
    from pyrite.planar_source import score_planar_source

    state = pr.TransverseStateSpec.recipe("laguerre_gauss", ell=1, waist_ang=2.0)
    proj = _project(state)
    assert proj.source_dimension == 2
    with pytest.raises(ValueError, match="2-D state requires explicit"):
        score_planar_source(
            proj,
            depth_ang=20.0,
            photon_energy_eV=np.linspace(5e3, 60e3, 12),
            theta_rad=0.0,
            absorption=False,
        )


@pytest.mark.parametrize("unit", ["delta_degC", "delta_degF"])
def test_projection_refuses_temperature_difference_as_absolute_temperature(unit):
    spec = pr.TransverseStateSpec.recipe("gaussian", waist_ang=0.8, dimension=1)
    with pytest.raises(ValueError, match="project_planar_source.temperature_K.*difference"):
        _project(spec, temperature_K=ureg.Quantity(20, unit))
    with pytest.raises(ValueError, match="project_planar_source.temperature_K.*difference"):
        _setup().project_state(
            spec,
            crystal="silicon",
            kinetic_energy_eV=ENERGY_EV,
            incidence_rad=0.0,
            temperature_K=ureg.Quantity(20, unit),
        )
