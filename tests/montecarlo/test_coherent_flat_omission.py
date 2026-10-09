"""Coherent all-electron term omission on identical trajectories (#362).

Omitting ``|sum_e S_e|^2`` where ``F (N_e - 1) <= limit`` keeps the grouped
floor ``G = sum_e |S_e|^2``; Cauchy-Schwarz bounds the change by ``limit * G``
per row and energy. Every comparison below reuses one segment set, so the
difference is the omission alone, never Monte Carlo noise.

Validation: coherent-flat-term-omission
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite._backend import REAL, xp
from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo import mc_spectrum
from pyrite.montecarlo.case import Case
from pyrite.montecarlo.runner.line_grid import coherent_flat_omission_limit
from pyrite.montecarlo.transport import C_ANG_PER_FS

ENERGY_GRID = np.arange(700.0, 1500.0, 2.0)
KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "n_hat": np.array([1.0, 0.0, 0.1]),
}
T0_ANG = np.array([137.0, -412.0, 55.0, 260.0, -90.0, 333.0])
#: Limit large enough to omit every energy: the spectrum is then the floor.
FLOOR_LIMIT = 1.0e300
#: Agreement of the same term evaluated on a sub-grid versus the full grid.
SUBGRID_RTOL = max(1e-12, 100.0 * float(np.finfo(REAL).eps))


def _segments(t0_ang, *, footprint):
    count = t0_ang.size
    # Distinct emission points and ages keep each S_e distinct.
    r_mid = np.column_stack([4.0 - 0.7 * np.arange(count), np.zeros(count), np.full(count, 5.0)])
    segments = {
        "r_mid": r_mid,
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, 30.0),
        "t_ang": 3.0 * np.arange(count, dtype=float),
        "t0_ang": t0_ang,
        "elec_id": np.arange(count),
        "layer": np.zeros(count, dtype=int),
        "Ne": count,
        "thickness_ang": 10.0,
        "crystal_width_ang": 10.0 if footprint else None,
        "crystal_height_ang": 10.0 if footprint else None,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
        "initial_t0_ang": t0_ang,
        "initial_r_ang": np.zeros((count, 3)),
    }
    return segments


def _groove_kwargs():
    """Grooved escape: keeps the call on the per-(reflection, orientation) route."""
    from pyrite.montecarlo.geometry import tilted_geometry
    from pyrite.montecarlo.groove import blazed_groove_spec

    tilt = np.deg2rad(45.0)
    _, n_hat = tilted_geometry(np.pi / 2, tilt, np.pi)
    return {
        "crystal": "hopg",
        "hkl_list": [(0, 0, 2)],
        "B_ang2": 0.8,
        "n_hat": n_hat,
        "theta_obs_rad": np.pi / 2,
        "groove": blazed_groove_spec(
            spacing_ang=2.0e4,
            theta_obs_rad=np.pi / 2,
            tilt_polar_rad=tilt,
            tilt_azim_rad=np.pi,
        ),
    }


def _empirical_F(t0_ang):
    omega = ENERGY_GRID / HBARC_EV_ANG
    return np.abs(np.mean(np.exp(1j * omega[:, None] * t0_ang[None, :]), axis=1)) ** 2


SCENARIOS = {
    # Infinite slab, empirical characteristic function, batched route.
    "slab-batched": ({"footprint": False}, KWARGS, {}),
    # Infinite slab on the per-hkl route, exact and sinc-windowed.
    "slab-per-hkl": ({"footprint": False}, None, {}),
    "slab-per-hkl-windowed": ({"footprint": False}, None, {"sinc_cutoff": 4.0}),
    # Finite footprint, analytic Gaussian F_z, batched route.
    "footprint-batched": ({"footprint": True}, KWARGS, {"longitudinal_rms_fs": 1.0e-3}),
}


def _run(name, limit, t0_ang=T0_ANG):
    seg_kw, kwargs, extra = SCENARIOS[name]
    kwargs = _groove_kwargs() if kwargs is None else kwargs
    return mc_spectrum(
        _segments(t0_ang, **seg_kw),
        ENERGY_GRID,
        coherent=True,
        coherent_flat_omission_limit=limit,
        **kwargs,
        **extra,
    )


def _expected_F(name):
    _, _, extra = SCENARIOS[name]
    if "longitudinal_rms_fs" in extra:
        sigma_z = extra["longitudinal_rms_fs"] * C_ANG_PER_FS
        return np.exp(-((ENERGY_GRID / HBARC_EV_ANG * sigma_z) ** 2))
    return _empirical_F(T0_ANG)


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("limit", [0.05, 0.5])
def test_omission_stays_inside_the_cauchy_schwarz_bound(name, limit):
    full = _run(name, 0.0)
    floor = _run(name, FLOOR_LIMIT)
    masked = _run(name, limit)

    omitted = _expected_F(name) * (T0_ANG.size - 1) <= limit
    # Non-vacuous: the limit both omits and keeps energies (margin keeps the
    # independent F off the decision edge).
    assert omitted.any() and (~omitted).any()
    peak = float(np.max(np.abs(full)))
    assert peak > 0.0

    # Omitted energies are the floor; kept ones the full blend.
    np.testing.assert_array_equal(masked[omitted], floor[omitted])
    np.testing.assert_allclose(
        masked[~omitted], full[~omitted], rtol=SUBGRID_RTOL, atol=SUBGRID_RTOL * 1e-2 * peak
    )
    # |Total - G| <= F (N_e - 1) G <= limit G per row and energy.
    excess = np.abs(masked - full)
    assert np.all(excess <= limit * floor + SUBGRID_RTOL * peak)
    assert float(np.max(excess)) > 0.0  # the omission really moved something


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_zero_limit_is_the_unchanged_reducer(name):
    seg_kw, kwargs, extra = SCENARIOS[name]
    kwargs = _groove_kwargs() if kwargs is None else kwargs
    default = mc_spectrum(
        _segments(T0_ANG, **seg_kw), ENERGY_GRID, coherent=True, **kwargs, **extra
    )
    np.testing.assert_array_equal(_run(name, 0.0), default)
    # A positive limit that omits nothing routes through the masked blend
    # and still reproduces the full reducer.
    np.testing.assert_allclose(_run(name, 1.0e-300), default, rtol=SUBGRID_RTOL, atol=0.0)


def test_omission_is_inert_without_bunch_offsets():
    segments = _segments(np.zeros(T0_ANG.size), footprint=False)
    segments.pop("initial_t0_ang")
    segments.pop("initial_r_ang")
    full = mc_spectrum(segments, ENERGY_GRID, coherent=True, **KWARGS)
    masked = mc_spectrum(
        segments, ENERGY_GRID, coherent=True, coherent_flat_omission_limit=0.5, **KWARGS
    )
    np.testing.assert_array_equal(masked, full)


@pytest.mark.parametrize("name", ["slab-batched", "footprint-batched"])
def test_single_electron_terms_are_identical(name):
    """``N_e = 1``: ``Flat = G``, so the bound is zero and every energy omits."""
    t0 = np.array([137.0])
    full = _run(name, 0.0, t0)
    masked = _run(name, 1.0e-12, t0)
    floor = _run(name, FLOOR_LIMIT, t0)
    np.testing.assert_array_equal(masked, floor)
    peak = float(np.max(np.abs(full)))
    np.testing.assert_allclose(masked, full, rtol=SUBGRID_RTOL, atol=SUBGRID_RTOL * 1e-2 * peak)


def test_vanishing_form_factor_gives_the_floor():
    """``F -> 0`` (long Gaussian bunch): every energy omits and the spectrum is G."""
    segments = _segments(T0_ANG, footprint=True)
    kw = {**KWARGS, "longitudinal_rms_fs": 200.0}
    masked = mc_spectrum(
        segments, ENERGY_GRID, coherent=True, coherent_flat_omission_limit=1.0e-4, **kw
    )
    floor = mc_spectrum(
        segments, ENERGY_GRID, coherent=True, coherent_flat_omission_limit=FLOOR_LIMIT, **kw
    )
    full = mc_spectrum(segments, ENERGY_GRID, coherent=True, **kw)
    np.testing.assert_array_equal(masked, floor)
    np.testing.assert_allclose(masked, full, rtol=SUBGRID_RTOL)


@pytest.mark.parametrize("limit", [-1.0, float("nan"), float("inf")])
def test_invalid_limit_is_refused(limit):
    with pytest.raises(ValueError, match="coherent_flat_omission_limit"):
        mc_spectrum(
            _segments(T0_ANG, footprint=False),
            ENERGY_GRID,
            coherent=True,
            coherent_flat_omission_limit=limit,
            **KWARGS,
        )


def test_runner_limit_defaults_to_policy_and_honours_the_case():
    from pyrite._line_grid_policy import DEFAULT_COHERENT_FLAT_OMISSION_LIMIT

    assert coherent_flat_omission_limit({}) == DEFAULT_COHERENT_FLAT_OMISSION_LIMIT
    assert coherent_flat_omission_limit({"coherent_flat_omission_limit": 0}) == 0.0


def test_case_validates_the_limit():
    from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases

    sweep = Sweep(material="mose2", beam=BeamSpec(energy_keV=30.0), tilt_deg=5.0)
    legacy = build_cases(sweep, n_electrons=2, n_electrons_brem=1)[0]
    legacy = legacy.to_dict() if isinstance(legacy, Case) else dict(legacy)
    assert "coherent_flat_omission_limit" not in Case(**legacy)
    case = Case(**legacy, coherent_flat_omission_limit=0.0)
    assert case["coherent_flat_omission_limit"] == 0.0
    assert list(case)[-1] == "coherent_flat_omission_limit"
    with pytest.raises(ValueError, match="coherent_flat_omission_limit"):
        Case(**legacy, coherent_flat_omission_limit=-1.0)


def test_mask_keeps_uncertified_factors_and_rounds_the_bound_outward():
    from pyrite.montecarlo.spectrum.lines._per_hkl import _flat_energy_keep

    st = SimpleNamespace(
        Ne=3,
        finite_footprint_now=False,
        request=SimpleNamespace(coherent_flat_omission_limit=0.5),
    )
    factors = np.array([np.nan, np.inf, -0.1, 1.1, 0.25, 0.0, 0.1])
    np.testing.assert_array_equal(_flat_energy_keep(st, factors), [0, 1, 2, 3, 4])
    second_row = factors.copy()
    second_row[-1] = 0.6
    np.testing.assert_array_equal(
        _flat_energy_keep(st, np.stack([factors, second_row])), [0, 1, 2, 3, 4, 6]
    )


def test_gaussian_mask_refuses_a_rounded_factor_below_the_true_threshold():
    from pyrite.montecarlo.spectrum.coherent_form_factor import gaussian_form_factor_bounds
    from pyrite.montecarlo.spectrum.lines._per_hkl import _flat_energy_keep

    energy = np.array([800.0, 900.0])
    rms = 1e-3
    _, upper = gaussian_form_factor_bounds(
        energy[0],
        energy[0],
        sigma_z_ang=rms * C_ANG_PER_FS,
    )
    # A falsely small supplied F must not license omission against the
    # separately certified analytic law at the first coordinate.
    st = SimpleNamespace(
        Ne=2,
        finite_footprint_now=True,
        E_grid=energy,
        request=SimpleNamespace(
            coherent_flat_omission_limit=upper / 2,
            longitudinal_rms_fs=rms,
        ),
    )
    assert 0 in _flat_energy_keep(st, np.zeros(2))


def test_coherent_capture_refuses_active_omission():
    from pyrite.montecarlo.runner.coherent_audit import CoherentGridAudit

    audit = CoherentGridAudit({"coherent_emission": True, "_coherent_yield_audit": {}})
    st = SimpleNamespace(
        decoherence_active=True,
        request=SimpleNamespace(coherent_flat_omission_limit=1e-4),
    )
    with pytest.raises(ValueError, match="certifies the full inter-electron blend"):
        audit.capture(st, None, None, None, None)


@pytest.fixture(params=["auto", "eager", "jit"] if xp.__name__ == "cupy" else ["auto"])
def coherent_route(request, monkeypatch):
    """Exercise each CUDA reducer; CPU has one implementation per geometry."""
    from pyrite.montecarlo.spectrum.lines import _policy

    if request.param != "auto":
        monkeypatch.setattr(_policy, "_USE_JIT_COHERENT_STREAM", False)
        monkeypatch.setattr(_policy, "_USE_JIT_COHERENT_REDUCTION", request.param == "jit")
    return request.param


@pytest.mark.parametrize("population", [1.5, 12.0, 120.0])
@pytest.mark.parametrize("route", ["batched", "per-hkl", "per-hkl-windowed"])
def test_physical_population_bound_on_identical_finite_footprint_fields(
    population, route, coherent_route
):
    """N differs from M=6: the switch and error budget must use physical N.

    Identical fields give P=M G, attaining the Cauchy-Schwarz upper bound.
    The Gaussian crosses F=0.05 on this axis, testing kept and omitted bins.
    """
    segments = _segments(T0_ANG, footprint=True)
    segments["r_mid"][:] = [0.0, 0.0, 5.0]
    segments["t_ang"][:] = 0.0
    kwargs = KWARGS if route == "batched" else _groove_kwargs()
    extra = {"sinc_cutoff": 4.0} if route == "per-hkl-windowed" else {}
    common = dict(
        coherent=True,
        physical_electrons=population,
        longitudinal_rms_fs=1e-3,
        **kwargs,
        **extra,
    )
    limit = 0.05 * (population - 1)
    full = mc_spectrum(segments, ENERGY_GRID, **common)
    disabled = mc_spectrum(segments, ENERGY_GRID, coherent_flat_omission_limit=0, **common)
    floor = mc_spectrum(segments, ENERGY_GRID, coherent_flat_omission_limit=FLOOR_LIMIT, **common)
    masked = mc_spectrum(segments, ENERGY_GRID, coherent_flat_omission_limit=limit, **common)
    np.testing.assert_array_equal(disabled, full)
    factor = np.exp(-((ENERGY_GRID / HBARC_EV_ANG * 1e-3 * C_ANG_PER_FS) ** 2))
    omitted = factor < 0.05
    assert omitted.any() and (~omitted).any()
    peak = float(np.max(full))
    assert peak > 0
    # Aligned-field reference uses N, independently of reducer helpers.
    np.testing.assert_allclose(
        full,
        (1 + factor * (population - 1)) * floor,
        rtol=SUBGRID_RTOL,
        atol=SUBGRID_RTOL * peak,
    )
    np.testing.assert_array_equal(masked[omitted], floor[omitted])
    np.testing.assert_allclose(masked[~omitted], full[~omitted], rtol=SUBGRID_RTOL)
    assert np.all(np.abs(full - masked) <= limit * floor + SUBGRID_RTOL * peak)


@pytest.mark.parametrize("population", [0.0, 1.0])
def test_zero_pair_population_omits_even_without_offsets(population, coherent_route):
    segments = _segments(np.zeros(6), footprint=False)
    segments.pop("initial_t0_ang")
    segments.pop("initial_r_ang")
    common = dict(coherent=True, physical_electrons=population, **KWARGS)
    full = mc_spectrum(segments, ENERGY_GRID, **common)
    masked = mc_spectrum(segments, ENERGY_GRID, coherent_flat_omission_limit=1e-4, **common)
    np.testing.assert_allclose(masked, full, rtol=SUBGRID_RTOL)


@pytest.mark.parametrize("route", ["batched", "per-hkl"])
def test_physical_slab_uses_complete_fields_not_empirical_factor(route, coherent_route):
    # The empirical factor would allow omission at many coordinates, but
    # physical slabs use F=1. N=2 keeps the signed estimator nonnegative.
    segments = _segments(T0_ANG, footprint=False)
    kwargs = KWARGS if route == "batched" else _groove_kwargs()
    common = dict(coherent=True, physical_electrons=2.0, **kwargs)
    full = mc_spectrum(segments, ENERGY_GRID, **common)
    masked = mc_spectrum(segments, ENERGY_GRID, coherent_flat_omission_limit=0.05, **common)
    assert (_empirical_F(T0_ANG) < 0.05).any()
    np.testing.assert_allclose(masked, full, rtol=SUBGRID_RTOL)


def test_physical_pair_bound_cannot_use_the_smaller_sample_count():
    from pyrite.montecarlo.spectrum.lines._per_hkl import _flat_energy_keep

    st = SimpleNamespace(
        Ne=3,
        finite_footprint_now=False,
        request=SimpleNamespace(coherent_flat_omission_limit=0.5, physical_electrons=101.0),
    )
    # F=0.01 would pass the old M-1 bound, but exceeds the physical budget.
    np.testing.assert_array_equal(_flat_energy_keep(st, np.array([0.01, 0.001])), [0])


def test_omission_cannot_hide_unresolved_physical_pair_power():
    from pyrite.montecarlo.spectrum.coherent_population import CoherentSamplingError

    segments = _segments(np.zeros(2), footprint=False)
    segments["r_mid"][:] = [0.0, 0.0, 5.0]
    segments["t_ang"][:] = [0.0, 5.0]
    # Physical slab F=1 and N-1=11 exceed the small omission budget.
    # Preserve #350's refusal of a statistically unresolved signed estimate.
    with pytest.raises(CoherentSamplingError, match="negative or nonfinite pair estimate"):
        mc_spectrum(
            segments,
            ENERGY_GRID,
            coherent=True,
            physical_electrons=12.0,
            coherent_flat_omission_limit=0.05,
            **KWARGS,
        )
