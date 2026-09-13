"""Direct gates on mc_spectrum's phases, driven without calling mc_spectrum.

``mc_spectrum`` used to be one 1,892-line function whose internals were closures
over its locals, so nothing inside it could be reached -- or asserted on --
independently. These tests exercise each phase through its own entry point:
preparation, both accumulation routes, the route choice, and the final
reduction. They are the standing guard that the phases stay separable.
"""

import numpy as np
import pytest

from pyrite._backend import REAL
from pyrite.materials.crystal import (
    CRYSTALS,
    HBARC_EV_ANG,
    reciprocal_g_vector,
    refractive_index,
)
from pyrite.montecarlo import mc_spectrum
from pyrite.montecarlo.spectrum.lines import (
    SpectrumRequest,
    _accumulate_batched,
    _accumulate_per_hkl,
    _batched_block,
    _batched_tables,
    _finalize_spectrum,
    _needs_per_hkl_route,
    _prepare_spectrum,
)
from pyrite.montecarlo.transport import beta_from_keV
from tests.helpers import host_backend_only, scaled_rtol, to_device, to_host

# ``_prepare_spectrum`` fails closed on an accelerator once a flight carries
# numerical substeps: the grouped segmented complex reduction has no device port
# yet. Claims about that route say so rather than assert a behaviour the
# selected backend cannot reach.
_grouped_is_host_only = host_backend_only(
    "flight-grouped incoherent CXR: the segmented complex reduction over "
    "numerical substeps has no device port"
)

CRYSTAL = "hopg"
HKL = (0, 0, 2)
B_ANG2 = 0.8
E_KEV = 30.0
# Near-transverse observation: keeps the resonance inside the grid below and the
# in-medium correction to the denominator small but nonzero.
N_HAT = np.array([1.0, 0.0, -0.05]) / np.linalg.norm([1.0, 0.0, -0.05])
E_GRID = np.arange(700.0, 1500.0, 1.0)

# The batched (n_seg, N_g) route deliberately reassociates a handful of float
# reductions relative to the per-hkl reference; give the cross-route comparison
# the same order-of-eps latitude the chunk-invariance gate already uses.
BATCH_RTOL = max(1e-10, 500.0 * float(np.finfo(REAL).eps))


def _segments(count=4, *, n_elec=2, flights=False, footprint=False, thickness=4000.0):
    """Straight +z trajectories at fixed energy: an analytic resonance per row."""
    elec_id = np.repeat(np.arange(n_elec, dtype=np.int64), count // n_elec)
    segs = {
        "r_mid": np.column_stack(
            [np.zeros(count), np.zeros(count), np.linspace(200.0, thickness - 200.0, count)]
        ),
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 20.0),
        "E_keV": np.full(count, E_KEV),
        "t_ang": np.zeros(count),
        "t0_ang": np.zeros(count),
        "elec_id": elec_id,
        "layer": np.zeros(count, dtype=np.int16),
        "Ne": n_elec,
        "thickness_ang": thickness,
        "n_backscattered": 0,
        "n_missed": 0,
    }
    if flights:
        # two numerical substeps per physical flight -- the case whose rows must
        # add coherently rather than as independent emitters
        segs["flight_id"] = np.repeat(np.arange(count // 2, dtype=np.int64), 2)
        segs["substep_id"] = np.tile([0, 1], count // 2)
    if footprint:
        segs["crystal_width_ang"] = 6.0e4
        segs["crystal_height_ang"] = 6.0e4
    return segs


def _request(**over):
    kw = {
        "segments": _segments(),
        "E_grid_eV": E_GRID,
        "crystal": CRYSTAL,
        "hkl_list": [HKL],
        "B_ang2": B_ANG2,
        "n_hat": N_HAT,
    }
    kw.update(over)
    return SpectrumRequest(**kw)


def _resonance_eV():
    """E_res from the in-medium fixed point, solved here from the materials layer.

    Independent of ``spectrum.lines``: this is the model statement
    ``omega_res = v.g / (1 - Re n(E_res) v.n_hat)`` iterated to convergence, not
    a call into the code under test.

    At this near-transverse geometry the in-medium shift is 0.007 eV, well below
    the grid step, so the peak-position assertions gate the resonance relation
    and not the dispersion refinement -- that is `xray-in-medium-resonance`'s
    own row, exercised by tests/montecarlo/test_xray_dispersion.py.
    """
    g_vec, _ = reciprocal_g_vector(HKL, CRYSTALS[CRYSTAL]["lattice"])
    v = beta_from_keV(E_KEV) * np.array([0.0, 0.0, 1.0])
    v_dot_g = float(v @ g_vec)
    v_dot_n = float(v @ N_HAT)
    E = HBARC_EV_ANG * v_dot_g / (1.0 - v_dot_n)  # vacuum seed
    for _ in range(64):
        n_re = float(np.asarray(refractive_index(CRYSTAL, np.array([E]), True).real)[0])
        E = HBARC_EV_ANG * v_dot_g / (1.0 - n_re * v_dot_n)
    return E


# --------------------------------------------------------------------------
# phase 1: preparation
# --------------------------------------------------------------------------


def test_prepare_stages_zeroed_buffers_on_the_requested_grid():
    st = _prepare_spectrum(_request())

    for buf in (st.spec, st.spec_pxr, st.spec_cbs):
        assert buf.shape == (E_GRID.size,)
        assert not buf.any(), "accumulation must start from zero"
    assert st.Ne == 2
    assert st.v_all.shape == (4, 3)
    # tabulation window pads the output grid by 20% on both sides
    assert st.E_tab[0] <= E_GRID[0] and st.E_tab[-1] >= E_GRID[-1]


def test_prepare_resolves_the_observation_direction_from_the_polar_angle():
    theta = np.deg2rad(119.0)
    st = _prepare_spectrum(_request(n_hat=None, theta_obs_rad=theta))

    assert np.linalg.norm(st.n_hat) == pytest.approx(1.0)
    assert st.n_hat[2] == pytest.approx(np.cos(theta))
    assert st.n_hat[0] == pytest.approx(np.sin(theta))


@_grouped_is_host_only
def test_prepare_groups_numerical_substeps_but_not_whole_flights():
    """Substeps of one flight are integration detail, not independent emitters.

    Summing |A Q|^2 over them would divide the line peak by the substep count,
    so the grouped reduction must engage exactly when a flight was subdivided.
    """
    assert _prepare_spectrum(_request()).grouped is False
    assert _prepare_spectrum(_request(segments=_segments(flights=True))).grouped is True
    # one row per flight is algebraically the incoherent path: stay off it
    singleton = _segments()
    singleton["flight_id"] = np.arange(singleton["L_ang"].size, dtype=np.int64)
    assert _prepare_spectrum(_request(segments=singleton)).grouped is False


def test_prepare_leaves_coherent_precompute_absent_on_the_incoherent_path():
    st = _prepare_spectrum(_request())
    assert st.d_all is None and st.omega_grid is None and st.delta_omega_grid is None

    st_coh = _prepare_spectrum(_request(coherent=True))
    assert st_coh.omega_grid is not None
    # omega = E / hbar c, per row of the output grid. The precompute is staged
    # on the selected backend at REAL, so the single division it performs is one
    # rounding of that precision away from the fp64 oracle.
    np.testing.assert_allclose(
        to_host(st_coh.omega_grid),
        E_GRID / HBARC_EV_ANG,
        rtol=scaled_rtol(1e-12, eps_multiple=2.0),
        atol=0.0,
    )
    assert st_coh.decoherence_active is False  # no sampled offsets in this case


def test_prepare_detects_a_finite_crystal_footprint():
    assert _prepare_spectrum(_request()).finite_footprint is False
    assert _prepare_spectrum(_request(segments=_segments(footprint=True))).finite_footprint


def test_prepare_rejects_mutually_exclusive_and_missing_inputs():
    with pytest.raises(ValueError, match="B_ang2"):
        _prepare_spectrum(_request(B_ang2=None))
    with pytest.raises(ValueError, match="incompatible with components"):
        _prepare_spectrum(_request(coherent=True, components=True))
    with pytest.raises(NotImplementedError, match="LAYERED absorber"):
        _prepare_spectrum(_request(coherent=True, layers=[(0.0, 4000.0, [("C", 0.1136)])]))


def test_prepare_refuses_sinc_windowing_on_a_nonuniform_grid():
    """sinc_cutoff turns an energy half-width into a node index by dividing by
    ``E_grid[1] - E_grid[0]``, so a graded grid would silently window the wrong
    nodes. Issue #98: refuse explicitly; the unwindowed routes evaluate at the
    nodes themselves and stay allowed."""
    from pyrite.energy_grid.semantics import NonuniformEnergyGridError

    log_grid = np.logspace(np.log10(700.0), np.log10(1500.0), E_GRID.size)

    with pytest.raises(NonuniformEnergyGridError, match="sinc_cutoff"):
        _prepare_spectrum(_request(E_grid_eV=log_grid, sinc_cutoff=3.0))

    # unwindowed: allowed, because nothing reads a single spacing
    assert _prepare_spectrum(_request(E_grid_eV=log_grid)).E_grid.size == log_grid.size


# --------------------------------------------------------------------------
# route choice
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over, expected",
    [
        ({}, False),
        ({"segments": _segments(footprint=True)}, False),
        ({"layers": [(0.0, 4000.0, [("C", 0.1136)])]}, True),
        ({"coherent": True, "sinc_cutoff": 3.0}, True),
        pytest.param({"segments": _segments(flights=True)}, True, marks=_grouped_is_host_only),
        ({"coherent": True}, False),
        ({"sinc_cutoff": 3.0}, False),
    ],
)
def test_route_choice_keeps_uncovered_configurations_on_the_per_hkl_loop(over, expected):
    assert _needs_per_hkl_route(_prepare_spectrum(_request(**over))) is expected


# --------------------------------------------------------------------------
# phase 2: accumulation, both routes
# --------------------------------------------------------------------------


def test_batched_route_puts_the_line_at_the_analytic_resonance():
    st = _prepare_spectrum(_request())
    assert not st.spec.any()

    _accumulate_batched(st)

    assert st.spec.max() > 0.0
    peak_eV = float(E_GRID[int(np.argmax(st.spec))])
    # one grid step of slack: the peak bin, not a sub-bin interpolation
    assert peak_eV == pytest.approx(_resonance_eV(), abs=float(E_GRID[1] - E_GRID[0]))


def test_per_hkl_route_puts_the_line_at_the_same_resonance():
    st = _prepare_spectrum(_request(layers=[(0.0, 4000.0, [("C", 0.1136)])]))
    _accumulate_per_hkl(st)

    assert st.spec.max() > 0.0
    peak_eV = float(E_GRID[int(np.argmax(st.spec))])
    assert peak_eV == pytest.approx(_resonance_eV(), abs=float(E_GRID[1] - E_GRID[0]))


def test_per_hkl_route_records_the_escape_distance_for_a_finite_footprint():
    """The escape distance is g-independent, so the route hoists it per case."""
    st = _prepare_spectrum(
        _request(segments=_segments(footprint=True), sinc_cutoff=3.0, coherent=True)
    )
    assert st.L_esc_all is None

    _accumulate_per_hkl(st)

    assert st.L_esc_all is not None
    assert st.L_esc_all.shape == (4,)
    assert np.all(st.L_esc_all > 0.0)


def test_the_two_routes_agree_on_a_case_both_can_run():
    """The batched path reassociates float reductions but must not move physics."""
    batched = _prepare_spectrum(_request())
    _accumulate_batched(batched)

    # grouping is what forces the per-hkl route here; the segment set is the same
    # physical trajectory either way, so drive the compatibility route directly
    per_hkl = _prepare_spectrum(_request())
    _accumulate_per_hkl(per_hkl)

    batched_spec = to_host(batched.spec)
    per_hkl_spec = to_host(per_hkl.spec)
    peak = float(max(batched_spec.max(), per_hkl_spec.max()))
    assert peak > 0.0
    np.testing.assert_allclose(
        batched_spec, per_hkl_spec, rtol=BATCH_RTOL, atol=BATCH_RTOL * 1e-2 * peak
    )


def test_batched_route_supports_non_henke_form_factors():
    """The non-anomalous form-factor option reaches the batched route."""
    batched = mc_spectrum(
        _segments(), E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, n_hat=N_HAT, use_henke=False
    )
    per_hkl = mc_spectrum(
        _segments(),
        E_GRID,
        CRYSTAL,
        [HKL],
        B_ang2=B_ANG2,
        n_hat=N_HAT,
        use_henke=False,
        sinc_cutoff=1.0e6,
    )

    peak = float(max(batched.max(), per_hkl.max()))
    assert peak > 0.0
    np.testing.assert_allclose(batched, per_hkl, rtol=BATCH_RTOL, atol=BATCH_RTOL * 1e-2 * peak)


def test_batched_block_masks_lines_outside_the_padded_window():
    st = _prepare_spectrum(_request())
    bt = _batched_tables(st)
    blk = _batched_block(st, bt, slice(0, 4))

    assert blk.E_res.shape == (4, bt.N_g)
    assert blk.keep.shape == blk.E_res.shape
    assert blk.keep.all(), "this case's only line sits inside the grid"

    # a grid far below the resonance keeps nothing
    off = _prepare_spectrum(_request(E_grid_eV=np.arange(50.0, 90.0, 1.0)))
    off_bt = _batched_tables(off)
    assert not _batched_block(off, off_bt, slice(0, 4)).keep.any()


# --------------------------------------------------------------------------
# phase 3: reduction
# --------------------------------------------------------------------------


def test_finalize_normalises_per_incident_electron():
    st = _prepare_spectrum(_request())
    st.spec[:] = to_device(np.arange(E_GRID.size), st.spec.dtype)

    out = _finalize_spectrum(st)

    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.arange(E_GRID.size) / st.Ne, rtol=1e-12, atol=0.0)


def test_finalize_returns_the_component_split_normalised_alike():
    st = _prepare_spectrum(_request(components=True))
    st.spec[:] = 6.0
    st.spec_pxr[:] = 4.0
    st.spec_cbs[:] = 2.0

    total, pxr, cbs = _finalize_spectrum(st)

    assert st.Ne == 2
    np.testing.assert_allclose(total, 3.0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(pxr, 2.0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(cbs, 1.0, rtol=1e-12, atol=0.0)


def test_the_phases_compose_into_the_documented_pipeline():
    """prepare -> accumulate -> finalise is the whole contract of mc_spectrum."""
    st = _prepare_spectrum(_request())
    assert not _needs_per_hkl_route(st)
    _accumulate_batched(st)
    out = _finalize_spectrum(st)

    assert out.shape == (E_GRID.size,)
    assert np.all(np.isfinite(out))
    assert out.max() > 0.0
