"""Coherent-route line-grid band limit and its automatic-resolution refusal (#117).

The incoherent estimator bounds the per-segment retardation increment; the
coherent route sums the complex field first, so its fringes follow the total
span of that quantity. These pin the derivation and the refusal, not the
measured convergence ladder, which is heavy and lives in the validation doc.
"""

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.runner.cxr_grid import resolve_cxr_grid
from pyrite.montecarlo.spectrum.diagnostics import (
    coherent_fringe_spacing,
    sinc_feature_spacing,
)


def _segments(n_electrons=3, per_electron=4, spacing_ang=100.0):
    """Straight collinear tracks along +z, observed along +z."""
    n = n_electrons * per_electron
    elec = np.repeat(np.arange(n_electrons), per_electron)
    step = np.arange(per_electron, dtype=float) * spacing_ang
    z = np.tile(step, n_electrons) + elec * spacing_ang * per_electron
    r_mid = np.zeros((n, 3))
    r_mid[:, 2] = z
    v_hat = np.zeros((n, 3))
    v_hat[:, 2] = 1.0
    energy = np.full(n, 30.0)
    # Segment-start age is path time, not path length: t = z / beta [Ang, c=1].
    # With t = z the retardation scalar d = t - n.r collapses to a constant and
    # the fixture has no fringe at all.
    from pyrite.montecarlo.transport import beta_from_keV

    return {
        "E_keV": energy,
        "E_repr_keV": None,
        "L_ang": np.full(n, spacing_ang),
        "v_hat": v_hat,
        "r_mid": r_mid,
        "t_ang": z / beta_from_keV(energy),
        "elec_id": elec,
    }


def test_coherent_step_follows_the_total_span_not_the_per_segment_increment():
    segs = _segments()
    n_hat = np.array([0.0, 0.0, 1.0])
    step, span, count = coherent_fringe_spacing(segs, n_hat)

    d = (segs["t_ang"] + 0.5 * segs["L_ang"] / _beta(segs)) - segs["r_mid"] @ n_hat
    assert span == pytest.approx(d.max() - d.min())
    assert step == pytest.approx(np.pi * HBARC_EV_ANG / span)
    assert count == segs["elec_id"].size


def test_grouped_span_is_the_widest_single_electron_span():
    segs = _segments()
    n_hat = np.array([0.0, 0.0, 1.0])
    _, span_all, _ = coherent_fringe_spacing(segs, n_hat)
    _, span_grouped, _ = coherent_fringe_spacing(segs, n_hat, grouped=True)
    assert 0.0 < span_grouped < span_all


def test_coherent_step_is_finer_than_the_incoherent_sinc_step():
    """The identity that makes the two estimators different in principle."""
    segs = _segments(n_electrons=4, per_electron=8)
    n_hat = np.array([0.0, 0.0, 1.0])
    incoherent = sinc_feature_spacing(segs, n_hat, aliased_weight_limit=1e-3)[0]
    coherent = coherent_fringe_spacing(segs, n_hat)[0]
    assert coherent < incoherent


def test_per_segment_d_increment_reproduces_the_incoherent_first_zero():
    """`(1 - beta v.n) t_L` is both the sinc width scale and the d increment."""
    segs = _segments()
    n_hat = np.array([0.0, 0.0, 1.0])
    beta = _beta(segs)
    t_L = segs["L_ang"] / beta
    increment = (1.0 - beta * (segs["v_hat"] @ n_hat)) * t_L
    first_zero = 2.0 * np.pi * HBARC_EV_ANG / increment
    assert np.allclose(first_zero, 2.0 * np.pi * HBARC_EV_ANG / increment)
    assert np.all(increment > 0.0)


def test_automatic_resolution_refuses_a_coherent_case():
    segs = _segments()
    case = {"coherent_emission": True, "line_grid_policy": {"resolution": {}}}
    with pytest.raises(ValueError, match="does not support coherent_emission"):
        resolve_cxr_grid(case, segs, np.array([0.0, 0.0, 1.0]), 3, None)


def test_refusal_names_both_derived_steps():
    segs = _segments()
    case = {"coherent_emission": True, "line_grid_policy": {"resolution": {}}}
    with pytest.raises(ValueError) as excinfo:
        resolve_cxr_grid(case, segs, np.array([0.0, 0.0, 1.0]), 3, None)
    message = str(excinfo.value)
    assert "decoherence-grouped floor" in message
    assert "E_grid_line" in message


def test_an_explicit_grid_is_untouched_on_the_coherent_route():
    """Refusal is scoped to automatic resolution, not to coherent runs."""
    grid = np.linspace(10.0, 100.0, 16)
    out, record = resolve_cxr_grid(
        {"coherent_emission": True}, _segments(), np.array([0.0, 0.0, 1.0]), 3, grid
    )
    assert record is None
    assert np.array_equal(out, grid)


def _beta(segs):
    from pyrite.montecarlo.transport import beta_from_keV

    return beta_from_keV(np.asarray(segs["E_keV"], dtype=float))


# --------------------------------------------------------------------------
# reducer geometry per route (#1098): the estimator's d is the reducer's own
# --------------------------------------------------------------------------

_N_HAT_OBLIQUE = np.array([0.6, 0.0, 0.8])
_XY0 = np.array([[0.0, 0.0], [350.0, -120.0], [-900.0, 40.0]])
_T0 = np.array([0.0, 2.5e3, -1.2e3])


def _offset_segments(*, footprint=False):
    """Three electrons launched at sampled face offsets and bunch times."""
    segs = _segments()
    elec = segs["elec_id"]
    segs["r_mid"] = segs["r_mid"].copy()
    segs["r_mid"][:, :2] += _XY0[elec]
    initial_r = np.zeros((3, 3))
    initial_r[:, :2] = _XY0
    segs.update(
        initial_r_ang=initial_r,
        initial_t0_ang=_T0,
        t0_ang=_T0[elec],
        Ne=3,
        thickness_ang=4000.0,
        n_backscattered=0,
        n_missed=0,
        layer=np.zeros(elec.size, dtype=np.int16),
    )
    if footprint:
        segs["crystal_width_ang"] = 1.0e6
        segs["crystal_height_ang"] = 1.0e6
    return segs


def _reducer_d(segs, **request):
    """``d_all_geom`` the coherent reducer phases with, from its own setup phase."""
    from pyrite.montecarlo.spectrum.cxr import SpectrumRequest, _prepare_spectrum
    from tests.helpers import to_host

    st = _prepare_spectrum(
        SpectrumRequest(
            segments=segs,
            E_grid_eV=np.arange(700.0, 1500.0, 1.0),
            crystal="hopg",
            hkl_list=[(0, 0, 2)],
            B_ang2=0.8,
            n_hat=_N_HAT_OBLIQUE,
            coherent=True,
            **request,
        )
    )
    return np.asarray(to_host(st.d_all_geom), dtype=float)


def _sampled_d(segs, n_hat, *, t0=False):
    beta = _beta(segs)
    d = segs["t_ang"] + 0.5 * segs["L_ang"] / beta - segs["r_mid"] @ n_hat
    return d + segs["t0_ang"] if t0 else d


def test_infinite_slab_offsets_are_phased_offset_free():
    segs = _offset_segments()
    _, span, _ = coherent_fringe_spacing(segs, _N_HAT_OBLIQUE, electron_limit=3)
    d = _reducer_d(segs)
    expected = _sampled_d(segs, _N_HAT_OBLIQUE) + _XY0[segs["elec_id"]] @ _N_HAT_OBLIQUE[:2]
    assert np.allclose(d, expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max())
    assert span == pytest.approx(d.max() - d.min(), rel=1e-6)


def test_finite_footprint_span_equals_the_reducers_offset_free_span():
    """Since #370 the analytic finite-footprint branch is offset-free too."""
    segs = _offset_segments(footprint=True)
    # 100 fs makes F_z underflow on this grid, so no transverse spot is required.
    d = _reducer_d(segs, longitudinal_rms_fs=100.0)
    _, span, _ = coherent_fringe_spacing(segs, _N_HAT_OBLIQUE, electron_limit=3)
    sampled = _sampled_d(segs, _N_HAT_OBLIQUE)
    assert span == pytest.approx(d.max() - d.min(), rel=1e-6)
    assert span != pytest.approx(sampled.max() - sampled.min(), rel=1e-3)


def test_physical_electrons_span_includes_t0_and_sampled_offsets():
    segs = _offset_segments()
    d = _reducer_d(segs, physical_electrons=1.0e7)
    _, span, _ = coherent_fringe_spacing(
        segs, _N_HAT_OBLIQUE, electron_limit=3, physical_electrons=1.0e7
    )
    realized = _sampled_d(segs, _N_HAT_OBLIQUE, t0=True)
    assert np.allclose(d, realized, rtol=1e-6, atol=1e-6 * np.abs(realized).max())
    assert span == pytest.approx(d.max() - d.min(), rel=1e-6)
    offset_free_span = coherent_fringe_spacing(segs, _N_HAT_OBLIQUE, electron_limit=3)[1]
    assert span > offset_free_span


def test_bunch_front_gradient_keeps_the_offset_free_geometry_with_physical_electrons():
    """An analytic offset average outranks the realized-field pairing."""
    segs = _offset_segments()
    segs["beam_entry"] = {"bunch_front_gradient": [0.0, 0.0]}
    d = _reducer_d(segs, longitudinal_rms_fs=100.0, physical_electrons=1.0e7)
    _, span, _ = coherent_fringe_spacing(
        segs, _N_HAT_OBLIQUE, electron_limit=3, physical_electrons=1.0e7
    )
    free = coherent_fringe_spacing(segs, _N_HAT_OBLIQUE, electron_limit=3)[1]
    assert span == pytest.approx(d.max() - d.min(), rel=1e-6)
    assert span == pytest.approx(free)


def test_single_segment_has_no_fringe_and_returns_infinity():
    segs = {
        key: (value[:1] if isinstance(value, np.ndarray) else value)
        for key, value in _segments().items()
    }
    step, span, count = coherent_fringe_spacing(segs, np.array([0.0, 0.0, 1.0]))
    assert step == float("inf") and span == 0.0 and count == 1
