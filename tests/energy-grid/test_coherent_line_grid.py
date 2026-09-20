"""Coherent-route line-grid band limit and its automatic-resolution refusal (#117).

The incoherent estimator bounds the per-segment retardation increment; the
coherent route sums the complex field first, so its fringes follow the total
span of that quantity. These pin the derivation and the refusal, not the
measured convergence ladder, which is heavy and lives in the validation doc.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.runner.line_grid import resolve_line_grid
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
        resolve_line_grid(case, segs, np.array([0.0, 0.0, 1.0]), 3, None)


def test_refusal_names_both_derived_steps():
    segs = _segments()
    case = {"coherent_emission": True, "line_grid_policy": {"resolution": {}}}
    with pytest.raises(ValueError) as excinfo:
        resolve_line_grid(case, segs, np.array([0.0, 0.0, 1.0]), 3, None)
    message = str(excinfo.value)
    assert "decoherence-grouped floor" in message
    assert "E_grid_line" in message


def test_an_explicit_grid_is_untouched_on_the_coherent_route():
    """Refusal is scoped to automatic resolution, not to coherent runs."""
    grid = np.linspace(10.0, 100.0, 16)
    out, record = resolve_line_grid(
        {"coherent_emission": True}, _segments(), np.array([0.0, 0.0, 1.0]), 3, grid
    )
    assert record is None
    assert np.array_equal(out, grid)


def _beta(segs):
    from pyrite.montecarlo.transport import beta_from_keV

    return beta_from_keV(np.asarray(segs["E_keV"], dtype=float))
