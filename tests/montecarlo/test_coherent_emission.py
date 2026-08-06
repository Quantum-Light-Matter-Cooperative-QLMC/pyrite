"""Numerical limiting-case gates for opt-in coherent segment summation."""

import numpy as np
import pytest

from cxr_mc.materials.crystal import (
    CRYSTALS,
    HBARC_EV_ANG,
    beta_from_Ee,
    reciprocal_g_vector,
)
from cxr_mc.montecarlo import mc_spectrum
from cxr_mc.montecarlo._backend import REAL

E_GRID = np.arange(700.0, 1500.0)

KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "n_hat": np.array([1.0, 0.0, 0.01]),
}

RTOL = max(1e-12, 100.0 * float(np.finfo(REAL).eps))
# The batched (n_seg, N_g) path intentionally reassociates a handful of float
# operations relative to the legacy per-hkl reference.  Give that cross-path
# comparison the same order-of-eps latitude as the chunk-invariance gate while
# retaining a peak-scaled absolute floor for bins near an exact zero.
BATCH_RTOL = max(1e-10, 500.0 * float(np.finfo(REAL).eps))
ATOL = 1e-8


def _assert_batch_close(actual, reference):
    peak = float(max(np.max(np.abs(actual)), np.max(np.abs(reference))))
    assert peak > 0.0
    np.testing.assert_allclose(
        actual,
        reference,
        rtol=BATCH_RTOL,
        atol=BATCH_RTOL * 1e-2 * peak,
    )


def _segments(count=1):
    return {
        "r_mid": np.tile([4.0, 0.0, 5.0], (count, 1)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, 30.0),
        "t_ang": np.zeros(count),
        "t0_ang": np.zeros(count),
        "elec_id": np.arange(count),
        "layer": np.zeros(count, dtype=int),
        "Ne": count,
        "thickness_ang": 10.0,
        "crystal_width_ang": 10.0,
        "crystal_height_ang": 10.0,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def test_single_segment_coherent_equals_incoherent_self_term():
    segments = _segments()

    incoherent = mc_spectrum(segments, E_GRID, coherent=False, **KWARGS)
    coherent = mc_spectrum(segments, E_GRID, coherent=True, **KWARGS)
    peak = max(np.max(np.abs(coherent)), np.max(np.abs(incoherent)))
    assert np.max(incoherent) > 0.0
    np.testing.assert_allclose(coherent, incoherent, rtol=RTOL, atol=peak * ATOL)


def test_identical_in_phase_electrons_reach_n_squared_limit():
    single = mc_spectrum(_segments(), E_GRID, coherent=True, **KWARGS)
    pair = mc_spectrum(_segments(2), E_GRID, coherent=True, **KWARGS)

    # Field doubles, intensity quadruples, then per-electron /Ne normalization
    # leaves twice the one-electron yield.
    np.testing.assert_allclose(pair, 2.0 * single, rtol=RTOL)


@pytest.mark.parametrize("sinc_cutoff", [None, 4.0])
def test_straight_trajectory_segments_are_in_phase_at_resonance(sinc_cutoff):
    """The reciprocal-harmonic phase cancels propagation phase on resonance."""
    n_hat = np.asarray(KWARGS["n_hat"], dtype=float)
    n_hat /= np.linalg.norm(n_hat)
    _, g_norm = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    beta = float(beta_from_Ee(30e3))
    k_gamma = beta * g_norm / (1.0 - beta * n_hat[2])
    target_energy = HBARC_EV_ANG * k_gamma
    dz = np.pi / g_norm
    segments = _segments(2)
    segments.update(
        r_mid=np.array([[4.0, 0.0, 5.0], [4.0, 0.0, 5.0 + dz]]),
        L_ang=np.full(2, dz),
        t_ang=np.array([0.0, dz / beta]),
        elec_id=np.zeros(2, dtype=int),
        Ne=1,
        thickness_ang=100.0,
        crystal_width_ang=100.0,
        crystal_height_ang=100.0,
    )
    single = {
        key: value[:1] if isinstance(value, np.ndarray) else value
        for key, value in segments.items()
    }
    single["Ne"] = 1

    energy_grid = target_energy + np.array([-1.0, 0.0, 1.0])
    one_segment = mc_spectrum(single, energy_grid, coherent=True, sinc_cutoff=sinc_cutoff, **KWARGS)
    two_segments = mc_spectrum(
        segments, energy_grid, coherent=True, sinc_cutoff=sinc_cutoff, **KWARGS
    )

    assert one_segment[1] > 0.0
    np.testing.assert_allclose(two_segments[1], 4.0 * one_segment[1], rtol=RTOL)


def test_batched_coherent_matches_legacy_per_hkl_reference():
    """The new batched coherent setup must preserve the legacy per-hkl result.

    ``sinc_cutoff=None`` selects the batched coherent route.  A deliberately
    enormous finite cutoff selects the legacy per-hkl route, but its energy
    window is far wider than this test grid, so no sinc tail is actually
    removed.  The two calls therefore differ only in implementation/reduction
    ordering, not in the modeled physics.
    """
    segments = _segments(2)

    batched = mc_spectrum(segments, E_GRID, coherent=True, sinc_cutoff=None, **KWARGS)
    legacy = mc_spectrum(segments, E_GRID, coherent=True, sinc_cutoff=1.0e6, **KWARGS)

    _assert_batch_close(batched, legacy)


def test_coherent_reflections_remain_incoherent_under_batching():
    """Batching across g must not create cross-reflection field interference."""
    segments = _segments(2)
    energy_grid = np.arange(500.0, 3200.0, 2.0)
    hkls = ((0, 0, 2), (0, 0, 4))

    singles = []
    for hkl in hkls:
        spec = mc_spectrum(
            segments,
            energy_grid,
            coherent=True,
            **{**KWARGS, "hkl_list": [hkl]},
        )
        assert np.max(spec) > 0.0
        singles.append(spec)

    together = mc_spectrum(
        segments,
        energy_grid,
        coherent=True,
        **{**KWARGS, "hkl_list": list(hkls)},
    )

    _assert_batch_close(together, singles[0] + singles[1])


def test_coherent_components_are_rejected():
    with pytest.raises(ValueError, match="incompatible with components"):
        mc_spectrum(_segments(), E_GRID, coherent=True, components=True, **KWARGS)


def _runner_case(*, coherent=False):
    return {
        "crystal": "hopg",
        "hkl_list": KWARGS["hkl_list"],
        "B_ang2": KWARGS["B_ang2"],
        "composition": None,
        "E0_keV": 30.0,
        "coherent_emission": coherent,
    }


def _runner_tp(segs, *, ne_lines=None, ne_brem=None):
    ne = int(segs["Ne"])

    return {
        "E_grid": E_GRID,
        "E_brem": E_GRID,
        "n_hat": KWARGS["n_hat"],
        "segs": segs,
        "Ne_lines": ne if ne_lines is None else ne_lines,
        "Ne_brem": ne if ne_brem is None else ne_brem,
        "groove": None,
    }


def test_runner_always_stores_incoherent_spec_and_omits_spec_coherent(monkeypatch):
    """Default (incoherent) emission: `spec` is the incoherent line sum and no
    `spec_coherent` is attached -- so no downstream `record["spec"]` consumer can
    KeyError and no coherent grid is paid for."""
    import cxr_mc.montecarlo.runner as runner

    monkeypatch.setattr(runner, "_brem_wide_from_segments", lambda *a, **k: np.zeros_like(E_GRID))

    segs = _segments(2)
    tp = _runner_tp(segs, ne_lines=2, ne_brem=2)
    out = runner._spectrum_case_impl(_runner_case(), tp)

    assert "spec_coherent" not in out
    direct_incoherent = mc_spectrum(segs, E_GRID, coherent=False, **KWARGS)
    np.testing.assert_allclose(out["spec"], direct_incoherent, rtol=RTOL)


def test_runner_dual_spectra_from_one_transport(monkeypatch):
    """emission includes coherent: ONE transport (`tp["segs"]`) yields both the
    incoherent `spec` and a `spec_coherent`, each matching a direct
    `mc_spectrum` on the SAME segments -- the single-transport / dual-kernel
    invariant."""
    import cxr_mc.montecarlo.runner as runner

    monkeypatch.setattr(runner, "_brem_wide_from_segments", lambda *a, **k: np.zeros_like(E_GRID))
    segs = _segments(2)
    tp = _runner_tp(segs, ne_lines=2, ne_brem=2)

    case = _runner_case(coherent=True)
    out = runner._spectrum_case_impl(case, tp)

    direct_incoherent = mc_spectrum(segs, E_GRID, coherent=False, **KWARGS)
    direct_coherent = mc_spectrum(segs, E_GRID, coherent=True, **KWARGS)

    np.testing.assert_allclose(out["spec"], direct_incoherent, rtol=RTOL)
    np.testing.assert_allclose(out["spec_coherent"], direct_coherent, rtol=RTOL)
    # The two kernels genuinely differ for this in-phase pair (n^2 build-up),
    # so `spec` is NOT silently the coherent array.
    assert not np.allclose(out["spec"], out["spec_coherent"], rtol=RTOL, atol=0.0)


# ---- store_result: the transport payload must reach the checkpoint record ----
def _store_case():
    return {
        "name": "coh",
        "E0_keV": 30.0,
        "theta_obs_rad": np.pi / 2,
        "dtheta_obs_rad": 0.0,
        "domega_sr": 1.0,
    }


def _store_out(*, coherent=False):
    E = np.arange(50.0, 151.0, 1.0)
    out = {
        "E_grid": E,
        "spec": np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2),
        "brem": np.full_like(E, 0.1),
        "eta": 1.0,
    }
    if coherent:
        out["spec_coherent"] = 2.0 * out["spec"]
    return out


def test_store_result_keeps_spec_coherent_from_transport():
    """Regression: store_result built its record from a hardcoded field list that
    never mentioned `spec_coherent`, so every emission coherent/both run stored
    incoherent-only data no matter what the transport computed -- the analysis UI
    then found Coherent/Both permanently disabled."""
    from cxr_mc.results import store_result

    out = _store_out(coherent=True)
    results = {}
    store_result(results, _store_case(), out)
    record = results["coh"][30.0]

    np.testing.assert_array_equal(record["spec_coherent"], out["spec_coherent"])
    # and the incoherent array is untouched by the companion
    np.testing.assert_array_equal(record["spec"], out["spec"])


def test_store_result_omits_spec_coherent_for_incoherent_run():
    """An incoherent transport grows no key at all, so presence stays the honest
    gate every reader (emission_menu, the altair overlay, reline) tests on."""
    from cxr_mc.results import store_result

    results = {}
    store_result(results, _store_case(), _store_out())

    assert "spec_coherent" not in results["coh"][30.0]
