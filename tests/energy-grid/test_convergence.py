"""Refinement-ladder harness: nesting, Richardson acceptance, segment identity.

The analytic spectra here are sums of the line kernels' ``sinc^2`` feature,
whose exact integral ``pi / a`` and exact trapezoid threshold ``h <= pi / a``
(issue #109) are independent of the harness, so the accepted spacing is checked
against sampling theory rather than against the harness's own arithmetic.
"""

from functools import partial

import numpy as np
import pytest
from scipy.special import ndtri

from pyrite.energy_grid.convergence import (
    GATED_OBSERVABLES,
    Rung,
    SegmentMismatchError,
    SpectrumSample,
    evaluate_ladder,
    nested_uniform_ladder,
    representative_spacing,
    require_identical_segments,
    richardson_acceptance,
    segment_fingerprint,
    spectrum_observables,
)

SOURCE_ONLY = {name: kind for name, kind in GATED_OBSERVABLES.items() if kind == "intrinsic_source"}
SOURCE_AND_SHAPE = {
    name: kind for name, kind in GATED_OBSERVABLES.items() if kind in {"intrinsic_source", "shape"}
}
NO_DETECTORS = partial(spectrum_observables, detectors={})
YIELD = {"yield": "intrinsic_source"}


def _rung(spacing, **observables):
    return Rung(
        spacing_eV=spacing,
        n_points=2,
        start_eV=0.0,
        stop_eV=1.0,
        observables=observables,
        informational={},
        wall_s=0.0,
        host_peak_rss_mib=0.0,
        device_peak_mib=None,
    )


def _sinc_squared_lines(a_per_eV, centres_eV, amplitudes):
    """Line density built only from the kernel lineshape, and its exact integral."""

    def density(E):
        E = np.asarray(E, dtype=float)
        total = np.zeros_like(E)
        for centre, amplitude in zip(centres_eV, amplitudes, strict=True):
            total += amplitude * np.sinc(a_per_eV * (E - centre) / np.pi) ** 2
        return total

    return density, float(np.sum(amplitudes) * np.pi / a_per_eV)


def test_nested_uniform_ladder_halves_exactly_and_nests():
    grids = nested_uniform_ladder(50.0, 9100.0, 12.0, 4)

    spacings = [representative_spacing(grid) for grid in grids]
    assert spacings[0] <= 12.0
    assert spacings[1:] == pytest.approx([spacings[0] / 2**k for k in (1, 2, 3)], rel=1e-12)
    for coarse, fine in zip(grids, grids[1:], strict=False):
        assert np.array_equal(fine[::2], coarse)


def test_accepted_spacing_resolves_the_sinc_nyquist_step():
    a = 1.0  # first-zero step pi/a ~= 3.14 eV
    # Gaussian-quantile resonance energies: a smooth multiple-scattering-like
    # envelope (sigma = 40 eV) built from features far narrower than it, with no
    # sampling ripple for the dominant-peak finder to jump between.
    centres = 1000.0 + 40.0 * ndtri((np.arange(2000) + 0.5) / 2000)
    density, exact_yield = _sinc_squared_lines(a, centres, np.full(centres.size, 0.2))
    background = lambda E: np.full_like(E, 1.0e-3)  # noqa: E731
    grids = nested_uniform_ladder(0.0, 2000.0, 12.0, 6)

    rungs = evaluate_ladder(
        grids,
        lambda E: SpectrumSample(density(E), background(E)),
        observables=NO_DETECTORS,
    )
    report = richardson_acceptance(rungs, gated=SOURCE_AND_SHAPE)

    nyquist = np.pi / a
    for rung in rungs:
        if rung.spacing_eV <= nyquist:
            # Band-limited integrand below Nyquist: only the window-truncated
            # 1/(a^2 D) tail mass separates the trapezoid from pi/a.
            assert rung.observables["yield"] == pytest.approx(exact_yield, rel=2e-3)
    # Above Nyquist the integrated observables of many phase-spread features can
    # stay inside tolerance (aliases average out by Poisson summation), so the
    # accepted spacing is not bounded by pi/a; what the gate guarantees is that
    # the accepted rung meets the tolerance against the independent integral,
    # and that the grossly aliased 12 eV triple is rejected.
    assert report.triples[0].accepted is False
    assert report.accepted_spacing_eV is not None
    accepted = next(r for r in rungs if r.spacing_eV == report.accepted_spacing_eV)
    assert accepted.observables["yield"] == pytest.approx(exact_yield, rel=2e-3)


def test_one_lucky_pair_is_not_convergence():
    rungs = [_rung(h, **{"yield": q}) for h, q in ((4.0, 1.0), (2.0, 1.0), (1.0, 1.01))]

    report = richardson_acceptance(rungs, gated=YIELD)

    (verdict,) = report.triples[0].observables
    assert verdict.accepted is False
    assert verdict.reason == "h/2->h/4 larger"
    assert report.accepted_spacing_eV is None


def test_two_tiny_changes_accept_even_when_the_second_is_slightly_larger():
    # tol = 1e-3 * 1.000025; both changes (1.0e-5, 1.5e-5) sit below 0.1 * tol.
    rungs = [_rung(h, **{"yield": q}) for h, q in ((4.0, 1.0), (2.0, 1.00001), (1.0, 1.000025))]

    report = richardson_acceptance(rungs, gated=YIELD)

    (verdict,) = report.triples[0].observables
    assert verdict.accepted is True
    assert verdict.reason == "converged below noise fraction"
    assert report.accepted_spacing_eV == 4.0


def test_a_larger_second_change_near_the_tolerance_is_rejected():
    # d1 = 8e-4 < tol ~ 1.0017e-3, but d2 = 9e-4 > d1 and far above 0.1 * tol.
    rungs = [_rung(h, **{"yield": q}) for h, q in ((4.0, 1.0), (2.0, 1.0008), (1.0, 1.0017))]

    report = richardson_acceptance(rungs, gated=YIELD)

    (verdict,) = report.triples[0].observables
    assert verdict.accepted is False
    assert verdict.reason == "h/2->h/4 larger"


def test_equal_changes_and_an_exactly_constant_zero_observable_are_accepted():
    equal = [_rung(h, **{"yield": q}) for h, q in ((4.0, 1.0), (2.0, 1.0005), (1.0, 1.0))]
    constant = [_rung(h, **{"yield": 1.0, "centroid_eV": 0.0}) for h in (4.0, 2.0, 1.0)]

    assert richardson_acceptance(equal, gated=YIELD).accepted_spacing_eV == 4.0
    report = richardson_acceptance(
        constant, gated={"yield": "intrinsic_source", "centroid_eV": "intrinsic_source"}
    )
    assert report.accepted_spacing_eV == 4.0


def test_shape_is_gated_at_one_percent_and_reported_at_the_source_tolerance():
    # FWHM moves 0.05 then 0.005 eV on ~10 eV: the first change is inside the
    # 1e-2 shape gate but outside 1e-3, so the diagnostic accepts only from 4 eV.
    rungs = [
        _rung(h, **{"yield": 1.0, "centroid_eV": 500.0, "fwhm_eV": f, "line_background_ratio": 2.0})
        for h, f in ((8.0, 10.10), (4.0, 10.05), (2.0, 10.045), (1.0, 10.045))
    ]

    report = richardson_acceptance(rungs, gated=SOURCE_AND_SHAPE)

    assert report.rtol == {"intrinsic_source": 1e-3, "shape": 1e-2}
    assert report.accepted_spacing_eV == 8.0
    assert report.observable_accepted_spacing_eV["fwhm_eV"] == 8.0
    assert report.diagnostic_accepted_spacing_eV["fwhm_eV"] == 4.0
    assert report.diagnostic_accepted_spacing_eV["line_background_ratio"] == 8.0


def test_harness_shape_class_leaves_the_production_policy_tolerances_unchanged():
    from pyrite._line_grid_policy import DEFAULT_RTOL, OBSERVABLE_CLASSES

    assert dict(DEFAULT_RTOL) == {"intrinsic_source": 1e-3, "detected_counts": 1e-2}
    assert OBSERVABLE_CLASSES == ("intrinsic_source", "detected_counts")
    assert set(GATED_OBSERVABLES.values()) == {"intrinsic_source", "shape", "detected_counts"}


def test_monotone_shrinking_changes_are_accepted_per_observable_class():
    rungs = [
        _rung(4.0, **{"yield": 1.0009, "timepix3_counts": 1.009}),
        _rung(2.0, **{"yield": 1.0, "timepix3_counts": 1.0}),
        _rung(1.0, **{"yield": 1.0, "timepix3_counts": 1.0}),
    ]
    gated = {"yield": "intrinsic_source", "timepix3_counts": "detected_counts"}

    assert richardson_acceptance(rungs, gated=gated).accepted_spacing_eV == 4.0
    # The same 0.9% counts change fails once counts are held to the source tolerance.
    strict = richardson_acceptance(rungs, gated=gated, rtol={"detected_counts": 1e-3})
    assert strict.accepted_spacing_eV is None


def test_accepted_spacing_requires_every_finer_triple_to_pass():
    values = [1.0, 1.0, 1.0, 1.1, 1.0]  # coarse triple converged, finer ones not
    rungs = [_rung(2.0**-k, **{"yield": v}) for k, v in enumerate(values)]

    report = richardson_acceptance(rungs, gated=YIELD)

    assert report.triples[0].accepted is True
    assert report.accepted_spacing_eV is None


def test_empty_spectrum_is_accepted_through_the_absolute_floor():
    grids = nested_uniform_ladder(10.0, 110.0, 10.0, 3)
    rungs = evaluate_ladder(
        grids,
        lambda E: SpectrumSample(np.zeros_like(E), np.zeros_like(E)),
        observables=NO_DETECTORS,
    )

    report = richardson_acceptance(rungs, gated=SOURCE_ONLY)

    assert np.isnan(rungs[-1].observables["centroid_eV"])
    assert {verdict.reason for verdict in report.triples[0].observables} == {"near-zero"}
    assert report.accepted_spacing_eV == rungs[0].spacing_eV


def test_undefined_shape_observable_fails_closed_for_a_real_spectrum():
    rungs = [
        _rung(4.0, **{"yield": 1.0, "fwhm_eV": float("nan")}),
        _rung(2.0, **{"yield": 1.0, "fwhm_eV": 3.0}),
        _rung(1.0, **{"yield": 1.0, "fwhm_eV": 3.0}),
    ]

    report = richardson_acceptance(
        rungs, gated={"yield": "intrinsic_source", "fwhm_eV": "intrinsic_source"}
    )

    assert report.triples[0].observables[1].reason == "undefined"
    assert report.accepted_spacing_eV is None


def test_ladder_accepts_nonuniform_coordinates_and_labels_the_widest_step():
    grids = [np.geomspace(100.0, 5000.0, n) for n in (200, 400, 800)]
    rungs = evaluate_ladder(
        grids,
        lambda E: SpectrumSample(np.exp(-(((E - 2000.0) / 150.0) ** 2)), np.full_like(E, 1e-2)),
        observables=NO_DETECTORS,
    )

    assert [rung.spacing_eV for rung in rungs] == [float(np.diff(g).max()) for g in grids]
    assert rungs[-1].observables["fwhm_eV"] == pytest.approx(
        2.0 * 150.0 * np.sqrt(np.log(2.0)), rel=2e-3
    )


def test_ladder_refuses_grids_that_do_not_refine():
    grids = nested_uniform_ladder(0.0, 100.0, 1.0, 2)[::-1]
    with pytest.raises(ValueError, match="refine strictly"):
        evaluate_ladder(
            grids,
            lambda E: SpectrumSample(np.ones_like(E), np.ones_like(E)),
            observables=NO_DETECTORS,
        )


def _segments(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "E_keV": rng.uniform(10.0, 30.0, n),
        "L_ang": rng.uniform(10.0, 100.0, n),
        "v_hat": rng.standard_normal((n, 3)),
        "r_mid": rng.standard_normal((n, 3)),
        "elec_id": np.arange(n),
    }


def test_evaluator_that_changes_segments_is_refused():
    segments = _segments()

    def mutating(E):
        segments["L_ang"][0] *= 1.5
        return SpectrumSample(np.ones_like(E), np.ones_like(E))

    with pytest.raises(SegmentMismatchError, match="different segment kinematics"):
        evaluate_ladder(
            nested_uniform_ladder(0.0, 10.0, 1.0, 2),
            mutating,
            segments=segments,
            observables=NO_DETECTORS,
        )


def test_fixed_seed_resume_requires_identical_segment_count_and_kinematics():
    reference = segment_fingerprint(_segments())

    require_identical_segments(reference, _segments())
    with pytest.raises(SegmentMismatchError, match="n_segments 6 != 5"):
        require_identical_segments(reference, _segments(n=6))
    with pytest.raises(SegmentMismatchError, match="different segment kinematics"):
        require_identical_segments(reference, _segments(seed=1))


def test_detected_counts_are_finite_through_both_default_detectors():
    E = np.linspace(1000.0, 3000.0, 401)
    line = np.exp(-(((E - 2000.0) / 50.0) ** 2))

    observables = spectrum_observables(E, line, np.full_like(E, 1e-3))

    for name in ("timepix3_counts", "eaglexo_counts"):
        assert np.isfinite(observables[name])
        assert 0.0 < observables[name] <= observables["yield"]
