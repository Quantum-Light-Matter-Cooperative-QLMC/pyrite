"""Adaptive electron counts on full-fiber HOPG CXR cases (#1503).

The stopping monitor reads the roll- and mosaic-averaged per-electron line
mass that :func:`fiber_lines` folds (#1489), so an adaptive fiber case is the
fixed-N fiber case at its realized count.

Validation: adaptive-sample-size-stopping
Validation: cxr-fiber-intensity-average
"""

import warnings

import numpy as np

import pyrite as pr
from pyrite import api
from pyrite._backend import _to_cpu
from pyrite._cxr_grid_policy import CxrYieldStatisticsWarning
from pyrite._sampling import adaptive_sampling
from pyrite.config import CxrReflections, FiberAverage
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import fiber_emission
from pyrite.montecarlo.runner.adaptive import AdaptiveSettings, case_measure
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed, streaming_refusal
from pyrite.montecarlo.spectrum.cxr._bin_fourier import FOURIER_RESIDUAL_TOL

FIELDS = ("spec", "spec_characteristic", "brem", "brem_wide", "E_grid", "E_grid_brem")


def _scene():
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "hopg",
            thickness_ang=5000.0,
            tilt_deg=15.0,
            footprint=pr.geometry.Footprint(5.0, 5.0),
            mosaic=True,
        ),
        pr.Detector(),
        physics=pr.Physics(bremsstrahlung_model="bremslib", radiative_model="uncoupled"),
    )


def _precision(**extra):
    # CXR starts checking at 30, brem at 20: brem freezes first, so later
    # chunks score lines only and Ne_brem < Ne.
    return pr.config.AdaptiveSampling(
        target_rse=1.0,
        min_histories=30,
        max_histories=60,
        observables=("cxr", "brem"),
        brem_target_rse=1.0,
        brem_min_histories=20,
        brem_max_histories=60,
        max_history_share=1.0,
        min_effective_histories=0.0,
        max_variance_of_variance=1.0,
        stability_checks=0,
        **extra,
    )


def _case(precision):
    numerics = pr.Numerics(
        sampling=adaptive_sampling(precision),
        transport_core="per-electron",
        cxr_reflections=CxrReflections(
            mosaic_route="mc",
            mosaic_nodes=2,
            fiber=FiberAverage(cutoff_invang=4.0, azimuth_count=2, reflection_batch_size=16),
        ),
        energy_bins=pr.detectors.EnergyBins(
            cxr=np.arange(200.0, 3000.0, 40.0), brem=np.arange(500.0, 30000.0, 500.0)
        ),
    )
    return api.build_case(_scene(), numerics)


def _fixed(case, n, n_brem):
    fixed = {key: value for key, value in case.items() if key != "adaptive_precision"}
    fixed.update(Ne=n, Ne_brem=n_brem)
    return runner.run_case(fixed, transport_core="per-electron")


def test_fiber_case_with_mc_mosaic_accepts_adaptive_precision():
    case = _case(_precision())
    assert case.get("cxr_fiber") is not None and case.get("adaptive_precision") is not None
    assert case.get("mosaic_mc_fwhm_rad") and case.get("mosaic_fwhm_rad") is None
    fiber_emission.validate_fiber_case(dict(case))
    assert streaming_refusal(dict(case), live=True) is None


def test_fiber_adaptive_batch_means_require_retained_scoring():
    case = _case(_precision(batch_means_band_eV=(500.0, 2500.0)))
    assert streaming_refusal(dict(case), live=True) == (
        "adaptive batch means need every segment at once"
    )


def test_adaptive_fiber_run_is_the_fixed_n_run_at_its_realized_count():
    case = _case(_precision())
    output = runner.run_case(case, transport_core="per-electron")
    stats = output["adaptive_sampling"]
    n, n_brem = stats["realized_electrons"], stats["realized_electrons_brem"]
    assert stats["stop_reason"] == "converged" and n < case.Ne and n_brem < case.Ne_brem
    assert n_brem < n
    assert float(np.max(output["spec"])) > 0.0
    reference = _fixed(case, n, n_brem)
    for field in FIELDS:
        np.testing.assert_array_equal(output[field], reference[field], err_msg=field)
    assert output["brem_lattice_share"] == reference["brem_lattice_share"]
    record = output["brem_lattice_share"]["layers"][0]
    assert record["reflections"] == len(fiber_emission.fiber_domain(case).hkls)


def test_streamed_adaptive_fiber_run_matches_the_retained_run():
    case = _case(_precision())
    retained = runner.run_case(case, transport_core="per-electron")
    streamed = run_case_streamed(case, 7, transport_core="per-electron")
    for key in ("realized_electrons", "realized_electrons_brem", "stop_reason"):
        assert streamed["adaptive_sampling"][key] == retained["adaptive_sampling"][key], key
    # Streamed bin-mean lines agree within the Fourier image residual relative
    # to the peak (as in #1490); the rest is exact to summation order.
    peak = float(np.max(np.abs(retained["spec"])))
    np.testing.assert_allclose(
        streamed["spec"], retained["spec"], rtol=0.0, atol=FOURIER_RESIDUAL_TOL * peak
    )
    for field in FIELDS[1:]:
        np.testing.assert_allclose(
            streamed[field], retained[field], rtol=1e-12, atol=0.0, err_msg=field
        )


def test_fiber_measure_is_one_finite_value_per_electron():
    case = dict(_case(_precision()))
    fixed = {**case, "Ne": 20, "Ne_brem": 20}
    fixed.pop("adaptive_precision")
    tp = runner._transport_case(fixed, transport_core="per-electron")
    settings = AdaptiveSettings(
        target_rse=1.0,
        min_histories=20,
        max_histories=20,
        observables=("cxr", "brem"),
        brem_target_rse=1.0,
        brem_min_histories=20,
        brem_max_histories=20,
    )
    values = case_measure(fixed, settings)(0, 20, tp["segs"])
    for name in ("cxr", "brem"):
        value = np.asarray(_to_cpu(values[name]))
        assert value.shape == (20,) and np.all(np.isfinite(value)), name
        assert value.sum() > 0.0, name


def test_fiber_batch_means_reconstruct_the_band_of_the_final_spectra():
    band = (500.0, 2500.0)
    case = _case(_precision(batch_means_band_eV=band))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CxrYieldStatisticsWarning)
        output = runner.run_case(case, transport_core="per-electron")
    batch = output["adaptive_sampling"]["batch_means"]
    # Fiber lines are bin-mean: each batch's Fourier reducer picks its own span,
    # so line batches add to the run only to a few 1e-6 per bin (asserted at
    # 1e-5); basal bin-mean cases show the same gap. The continuum is exact.
    rtol = {"spec": 1e-5, "brem_wide": 1e-9}
    for key, grid in (("spec", "E_grid"), ("brem_wide", "E_grid_brem")):
        energies = output[grid]
        inside = (energies >= band[0]) & (energies <= band[1])
        assert np.max(output[key][inside]) > 0.0, key
        np.testing.assert_allclose(
            batch[key]["mean"], output[key][inside], rtol=rtol[key], atol=0.0
        )
        assert np.all(batch[key]["standard_error"] >= 0.0), key
