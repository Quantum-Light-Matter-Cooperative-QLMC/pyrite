"""Case-bound refinement ladder on a tiny real transport (issue #109).

Ne=3 through 1 um of HOPG at 30 keV is far from a converged spectrum; these
tests pin plumbing contracts that do not depend on Monte Carlo statistics:
one transport per ladder, production-identical spectrum reductions, grid
independence of the transport RNG stream, and fixed-seed resume only on
bit-identical trajectories.
"""

import numpy as np
import pytest

from pyrite.detectors.spec import EagleXO, Timepix3
from pyrite.energy_grid import convergence_case as cc
from pyrite.energy_grid.convergence import (
    SegmentMismatchError,
    SpectrumSample,
    evaluate_ladder,
    richardson_acceptance,
    segment_fingerprint,
    spectrum_observables,
)
from pyrite.montecarlo import runner

TINY = dict(thickness_ang=1.0e4, n_electrons=3, seed=0)
CONFIG = {
    "material": "hopg",
    "energy_keV": 30.0,
    "tilt_deg": 5.0,
    "tilt_azim_deg": 95.0,
    "thickness_ang": 1.0e4,
    "n_electrons": 3,
    "seed": 0,
}
SPACINGS = (96.0, 48.0, 24.0)


@pytest.fixture(scope="module")
def case():
    return cc.build_ladder_case("hopg", 30.0, 5.0, 95.0, **TINY)


@pytest.fixture(scope="module")
def ladder(case):
    return cc.CaseLadder(case, transport_core="lockstep")


def test_ladder_rung_at_the_case_grid_is_the_production_spectrum(case, ladder):
    production = runner._spectrum_case(case, ladder.transport)

    parts = ladder.evaluate_components(ladder.transport["E_grid"])

    assert np.array_equal(parts["lines"], np.asarray(production["spec"]))
    assert np.array_equal(parts["characteristic"], np.asarray(production["spec_characteristic"]))
    assert np.array_equal(parts["brem"], np.asarray(production["brem"]))


def test_line_grid_does_not_enter_the_transport_rng_stream(case, ladder):
    start, stop = ladder.bandwidth_eV
    regridded = {
        **case,
        "E_grid_line": np.linspace(start, stop, 7 * 1000 + 1),
        "line_grid_policy": None,
    }

    transport = runner._transport_case(regridded, transport_core="lockstep")

    assert transport["E_grid"].size != ladder.transport["E_grid"].size
    assert segment_fingerprint(transport["segs"]) == ladder.fingerprint


def test_spectrum_only_ladder_never_reruns_transport(ladder, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a ladder rung re-ran transport")

    monkeypatch.setattr(runner, "simulate_trajectories", forbidden)
    monkeypatch.setattr(runner, "_transport_case", forbidden)
    grids = cc.ladder_grids(*ladder.bandwidth_eV, SPACINGS)

    samples = [ladder.evaluate(grid) for grid in grids]

    assert [sample.line.shape for sample in samples] == [grid.shape for grid in grids]
    assert segment_fingerprint(ladder.segments) == ladder.fingerprint


def test_continuum_is_evaluated_directly_on_each_candidate_grid(ladder, monkeypatch):
    original = runner._brem_wide_from_segments
    evaluated = []

    def record_grid(segments, energy_eV, *args, **kwargs):
        evaluated.append(np.asarray(energy_eV).copy())
        return original(segments, energy_eV, *args, **kwargs)

    monkeypatch.setattr(runner, "_brem_wide_from_segments", record_grid)
    grids = ladder.continuum_grids((65, 129), 29_000.0)

    continua = [ladder.continuum(grid) for grid in grids]

    assert [continuum.shape for continuum in continua] == [grid.shape for grid in grids]
    assert all(
        np.array_equal(actual, expected) for actual, expected in zip(evaluated, grids, strict=True)
    )
    assert segment_fingerprint(ladder.segments) == ladder.fingerprint


def test_continuum_observables_converge_on_identical_trajectories(ladder):
    # Rungs start at the medium's own derived continuum floor and carry the
    # edge/endpoint refinement (#100), not a literal 100.0 eV geometric start.
    grids = ladder.continuum_grids((2049, 4097, 8193), 29_000.0)
    detectors = {
        "timepix3_counts": Timepix3(n_mc=32, seed=7),
        "eaglexo_counts": EagleXO(),
    }

    def evaluate(grid):
        continuum = ladder.continuum(grid)
        return SpectrumSample(np.zeros_like(continuum), continuum)

    rungs = evaluate_ladder(
        grids,
        evaluate,
        segments=ladder.segments,
        observables=lambda E, line, continuum: spectrum_observables(
            E, line, continuum, detectors=detectors
        ),
    )
    gated = {
        "continuum_yield": "intrinsic_source",
        "continuum_centroid_eV": "intrinsic_source",
        "timepix3_continuum_counts": "detected_counts",
        "eaglexo_continuum_counts": "detected_counts",
    }
    report = richardson_acceptance(rungs, gated=gated, diagnostics={})

    assert report.accepted_spacing_eV == rungs[0].spacing_eV
    assert all(verdict.accepted for verdict in report.triples[0].observables)


def test_run_config_checkpoints_per_rung_and_resumes_on_identical_segments():
    checks = iter([False, True])
    saved = []

    partial, complete = cc.run_config(
        CONFIG, SPACINGS, save=saved.append, out_of_time=lambda: next(checks, True)
    )

    assert complete is False
    assert len(partial["rungs"]) == 1
    assert saved[-1] is partial

    finished, complete = cc.run_config(CONFIG, SPACINGS, state=partial)

    assert complete is True
    expected_grids = cc.ladder_grids(*finished["bandwidth_eV"], SPACINGS)
    expected_spacings = [float(np.diff(grid)[0]) for grid in expected_grids]
    assert [rung["spacing_eV"] for rung in finished["rungs"]] == pytest.approx(expected_spacings)
    assert finished["fingerprint"] == partial["fingerprint"]
    assert len(finished["transport_wall_s"]) == 2
    assert len(finished["report"]["triples"]) == 1
    assert finished["sinc_estimate"]["intrinsic_source"]["aliased_weight_limit"] == 1e-3
    for rung in finished["rungs"]:
        assert set(rung["observables"]) >= {
            "yield",
            "continuum_yield",
            "continuum_centroid_eV",
            "timepix3_counts",
            "eaglexo_counts",
            "timepix3_continuum_counts",
            "eaglexo_continuum_counts",
        }
        assert rung["wall_s"] >= 0.0 and rung["host_peak_rss_mib"] > 0.0


def test_run_config_refuses_to_resume_on_different_trajectories():
    tampered = {
        "spacings_eV": list(SPACINGS),
        "fingerprint": {"n_segments": 1, "digest": "0" * 32},
        "rungs": [],
    }

    with pytest.raises(SegmentMismatchError, match="n_segments"):
        cc.run_config(CONFIG, SPACINGS, state=tampered)


def test_run_config_refuses_a_checkpoint_from_another_ladder():
    with pytest.raises(ValueError, match="used spacings"):
        cc.run_config(CONFIG, SPACINGS, state={"spacings_eV": [10.0, 5.0, 2.5]})


def test_ladder_grids_nest_only_for_exact_halving():
    nested = cc.ladder_grids(10.0, 2600.0, (12.0, 6.0, 3.0))
    independent = cc.ladder_grids(10.0, 2600.0, (12.0, 5.0, 3.0))

    assert np.array_equal(nested[1][::2], nested[0])
    assert np.diff(independent[1]).max() <= 5.0
    with pytest.raises(ValueError, match="strictly decreasing"):
        cc.ladder_grids(10.0, 2600.0, (3.0, 6.0))


def test_continuum_scores_coupled_hard_photons_from_host_rows(monkeypatch):
    """Hard photons read transport-precision host rows, not the staged copy (#172)."""
    segments = {"L_ang": np.ones(3), "r_mid": np.zeros((3, 3))}
    transport = {"segs": segments, "n_hat": np.array([0.0, 0.0, 1.0]), "Ne_brem": 3}
    seen = {}

    def fake(_segments, grid, _case, _n_hat, _layers, *, groove=None, Ne=None, event_segments=None):
        seen["event_segments"] = event_segments
        return np.zeros(grid.size)

    monkeypatch.setattr(runner, "_brem_wide_from_segments", fake)
    ladder = cc.CaseLadder({"crystal": "silicon"}, transport=transport)
    ladder.continuum([1_000.0, 2_000.0])
    assert seen["event_segments"] is ladder.segments
