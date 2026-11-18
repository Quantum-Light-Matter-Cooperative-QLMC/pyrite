"""Live streamed scoring of adaptive cases (#1280).

A measure-only monitored pass keeps no chunk and returns the realized
``(Ne, Ne_brem)``; the fixed case at that pair then streams. For one seed the
result must equal the retained adaptive route: the same realized counts and
sampling record, spectra within streamed-vs-joined summation order.

Validation: adaptive-sample-size-stopping
"""

import numpy as np
import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import block_transport
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed, streaming_refusal
from tests.montecarlo.runner.test_adaptive_stopping import _case
from tests.montecarlo.transport.test_independent_stopping import _policy

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")


def _requested(spread=None, **policy):
    case = {
        **_case(energy_spread_frac=spread),
        # Seed 8 puts the maximum initial energy past the realized population,
        # forcing the energy-range replay (see test_independent_stopping).
        "seed": 8 if spread is not None else 1,
        "E_cut_lines_keV": 29.0,
        "E_cut_brem_keV": 1.0,
    }
    settings = _policy(
        target_rse=1.0,
        min_histories=80,
        max_histories=120,
        brem_target_rse=1.0,
        brem_min_histories=40,
        **policy,
    )
    return {
        **case,
        "Ne": settings.max_histories,
        "Ne_brem": settings.max_histories,
        "adaptive_precision": settings.to_dict(),
    }


def _retained(case):
    transport = runner._transport_case(case, transport_core="per-electron")
    return runner._spectrum_case(case, transport)


@pytest.mark.slow
@pytest.mark.parametrize("spread", [None, 0.001])
def test_streamed_adaptive_matches_retained(monkeypatch, spread):
    case = _requested(spread)
    reference = _retained(case)

    def no_join(*args, **kwargs):
        raise AssertionError("streamed adaptive joined kept blocks")

    monkeypatch.setattr(block_transport, "merge_electron_blocks", no_join)
    streamed = run_case_streamed(case, 13, transport_core="per-electron")

    record, expected = streamed["adaptive_sampling"], reference["adaptive_sampling"]
    assert record["realized_electrons_brem"] < record["realized_electrons"]
    assert record["replayed_transport"] is (spread is not None)
    assert record == expected
    np.testing.assert_array_equal(streamed["E_grid"], reference["E_grid"])
    np.testing.assert_array_equal(streamed["E_grid_brem"], reference["E_grid_brem"])
    for key in SPECTRA:
        scale = np.abs(reference[key]).max()
        np.testing.assert_allclose(
            streamed[key], reference[key], rtol=1e-12, atol=1e-14 * scale, err_msg=key
        )
    for key in ("eta", "hit_frac", "n_segments"):
        assert streamed[key] == reference[key], key


@pytest.mark.parametrize(
    ("extra", "policy", "match"),
    [
        ({"reflection_selection": {"automatic": True}}, {}, "reflection_selection"),
        ({}, {"batch_means_band_eV": (1000.0, 10000.0)}, "batch means"),
    ],
)
def test_population_wide_adaptive_outputs_are_refused(monkeypatch, extra, policy, match):
    """Automatic selection (#1281) and batch means keep the retained route."""
    case = {**_requested(**policy), **extra}
    assert streaming_refusal(case, live=True) is not None

    def no_transport(*args, **kwargs):
        raise AssertionError("transport ran")

    monkeypatch.setattr(runner, "_transport_case", no_transport)
    with pytest.raises(ValueError, match=match):
        run_case_streamed(case, 10)


def test_plain_adaptive_case_is_live_streamable():
    assert streaming_refusal(_requested(), live=True) is None
