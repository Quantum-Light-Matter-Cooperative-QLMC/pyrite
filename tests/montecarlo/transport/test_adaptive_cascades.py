"""Adaptive electron counts for shell-secondary cascades (#1240).

The sample is a primary history with all its descendants. The monitor reads
streamed cascade blocks; the returned transport is the fixed-count cascade at
the realized primaries, so it must equal the fixed run at that count.

Validation: adaptive-sample-size-stopping
"""

import warnings

import numpy as np
import pytest

from pyrite._cxr_grid_policy import CxrYieldStatisticsWarning
from pyrite._precision import Precision
from pyrite.montecarlo import runner
from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.runner.adaptive import RunningMoments, case_measure, run_case_adaptive
from tests.montecarlo.cascades.test_coupled_radiative_runner import (  # noqa: F401
    synthetic_si_tables,
)
from tests.montecarlo.transport.test_streamed_cascades import _cascade_case, _pair_case

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")
RTOL, ATOL = 1e-12, 1e-300
#: Guards disabled: the bare relative-standard-error rule.
_RULE_ONLY = dict(max_history_share=1.0, min_effective_histories=0.0, stability_blocks=0)
#: One monitor band over the line grid and the continuum grid, so both observables score.
BAND = dict(observables=("cxr", "brem"), band_eV=(1500.0, 150000.0))

pytestmark = pytest.mark.usefixtures("synthetic_si_tables")


@pytest.fixture(autouse=True)
def _require_pdatconf():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")


def _fixed(case, n, n_brem=None):
    fixed = {**case, "Ne": n, "Ne_brem": n if n_brem is None else n_brem}
    transport = runner._transport_case(fixed, transport_core="per-electron")
    return transport, runner._spectrum_case(fixed, transport)


def _adaptive(case, settings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CxrYieldStatisticsWarning)
        return run_case_adaptive(case, settings, transport_core="per-electron")


def _assert_matches(result, reference):
    for key in SPECTRA:
        np.testing.assert_allclose(result[key], reference[key], rtol=RTOL, atol=ATOL, err_msg=key)
    for key in ("eta", "hit_frac", "n_segments"):
        assert result[key] == reference[key], key


def _launched(transport):
    return sum(transport["segs"]["secondaries"]["tracks_per_generation"][1:])


@pytest.mark.parametrize(("block", "maximum"), [(1, 6), (4, 8), (6, 6)])
def test_adaptive_shell_cascade_is_the_fixed_run_at_its_count(block, maximum):
    """An unreachable target stops at the maximum; every block size gives the fixed run."""
    case = _cascade_case(coupled=False, Ne=4, Ne_brem=4, threshold=1000.0, seed=5)
    settings = Precision(1e-9, max(block, 2), maximum, block, **BAND, **_RULE_ONLY)
    result, stats = _adaptive(case, settings)
    assert stats["realized_electrons"] == maximum and stats["stop_reason"] == "max_electrons"
    for name in ("cxr", "brem"):
        assert stats["statistics"][name]["n_electrons"] == maximum
    transport, reference = _fixed(case, maximum)
    assert _launched(transport) > 0, "need launched descendants"
    _assert_matches(result, reference)


@pytest.mark.parametrize("spread", [None, 0.08])
def test_converged_shell_cascade_stop_is_the_fixed_run(spread):
    """A converged stop before the maximum, with and without the energy-range replay."""
    case = _cascade_case(coupled=False, Ne=4, Ne_brem=4, threshold=1000.0, seed=17, spread=spread)
    settings = Precision(0.5, 6, 30, 3, **BAND, **_RULE_ONLY)
    result, stats = _adaptive(case, settings)
    n = stats["realized_electrons"]
    assert stats["stop_reason"] == "converged" and 6 <= n < 30, stats["stop_reason"]
    assert stats["replayed_transport"] is (spread is not None)
    transport, reference = _fixed(case, n)
    assert _launched(transport) > 0
    _assert_matches(result, reference)
    again, again_stats = _adaptive(case, settings)
    assert again_stats == stats
    _assert_matches(again, result)


def test_independent_brem_stop_cascade_is_the_fixed_pair():
    """Primaries past the brem prefix run to the line cutoff; their descendants
    carry the root id and stay out of the brem population."""
    case = _cascade_case(coupled=False, Ne=4, Ne_brem=4, threshold=1000.0, seed=5)
    settings = Precision(
        1e-9,
        4,
        8,
        2,
        **BAND,
        brem_target_rse=1e-9,
        brem_max_histories=4,
        **_RULE_ONLY,
    )
    result, stats = _adaptive(case, settings)
    assert (stats["realized_electrons"], stats["realized_electrons_brem"]) == (8, 4)
    _, reference = _fixed(case, 8, 4)
    _assert_matches(result, reference)


def test_monitor_samples_are_root_summed_primaries():
    """Descendant rows add to their root primary's value; each primary is one sample."""
    case = _cascade_case(coupled=False, Ne=6, Ne_brem=6, threshold=1000.0, seed=5)
    transport, _ = _fixed(case, 6)
    segs = transport["segs"]
    assert _launched(transport) > 0
    settings = Precision(0.1, 6, 6, 6, **BAND)
    measure = case_measure({**case, "E_grid": transport["E_grid"]}, settings)
    whole = measure(0, 6, segs)
    generation = np.asarray(segs["generation"])
    primary_rows = {
        key: (
            value[generation == 0] if getattr(value, "shape", ())[:1] == generation.shape else value
        )
        for key, value in segs.items()
    }
    primaries = measure(0, 6, primary_rows)
    for name in ("cxr", "brem"):
        assert whole[name].shape == (6,)
        assert np.all(whole[name] >= primaries[name])
    # Descendants carry measurable brem onto some root.
    assert np.any(whole["brem"] > primaries["brem"])
    moments = RunningMoments()
    moments.add_block(whole["brem"])
    assert moments.n == 6
    assert moments.max_share() == pytest.approx(whole["brem"].max() / whole["brem"].sum())


@pytest.mark.parametrize("positrons", [False, True])
@pytest.mark.parametrize(("block", "maximum"), [(1, 4), (3, 6)])
def test_adaptive_pair_cascade_is_the_fixed_run_at_its_count(
    monkeypatch, positrons, block, maximum
):
    """Pair electrons and positrons are descendants of their root primary (#1240)."""
    if positrons:
        from tests.helpers.positron_tables import require_positron_tables

        require_positron_tables()
    case = _pair_case(monkeypatch, positrons=positrons, Ne=4, Ne_brem=4)
    settings = Precision(
        1e-9,
        max(block, 2),
        maximum,
        block,
        observables=("cxr", "brem"),
        band_eV=(1500.0, 2.5e6),
        **_RULE_ONLY,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # untransported positrons
        result, stats = _adaptive(case, settings)
        transport, reference = _fixed(case, maximum)
    assert stats["realized_electrons"] == maximum
    kinds = transport["segs"]["secondary_tracks"]["launch_kind"]
    assert np.any(kinds == 2), "need launched pair electrons"
    if positrons:
        assert np.any(kinds == 3), "need launched positrons"
    _assert_matches(result, reference)


def test_track_cap_fails_iff_the_fixed_run_does(monkeypatch):
    """The run-wide track cap decides identically for adaptive runs at every block size."""
    from pyrite.montecarlo.transport import secondaries

    case = _cascade_case(coupled=False, Ne=6, Ne_brem=6, threshold=1000.0, seed=5)
    launched = _launched(_fixed(case, 6)[0])
    assert launched >= 6
    tight, loose = (launched - 1) // 6, -(-launched // 6)
    monkeypatch.setattr(secondaries, "DEFAULT_SECONDARY_TRACKS_PER_PRIMARY", tight)
    with pytest.raises(RuntimeError, match="max_secondary_tracks"):
        _fixed(case, 6)
    for block in (2, 3, 6):
        settings = Precision(1e-9, block, 6, block, **BAND, **_RULE_ONLY)
        with pytest.raises(RuntimeError, match="max_secondary_tracks"):
            _adaptive(case, settings)
    monkeypatch.setattr(secondaries, "DEFAULT_SECONDARY_TRACKS_PER_PRIMARY", loose)
    for block in (2, 3, 6):
        settings = Precision(1e-9, block, 6, block, **BAND, **_RULE_ONLY)
        assert _adaptive(case, settings)[1]["realized_electrons"] == 6
