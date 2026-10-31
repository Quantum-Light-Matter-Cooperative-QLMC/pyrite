"""Live streamed spectrum scoring over electron blocks (#611).

Streaming scores each transport block as it finishes and keeps none of them.
Spectra must equal the joined full-trajectory path up to summation order,
transport counts must be exact, and unsupported cases must fail before any
transport.
"""

import gc
import weakref

import numpy as np
import pytest

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import block_transport
from pyrite.montecarlo.runner.streamed_scoring import (
    LIVE_STREAM_REFUSED_KEYS,
    STREAM_REFUSED_KEYS,
    run_case_streamed,
)

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")


@pytest.fixture(scope="module")
def case():
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=30.0,
        n_electrons=8,
        n_electrons_brem=8,
    )
    base = dict(build_cases(sweep)[0])
    # A final line grid (no case-local resolution) and unequal populations,
    # which exercise the per-estimator electron limits.
    base.pop("line_grid_policy")
    return {**base, "Ne": 40, "Ne_brem": 52}


@pytest.fixture(scope="module")
def reference(case):
    transport = runner._transport_case(case, transport_core="per-electron")
    return runner._spectrum_case(case, transport)


def test_case_is_streamable(case):
    assert case.get("line_grid_policy") is None


@pytest.mark.parametrize("block", [7, 20, 52, 1000])
def test_streamed_spectra_match_the_full_trajectory_path(case, reference, block):
    streamed = run_case_streamed(case, block, transport_core="per-electron")
    for key in SPECTRA:
        np.testing.assert_allclose(streamed[key], reference[key], rtol=1e-12, atol=0, err_msg=key)
    np.testing.assert_array_equal(streamed["E_grid"], reference["E_grid"])
    np.testing.assert_array_equal(streamed["E_grid_brem"], reference["E_grid_brem"])
    for key in ("eta", "hit_frac", "n_segments"):
        assert streamed[key] == reference[key], key


@pytest.mark.parametrize("key", STREAM_REFUSED_KEYS + LIVE_STREAM_REFUSED_KEYS)
def test_unsupported_cases_fail_before_transport(monkeypatch, case, key):
    def no_transport(*args, **kwargs):
        raise AssertionError("transport ran")

    monkeypatch.setattr(runner, "_transport_case", no_transport)
    with pytest.raises(ValueError, match="cannot stream"):
        run_case_streamed({**case, key: {"coherence": "x"}}, 10)


def test_bunch_transition_radiation_is_refused(monkeypatch, case):
    monkeypatch.setattr(runner, "_transport_case", None)
    with pytest.raises(ValueError, match="bunch transition radiation"):
        run_case_streamed({**case, "transition_radiation": {"coherence": "bunch"}}, 10)


def test_lockstep_core_is_refused(case):
    with pytest.raises(ValueError, match="per-electron or CUDA"):
        run_case_streamed(case, 10, transport_core="lockstep")


def test_consumed_blocks_are_released_before_the_next_block():
    live = []

    def simulate_block(start, stop):
        # The previous block must already be unreachable when the next starts.
        gc.collect()
        assert all(ref() is None for ref in live)
        L = np.ones(3 * (stop - start))
        live.append(weakref.ref(L))
        ids = np.repeat(np.arange(stop - start), 3)
        return {"L_ang": L, "electron_id": ids, "Ne": stop - start, "n_backscattered": 1}

    seen = []
    counts = block_transport.transport_electron_blocks(
        simulate_block,
        10,
        4,
        seed=0,
        bunch={},
        consume=lambda block: seen.append(int(block["electron_id"].max())),
    )
    assert seen == [3, 7, 9]  # globalized indices
    assert counts["Ne"] == 10 and counts["n_backscattered"] == 3


def test_capacity_hint_reaches_later_blocks_and_changes_nothing(monkeypatch, case):
    from pyrite.montecarlo.runner.transport_options import _TRANSPORT_TUNING

    plain = run_case_streamed(case, 13, transport_core="per-electron")
    configs = []
    simulate = runner.simulate_trajectories

    def spy(*args, **kwargs):
        configs.append(kwargs.get("per_electron_config"))
        return simulate(*args, **kwargs)

    monkeypatch.setattr(runner, "simulate_trajectories", spy)
    hinted = run_case_streamed(case, 13, transport_core="per-electron", capacity_hint=True)

    for key in (*SPECTRA, "eta", "hit_frac", "n_segments"):
        np.testing.assert_array_equal(hinted[key], plain[key], err_msg=key)
    assert configs[0] is None  # the first block learns from the default guess
    hinted_configs = [c for c in configs[1:] if c is not None]
    assert hinted_configs, "no later block received the hint"
    assert all(c.probe_electrons == 0 and c.seg_capacity > 1 for c in hinted_configs)
    assert _TRANSPORT_TUNING.get() is None  # override restored
