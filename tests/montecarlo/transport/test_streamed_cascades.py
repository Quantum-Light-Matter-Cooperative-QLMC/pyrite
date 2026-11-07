"""Live streamed scoring of secondary cascades (#923).

Cascade RNG is lineage-addressed, so each primary block runs its histories'
whole cascades exactly as the joined run. Streamed spectra must equal the
joined ``run_case`` path up to summation order on CPU fp64.
"""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.config import FixedSampling
from pyrite.detectors import EnergyBins
from pyrite.montecarlo import runner
from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed, streaming_refusal
from tests.helpers.backend import scaled_rtol
from tests.montecarlo.cascades.test_coupled_radiative_runner import (  # noqa: F401
    synthetic_si_tables,
)

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")
# Summation order only (fp64); a device-precision session gets the #611
# 16-epsilon whole-electron regrouping allowance. Zeros stay exact up to an
# absolute floor far below any populated bin of these small spectra.
RTOL, ATOL = scaled_rtol(1e-12, eps_multiple=16), 1e-300


pytestmark = pytest.mark.usefixtures("synthetic_si_tables")


@pytest.fixture(autouse=True)
def _require_pdatconf():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")


def _cascade_case(
    *, coupled, Ne, Ne_brem, threshold=5000.0, energy_keV=150.0, seed=None, spread=None
):
    """Silicon shell cascade inside the synthetic 5--200 keV BremsLib range."""
    radiative = (
        {"radiative_model": "bremslib-soft-hard", "radiative_cutoff_eV": 1000.0}
        if coupled
        else {"radiative_model": "uncoupled"}
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=Ne, brem_histories=Ne_brem),
        energy_model="midpoint",
        energy_bins=EnergyBins(
            cxr=np.linspace(1500.0, 2000.0, 20), brem=np.arange(5000.0, 150001.0, 5000.0)
        ),
        **({} if seed is None else {"seed": seed}),
    )
    physics = pr.Physics(
        elastic_model="mott",
        bremsstrahlung_model="bremslib",
        secondary_threshold_eV=threshold,
        **radiative,
    )
    target = pr.geometry.Slab(
        "silicon", thickness_ang=2.0e5, tilt_deg=30.0, footprint=pr.geometry.Footprint(5.0, 5.0)
    )
    beam = pr.Beam(
        energy_keV=energy_keV,
        transverse=pr.beams.Spot(),
        longitudinal=pr.beams.PointBunch(),
        energy_spread_frac=spread,
    )
    case = api.build_case(pr.Scene(beam, target, pr.Detector(), physics=physics), numerics)
    case = replace(case, E_cut_lines_keV=5.0, E_cut_brem_keV=5.0)
    assert case["cascade_rng_model"] == "lineage-addressed-v1"
    return case


def _joined(case, *, generations):
    """Joined reference; the fixture must launch at least ``generations - 1`` levels."""
    transport = runner._transport_case(case, transport_core="per-electron")
    per_generation = transport["segs"]["secondaries"]["tracks_per_generation"]
    assert len(per_generation) >= generations and all(per_generation), per_generation
    return runner._spectrum_case(case, transport)


def _assert_streamed_matches(case, reference, block):
    streamed = run_case_streamed(case, block, transport_core="per-electron")
    for key in SPECTRA:
        np.testing.assert_allclose(streamed[key], reference[key], rtol=RTOL, atol=ATOL, err_msg=key)
    for key in ("eta", "hit_frac", "n_segments"):
        assert streamed[key] == reference[key], key


def test_shell_cascade_stream_smoke():
    case = _cascade_case(coupled=False, Ne=5, Ne_brem=4, threshold=1000.0, seed=5)
    _assert_streamed_matches(case, _joined(case, generations=2), 2)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("Ne", "Ne_brem", "seed", "spread"),
    [(9, 12, 5, None), (12, 9, 5, None), (12, 9, 17, 0.08), (9, 12, 23, 0.08)],
)
def test_shell_cascade_streams_like_the_joined_run(Ne, Ne_brem, seed, spread):
    """Seeds, ``Ne``/``Ne_brem`` both ways and energy spread to generation 2."""
    case = _cascade_case(
        coupled=False, Ne=Ne, Ne_brem=Ne_brem, threshold=1000.0, seed=seed, spread=spread
    )
    assert streaming_refusal(case, live=True) is None
    reference = _joined(case, generations=3)
    n = max(Ne, Ne_brem)
    for block in (1, 5, n, n + 7):
        _assert_streamed_matches(case, reference, block)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("Ne", "Ne_brem", "seed", "spread"),
    [(30, 22, 3, None), (30, 22, 8, None), (22, 30, 8, None), (22, 30, 13, 0.05)],
)
def test_coupled_shell_cascade_streams_like_the_joined_run(Ne, Ne_brem, seed, spread):
    """The synthetic table floor (5 keV) bounds T_s, so these reach generation 1."""
    case = _cascade_case(coupled=True, Ne=Ne, Ne_brem=Ne_brem, seed=seed, spread=spread)
    assert case["radiative_model"] == "bremslib-soft-hard"
    reference = _joined(case, generations=2)
    n = max(Ne, Ne_brem)
    for block in (1, 7, n, n + 11):
        _assert_streamed_matches(case, reference, block)


def test_joined_block_cascades_stay_refused():
    case = _cascade_case(coupled=False, Ne=4, Ne_brem=4)
    with pytest.raises(ValueError, match="only for live streamed scoring"):
        runner._transport_case(case, transport_core="per-electron", block_electrons=2)


def test_unmarked_cascade_records_are_refused():
    """A pre-#923 cascade record has no marker; it is never replayed (#923)."""
    case = dict(_cascade_case(coupled=False, Ne=4, Ne_brem=4))
    del case["cascade_rng_model"]
    with pytest.raises(ValueError, match="dense-track cascade RNG"):
        runner._transport_case(case, transport_core="per-electron")
    with pytest.raises(ValueError, match="unsupported cascade_rng_model"):
        runner._transport_case(
            {**case, "cascade_rng_model": "dense-track-v0"}, transport_core="per-electron"
        )


def _pair_case(monkeypatch, *, positrons, Ne, Ne_brem, seed=11):
    """3 MeV Si with boosted synthetic BremsLib and pair rates, as ``_cascade`` uses."""
    from pyrite.materials.photon_cross_sections import photon_cross_sections_ang2
    from pyrite.montecarlo.runner import case_tables
    from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
    from pyrite.montecarlo.transport import pair_production
    from pyrite.xsgen.bremslib import tables as bremslib_tables
    from tests.helpers.bremslib import synthetic_bremslib_arrays

    def scaled(element, energies):
        sigma = dict(photon_cross_sections_ang2(element, energies))
        sigma["pair_nuclear"] = sigma["pair_nuclear"] * 1e9
        sigma["pair_electron"] = sigma["pair_electron"] * 1e9
        return sigma

    monkeypatch.setattr(pair_production, "photon_cross_sections_ang2", scaled)
    arrays = synthetic_bremslib_arrays(t1_MeV=np.array([1e-2, 1e-1, 1.0, 2.0, 5.0]))
    table = prepare_bremslib_table(arrays, atomic_number=14)
    table = replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * 3e3,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * 3e3,
    )
    monkeypatch.setattr(bremslib_tables, "resolve_bremsstrahlung_model", lambda *_: "bremslib")
    monkeypatch.setattr(case_tables, "_case_bremslib_tables", lambda _case: {"Si": table})
    monkeypatch.setattr(
        bremslib_tables, "load_bremsstrahlung_tables", lambda *_args, **_kwargs: {"Si": table}
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=Ne, brem_histories=Ne_brem),
        energy_model="midpoint",
        seed=seed,
        energy_bins=EnergyBins(
            cxr=np.linspace(1500.0, 2000.0, 20), brem=np.geomspace(2.0e4, 2.5e6, 24)
        ),
    )
    physics = pr.Physics(
        # Positron transport needs ELSEPA positron elastic tables.
        elastic_model="elsepa" if positrons else "mott",
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        secondary_threshold_eV=100_000.0,
        pair_production_model="penelope-2024",
        positron_transport=positrons,
    )
    target = pr.geometry.Slab(
        "silicon",
        thickness_ang=1.0e6,
        footprint=pr.geometry.Footprint(5.0, 5.0),
    )
    beam = pr.Beam(
        energy_keV=3000.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
    )
    case = api.build_case(pr.Scene(beam, target, pr.Detector(), physics=physics), numerics)
    return replace(case, E_cut_lines_keV=100.0, E_cut_brem_keV=100.0)


@pytest.mark.slow
@pytest.mark.parametrize("positrons", [False, True])
@pytest.mark.parametrize(("Ne", "Ne_brem", "seed"), [(6, 5, 11), (5, 6, 29)])
def test_pair_cascade_streams_like_the_joined_run(monkeypatch, positrons, Ne, Ne_brem, seed):
    """Pair conversion completes directions per block; streams match joined (#923)."""
    import warnings

    if positrons:
        from tests.helpers.positron_tables import require_positron_tables

        require_positron_tables()
    case = _pair_case(monkeypatch, positrons=positrons, Ne=Ne, Ne_brem=Ne_brem, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # untransported positrons
        transport = runner._transport_case(case, transport_core="per-electron")
        kinds = transport["segs"]["secondary_tracks"]["launch_kind"]
        assert np.any(kinds == 2), "need launched pair electrons"
        if positrons:
            assert np.any(kinds == 3), "need launched positrons"
        reference = runner._spectrum_case(case, transport)
        for block in (1, 4, 6, 9):  # 6 = max(Ne, Ne_brem)
            _assert_streamed_matches(case, reference, block)


def test_track_budget_fails_iff_the_joined_run_does(monkeypatch):
    """The run-wide track cap decides identically at every block size (#923)."""
    from pyrite.montecarlo.transport import secondaries

    case = _cascade_case(coupled=False, Ne=5, Ne_brem=4, threshold=1000.0, seed=5)
    joined = runner._transport_case(case, transport_core="per-electron")["segs"]
    launched = sum(joined["secondaries"]["tracks_per_generation"][1:])
    n = 5
    assert launched >= n
    tight, loose = (launched - 1) // n, -(-launched // n)
    monkeypatch.setattr(secondaries, "DEFAULT_SECONDARY_TRACKS_PER_PRIMARY", tight)
    with pytest.raises(RuntimeError, match="max_secondary_tracks"):
        runner._transport_case(case, transport_core="per-electron")
    for block in (1, 2, n):
        with pytest.raises(RuntimeError, match="max_secondary_tracks"):
            run_case_streamed(case, block, transport_core="per-electron")
    monkeypatch.setattr(secondaries, "DEFAULT_SECONDARY_TRACKS_PER_PRIMARY", loose)
    for block in (1, 2, n):
        run_case_streamed(case, block, transport_core="per-electron")


class _RecordedBudget:
    """Spy on the run-wide track budget: committed launches per block."""

    def __init__(self, monkeypatch):
        from pyrite.montecarlo.runner import block_transport

        self.instances = []
        base = block_transport._TrackBudget
        recorder = self

        class Budget(base):
            def __init__(self, n):
                super().__init__(n)
                self.initial = self.remaining
                recorder.instances.append(self)

        monkeypatch.setattr(block_transport, "_TrackBudget", Budget)

    def committed(self):
        (budget,) = self.instances
        return budget.initial - budget.remaining


@pytest.mark.parametrize("phase", ["transport", "brem_wide"])
def test_cascade_block_oom_replays_without_double_counting(monkeypatch, phase):
    """An injected allocator OOM replays only the failed cascade block (#923).

    The reference is the same simulated-CUDA streamed run without the OOM:
    the CUDA core disables the shell transport LUT, so it is its own
    realization. Earlier blocks are not re-scored, and the track budget
    counts the replayed block's descendants once.
    """
    from pyrite.montecarlo.runner import case_driver, pool
    from pyrite.montecarlo.runner.emission_adapters import EmissionAdapter

    case = _cascade_case(coupled=False, Ne=5, Ne_brem=4, threshold=1000.0, seed=5)
    simulate, evaluate = case_driver.simulate_trajectories, EmissionAdapter.evaluate
    calls, failed, armed = [], [], []

    class AllocatorOOM(RuntimeError):
        pass

    def transport(*args, **kwargs):
        start, resident = kwargs["_electron_start"], kwargs["keep_segments_on_device"]
        calls.append((start, resident))
        if armed and phase == "transport" and start == 2 and resident and not failed:
            failed.append(True)
            raise AllocatorOOM()
        # Exercise the exact driver/recovery path without requiring hardware.
        kwargs.update(transport_core="per-electron", keep_segments_on_device=False)
        return simulate(*args, **kwargs)

    def score(self, batch, slot, **kwargs):
        rows = batch.segments["electron_id"]
        if armed and slot == phase and rows.size and int(rows.min()) == 2 and not failed:
            failed.append(True)
            raise AllocatorOOM()
        return evaluate(self, batch, slot, **kwargs)

    budget = _RecordedBudget(monkeypatch)
    monkeypatch.setattr(runner, "_case_transport_core", lambda *_: "cuda")
    monkeypatch.setattr(case_driver, "simulate_trajectories", transport)
    monkeypatch.setattr(runner, "_is_gpu_oom", lambda e: isinstance(e, AllocatorOOM))
    monkeypatch.setattr(pool, "_admit_host_segments", lambda: None)
    monkeypatch.setattr(EmissionAdapter, "evaluate", score)
    reference = run_case_streamed(case, 2, transport_core="cuda")
    calls.clear()
    armed.append(True)
    actual = run_case_streamed(case, 2, transport_core="cuda")
    assert failed
    assert calls == (
        [(0, True), (2, True), (2, False), (4, True)]
        if phase == "transport"
        else [(0, True), (2, True), (4, True)]
    )
    if phase == "transport":
        assert actual["_transport_gpu_oom_retries"] == 1
    clean, replayed = budget.instances
    assert clean.initial - clean.remaining > 0
    assert replayed.initial - replayed.remaining == clean.initial - clean.remaining
    for key in SPECTRA:
        np.testing.assert_allclose(actual[key], reference[key], rtol=RTOL, atol=ATOL, err_msg=key)
    for key in ("eta", "hit_frac", "n_segments"):
        assert actual[key] == reference[key], key


def test_cascade_step_budget_retry_reproduces_the_block(monkeypatch):
    """A step-limit retry reruns the whole cascade block from its keys (#923)."""
    from pyrite.montecarlo.runner import step_budget
    from pyrite.montecarlo.transport import TransportStepLimitError

    case = _cascade_case(coupled=False, Ne=5, Ne_brem=4, threshold=1000.0, seed=5)
    transport_ref = runner._transport_case(case, transport_core="per-electron")
    launched = sum(transport_ref["segs"]["secondaries"]["tracks_per_generation"][1:])
    reference = runner._spectrum_case(case, transport_ref)
    retry, limited = step_budget.retry_step_budget, []

    def counting_retry(simulate, *args, **kwargs):
        def attempt(*a, **kw):
            try:
                return simulate(*a, **kw)
            except TransportStepLimitError:
                limited.append(kw["max_steps"])
                raise

        return retry(attempt, *args, **kwargs)

    from pyrite.montecarlo.runner import block_transport

    budget = _RecordedBudget(monkeypatch)
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS", "4")
    monkeypatch.setattr(block_transport, "retry_step_budget", counting_retry)
    actual = run_case_streamed(case, 2, transport_core="per-electron")
    assert limited, "the small initial budget must force a retry"
    assert budget.committed() == launched
    for key in SPECTRA:
        np.testing.assert_allclose(actual[key], reference[key], rtol=RTOL, atol=ATOL, err_msg=key)
    for key in ("eta", "hit_frac", "n_segments"):
        assert actual[key] == reference[key], key


def _cuda_available():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


#: CUDA scoring regroups whole-electron fp32 sums: the #611 16-epsilon allowance.
CUDA_RTOL = 16 * np.finfo(np.float32).eps


def _assert_cuda_streamed_matches(case, reference, block):
    streamed = run_case_streamed(case, block, transport_core="cuda")
    for key in SPECTRA:
        np.testing.assert_allclose(
            streamed[key], reference[key], rtol=CUDA_RTOL, atol=0, err_msg=key
        )
    for key in ("eta", "hit_frac", "n_segments"):
        assert streamed[key] == reference[key], key


@pytest.mark.slow
@pytest.mark.hardware
@pytest.mark.skipif(not _cuda_available(), reason="no CUDA device")
@pytest.mark.parametrize(
    ("coupled", "Ne", "Ne_brem", "spread"),
    [(False, 12, 9, None), (False, 9, 12, 0.08), (True, 30, 22, None), (True, 22, 30, 0.05)],
)
def test_cuda_shell_cascade_streams_like_the_joined_run(coupled, Ne, Ne_brem, spread):
    """CUDA joined vs streamed shell cascades share one lineage realization (#923)."""
    kwargs = dict(threshold=1000.0) if not coupled else {}
    case = _cascade_case(coupled=coupled, Ne=Ne, Ne_brem=Ne_brem, seed=5, spread=spread, **kwargs)
    transport = runner._transport_case(case, transport_core="cuda")
    assert all(transport["segs"]["secondaries"]["tracks_per_generation"][:2])
    reference = runner._spectrum_case(case, transport)
    n = max(Ne, Ne_brem)
    for block in (1, 5, n, n + 7):
        _assert_cuda_streamed_matches(case, reference, block)


@pytest.mark.slow
@pytest.mark.hardware
@pytest.mark.skipif(not _cuda_available(), reason="no CUDA device")
@pytest.mark.parametrize("positrons", [False, True])
def test_cuda_pair_cascade_streams_like_the_joined_run(monkeypatch, positrons):
    import warnings

    if positrons:
        from tests.helpers.positron_tables import require_positron_tables

        require_positron_tables()
    case = _pair_case(monkeypatch, positrons=positrons, Ne=6, Ne_brem=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # untransported positrons
        transport = runner._transport_case(case, transport_core="cuda")
        kinds = np.asarray(transport["segs"]["secondary_tracks"]["launch_kind"])
        assert np.any(kinds == 2), "need launched pair electrons"
        if positrons:
            assert np.any(kinds == 3), "need launched positrons"
        reference = runner._spectrum_case(case, transport)
        for block in (1, 4, 6, 9):
            _assert_cuda_streamed_matches(case, reference, block)
