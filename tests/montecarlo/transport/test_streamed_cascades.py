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
from tests.montecarlo.cascades.test_coupled_radiative_runner import (  # noqa: F401
    synthetic_si_tables,
)

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")
# Summation order only; zeros stay exact up to an absolute floor far below any
# populated bin of these small spectra.
RTOL, ATOL = 1e-12, 1e-300


pytestmark = pytest.mark.usefixtures("synthetic_si_tables")


@pytest.fixture(autouse=True)
def _require_pdatconf():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")


def _cascade_case(*, coupled, Ne, Ne_brem, threshold=5000.0, energy_keV=150.0, seed=None):
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
        energy_keV=energy_keV, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
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
@pytest.mark.parametrize(("Ne", "Ne_brem"), [(9, 12), (12, 9)])
def test_shell_cascade_streams_like_the_joined_run(Ne, Ne_brem):
    case = _cascade_case(coupled=False, Ne=Ne, Ne_brem=Ne_brem, threshold=1000.0, seed=5)
    assert streaming_refusal(case, live=True) is None
    reference = _joined(case, generations=3)
    n = max(Ne, Ne_brem)
    for block in (1, 5, n, n + 7):
        _assert_streamed_matches(case, reference, block)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [3, 8])
def test_coupled_shell_cascade_streams_like_the_joined_run(seed):
    """The synthetic table floor (5 keV) bounds T_s, so these reach generation 1."""
    case = _cascade_case(coupled=True, Ne=30, Ne_brem=22, seed=seed)
    assert case["radiative_model"] == "bremslib-soft-hard"
    reference = _joined(case, generations=2)
    for block in (1, 7, 30, 41):
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
def test_pair_cascade_streams_like_the_joined_run(monkeypatch, positrons):
    """Pair conversion completes directions per block; streams match joined (#923)."""
    import warnings

    if positrons:
        from tests.helpers.positron_tables import require_positron_tables

        require_positron_tables()
    case = _pair_case(monkeypatch, positrons=positrons, Ne=6, Ne_brem=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # untransported positrons
        transport = runner._transport_case(case, transport_core="per-electron")
        kinds = transport["segs"]["secondary_tracks"]["launch_kind"]
        assert np.any(kinds == 2), "need launched pair electrons"
        if positrons:
            assert np.any(kinds == 3), "need launched positrons"
        reference = runner._spectrum_case(case, transport)
        for block in (1, 4, 6, 9):
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
