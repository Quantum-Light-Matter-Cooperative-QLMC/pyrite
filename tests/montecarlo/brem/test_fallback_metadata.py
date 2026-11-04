"""Measured substitutions count scored rows, independently of photon/reduction chunks."""

import warnings

import numpy as np
import pytest

from pyrite._backend import xp
from pyrite._physics_fallback import PhysicsFallbackWarning, collect_physics_fallbacks
from pyrite.montecarlo.spectrum import brem
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from tests.helpers.bremslib import synthetic_bremslib_arrays


@pytest.fixture
def table():
    return prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6, key="k", digest="d")


def segments(energies):
    n = len(energies)
    return dict(
        r_mid=np.tile([0.0, 0.0, 50.0], (n, 1)),
        v_hat=np.tile([0.0, 0.0, 1.0], (n, 1)),
        L_ang=np.full(n, 10.0),
        E_keV=np.asarray(energies, dtype=float),
        elec_id=np.arange(n),
        Ne=n,
        thickness_ang=100.0,
    )


def score(energies, table, **kwargs):
    return brem.mc_brem_spectrum(
        segments(energies),
        np.array([100.0, 200.0, 400.0]),
        composition=[("C", 0.1)],
        cross_section_model="bremslib",
        bremslib_tables={"C": table},
        **kwargs,
    )


def missing_eedl(*_args, **_kwargs):
    raise brem.EEDLBremsstrahlungDataUnavailable("test EEDL is absent")


def test_covered_bremslib_never_loads_unused_eedl(monkeypatch, table):
    def forbidden(*_args, **_kwargs):
        pytest.fail("covered BremsLib rows must not load an unused surrogate")

    monkeypatch.setattr(brem, "load_bremsstrahlung_cross_sections", forbidden)
    with collect_physics_fallbacks() as audit:
        result = score(
            [table.minimum_incident_energy_keV, table.maximum_incident_energy_keV], table
        )
    assert np.all(np.isfinite(result))
    assert audit.events() == []


def test_strict_refuses_below_floor_even_with_ignored_warnings(table):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with collect_physics_fallbacks():
            with pytest.raises(PhysicsFallbackWarning, match="outside the BremsLib range"):
                score([5.0, 20.0], table)


@pytest.mark.parametrize("block", [None, 1, 2])
def test_chained_fallback_counts_and_spans_ignore_photon_blocks(monkeypatch, table, block):
    monkeypatch.setattr(brem, "load_bremsstrahlung_cross_sections", missing_eedl)
    with pytest.warns(PhysicsFallbackWarning), collect_physics_fallbacks(allow=True) as audit:
        score(
            [
                5.0,
                table.minimum_incident_energy_keV,
                20.0,
                table.maximum_incident_energy_keV,
                400.0,
            ],
            table,
            photon_block=block,
            chunk=1,
        )
    events = audit.events()
    assert [(e["model_requested"], e["model_used"]) for e in events] == [
        ("bremslib", "eedl"),
        ("eedl", "bethe-heitler"),
    ]
    for event in events:
        assert event["elements"] == ["C"]
        assert event["segments_total"] == 5
        assert event["segments_affected"] == 2
        assert event["fraction_segments_affected"] == 0.4
        assert event["energy_span_keV"] == [5.0, 400.0]


def test_covered_batches_contribute_to_chained_fallback_denominator(monkeypatch, table):
    monkeypatch.setattr(brem, "load_bremsstrahlung_cross_sections", missing_eedl)
    with pytest.warns(PhysicsFallbackWarning), collect_physics_fallbacks(allow=True) as audit:
        score([table.minimum_incident_energy_keV, 20.0, table.maximum_incident_energy_keV], table)
        score([5.0, 400.0], table)
    assert all(event["segments_total"] == 5 for event in audit.events())
    assert all(event["fraction_segments_affected"] == 0.4 for event in audit.events())


def test_eedl_coverage_gap_counts_only_rows_not_served_by_bremslib(monkeypatch, table):
    # Real EEDL covers 400 keV but not this far-above-table row; 20 keV stays BremsLib.
    with pytest.warns(PhysicsFallbackWarning), collect_physics_fallbacks(allow=True) as audit:
        score([20.0, 400.0, 1.0e9], table)
    events = {event["model_used"]: event for event in audit.events()}
    assert events["eedl"]["segments_affected"] == 2
    assert events["bethe-heitler"]["segments_affected"] == 1
    assert events["bethe-heitler"]["segments_total"] == 3
    assert events["bethe-heitler"]["energy_span_keV"] == [1.0e9, 1.0e9]


def test_counts_follow_scored_escape_rows_and_electron_selection(monkeypatch, table):
    monkeypatch.setattr(brem, "load_bremsstrahlung_cross_sections", missing_eedl)

    # Split the low-energy row at an escape boundary; exclude the final primary.
    def split(*_args, **_kwargs):
        return xp.array([0, 0, 1]), xp.array([0.5, 0.5, 1.0]), xp.zeros((3, 1)), xp.ones((3, 1))

    monkeypatch.setattr(brem, "segment_escape_paths", split)
    with pytest.warns(PhysicsFallbackWarning), collect_physics_fallbacks(allow=True) as audit:
        score([5.0, 20.0, 400.0], table, electron_limit=2)
    for event in audit.events():
        assert event["segments_total"] == 3
        assert event["segments_affected"] == 2
        assert event["fraction_segments_affected"] == pytest.approx(2 / 3)
        assert event["energy_span_keV"] == [5.0, 5.0]


def test_worker_metadata_survives_components_and_cache_payload(monkeypatch, tmp_path):
    from pyrite.checkpoints import _checkpoint_io, _checkpoint_store
    from pyrite.checkpoints.persistence import _cas_payload_from_record
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import case_driver
    from pyrite.results import store_result
    from pyrite.runs.scan import _physics_fallback_identity

    event = dict(
        model_requested="auto (bremslib-soft-hard)",
        model_used="uncoupled",
        scope="entire radiative transport",
        reason="test missing tables",
    )
    case = dict(
        name="fallback",
        E0_keV=40.0,
        theta_obs_rad=1.0,
        dtheta_obs_rad=0.0,
        domega_sr=1.0,
        composition=[("C", 0.1)],
        allow_physics_fallback=True,
        **_physics_fallback_identity("auto", "eedl", ["C"], [event]),
    )
    rows = segments([5.0, 20.0, 40.0])
    grid = np.array([100.0, 200.0, 400.0])

    def spectrum(_case, _tp, _timed):
        values = brem.mc_brem_spectrum(
            rows, grid, composition=case["composition"], cross_section_model="eedl"
        )
        return dict(
            E_grid=grid, spec=np.ones(3), brem=values, brem_wide=values, E_grid_brem=grid, eta=1.0
        )

    monkeypatch.setattr(runner, "_TIMING", False)
    monkeypatch.setattr(case_driver, "_spectrum_case_impl", spectrum)
    out = case_driver._spectrum_case(case, {"segs": rows})
    events = out["physics_fallbacks"]
    assert {event["model_used"] for event in events} == {"eedl", "uncoupled"}
    for event in events:
        assert event["segments_total"] == event["segments_affected"] == 3
        assert event["fraction_segments_affected"] == 1.0
        assert event["energy_span_keV"] == [5.0, 40.0]
    results = {}
    store_result(results, case, out)
    _checkpoint_store.save("fallback", tmp_path, results)
    for component in ("line", "brem"):
        path = tmp_path / "fallback" / f"{component}.h5"
        assert _checkpoint_io.load(str(path))["fallback"][40.0]["physics_fallbacks"] == events
    loaded = _checkpoint_store.load("fallback", tmp_path)["fallback"][40.0]
    assert _cas_payload_from_record(loaded)["physics_fallbacks"] == events


def test_opt_in_cases_cannot_share_strict_content_cache():
    from pyrite._physics_fallback import fallback_cases
    from pyrite.campaign.profiles import case_content_key

    case = dict(name="test", E0_keV=30.0, crystal="hopg")
    permitted = fallback_cases([case], allow=True)[0]
    assert case_content_key(case) != case_content_key(permitted)


@pytest.mark.parametrize("allow", [False, True])
def test_streamed_blocks_share_one_row_denominator(monkeypatch, table, allow):
    from types import SimpleNamespace

    from pyrite.montecarlo.runner import emission_adapters
    from pyrite.montecarlo.runner.emission_adapters import EmissionOutput
    from pyrite.montecarlo.runner.streamed_scoring import BlockSpectrumScorer

    grid = np.array([100.0, 200.0, 400.0])
    case = dict(
        name="stream",
        crystal="hopg",
        E0_keV=400.0,
        composition=[("C", 0.1)],
        allow_physics_fallback=allow,
    )
    inputs = dict(
        E_grid=grid,
        E_brem=grid,
        n_hat=np.array([0.0, 0.0, -1.0]),
        groove=None,
        Ne_lines=5,
        Ne_brem=5,
    )
    zeros = SimpleNamespace(evaluate=lambda batch, slot: EmissionOutput(slot, np.zeros(3)))

    def continuum(batch, slot):
        return EmissionOutput(
            slot,
            brem.mc_brem_spectrum(
                batch.segments,
                batch.energy_eV,
                composition=case["composition"],
                cross_section_model="bremslib",
                bremslib_tables={"C": table},
                electron_limit=5,
            ),
        )

    selection = SimpleNamespace(
        preflight=lambda *_a, **_k: None,
        cxr=zeros,
        characteristic=zeros,
        continuum=SimpleNamespace(evaluate=continuum),
    )
    monkeypatch.setattr(emission_adapters, "select_emission", lambda _case: selection)
    monkeypatch.setattr(brem, "load_bremsstrahlung_cross_sections", missing_eedl)
    scorer = BlockSpectrumScorer(case, inputs)
    scorer.add(
        segments([table.minimum_incident_energy_keV, 20.0, table.maximum_incident_energy_keV])
    )
    second = segments([5.0, 400.0])
    second["elec_id"] += 3
    if not allow:
        with pytest.raises(PhysicsFallbackWarning):
            scorer.add(second)
        assert scorer.fallbacks.events() == []  # failed block is rolled back
        assert scorer.n_segments == 3
        return
    with pytest.warns(PhysicsFallbackWarning):
        scorer.add(second)
    out = scorer.result(Ne=5, n_backscattered=0, n_missed=0, line_coverage={})
    for event in out["physics_fallbacks"]:
        assert event["segments_total"] == 5
        assert event["segments_affected"] == 2
        assert event["fraction_segments_affected"] == 0.4
