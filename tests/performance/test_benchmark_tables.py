"""Invocation-local refinements preserve physical inputs and reject stale tables."""

import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.perf._benchmark_tables import (
    checked_table,
    nested_energy_grid,
    select_elastic_tables,
    select_stopping_table,
    table_coverage,
)
from pyrite.xsgen.elsepa.catalog import ElementElastic
from pyrite.xsgen.store import ElementTarget, MaterialTarget, TableRequest, store


def elastic(tmp_path, energies=(1000.0, 10000.0, 100000.0), **model):
    request = TableRequest(
        "elsepa",
        "source-pin",
        ElementTarget(14),
        "elastic_dcs",
        dict(mode="free_atom", energies_eV=list(energies), **model),
    )
    arrays = dict(
        energy_eV=np.asarray(energies),
        mu=np.array([0.0, 0.5, 1.0]),
        dcs_cm2_sr=np.ones((len(energies), 3)),
        total_elastic_cm2=np.ones(len(energies)),
        transport1_cm2=np.ones(len(energies)),
    )
    table = store(request, arrays, root=tmp_path)
    return ElementElastic("Si", 14, (table,), arrays)


def test_refinement_inserts_nested_log_midpoints_and_preserves_native_nodes():
    native = [100.0, 10000.0]
    assert nested_energy_grid(native, 1) == (100.0, 1000.0, 10000.0)
    fine = np.asarray(nested_energy_grid(native, 3))
    assert len(fine) == 9
    np.testing.assert_array_equal(fine[::4], [100.0, 1000.0, 10000.0])


def test_table_selection_is_local_and_retains_every_native_node(tmp_path):
    native = elastic(tmp_path / "native")
    refined = elastic(tmp_path / "refined", nested_energy_grid([1000.0, 10000.0, 100000.0], 1))
    selected = select_elastic_tables((native,), 1, [refined.tables[0].path])
    np.testing.assert_array_equal(selected[0].arrays["energy_eV"][::2], native.arrays["energy_eV"])
    assert selected[0].tables[0].key == refined.tables[0].key
    assert native.tables[0].path.exists()


def test_wrong_physics_is_not_an_energy_refinement(tmp_path):
    native = elastic(tmp_path / "native", exchange_model=1)
    wrong = elastic(
        tmp_path / "wrong", nested_energy_grid([1000.0, 10000.0, 100000.0], 1), exchange_model=2
    )
    with pytest.raises(ValueError, match="unused|request"):
        select_elastic_tables((native,), 1, [wrong.tables[0].path])


def test_changed_manifest_is_rejected_before_payload_admission(tmp_path):
    native = elastic(tmp_path)
    bad = replace(native.tables[0], manifest=dict(native.tables[0].manifest, code_version="other"))
    with pytest.raises(ValueError, match="manifest"):
        checked_table(bad)


def test_changed_payload_is_rejected(tmp_path):
    native = elastic(tmp_path)
    np.savez(native.tables[0].path, energy_eV=[1.0, 2.0])
    with pytest.raises(RuntimeError, match="corrupt|payload"):
        checked_table(native.tables[0])


@pytest.mark.parametrize("energies", [[0.0, 1.0], [2.0, 1.0], [1.0, float("nan")], [1.0]])
def test_invalid_grids_never_produce_refinement(energies):
    with pytest.raises(ValueError):
        nested_energy_grid(energies, 1)


def test_coverage_never_extrapolates():
    assert table_coverage([1000.0, 10000.0], 1.0, 10.0)["nodes"] == 2
    with pytest.raises(ValueError, match="coverage"):
        table_coverage([1000.0, 10000.0], 0.5, 10.0)


def test_sbethe_native_grid_cannot_be_refined_by_resampling(tmp_path):
    request = TableRequest(
        "sbethe",
        "source-pin",
        MaterialTarget("Si", "material-identity"),
        "collision_stopping",
        dict(projectile="electron"),
    )
    baseline = store(
        request, dict(stopping_energy_eV=np.array([1000.0, 10000.0])), root=tmp_path / "native"
    )
    denser = store(
        request,
        dict(stopping_energy_eV=np.array([1000.0, 3000.0, 10000.0])),
        root=tmp_path / "denser",
    )
    with pytest.raises(ValueError, match="native grid"):
        select_stopping_table(baseline, denser.path)


HARNESS = Path(__file__).resolve().parents[2] / "checks/geant4_performance/pyrite_bench.py"


def test_cutoffs_are_invocation_local_and_keep_scoring_bounds():
    bounds = runpy.run_path(str(HARNESS))["numerical_bounds"]
    nominal = dict(electron_cutoff_keV=10.0, hard_photon_cutoff_eV=1000.0, width_mm=10.0)
    refined = bounds(nominal, 300.0, 1.25, 125.0)
    assert refined == dict(electron_cutoff_keV=1.25, hard_photon_cutoff_eV=125.0, width_mm=10.0)
    assert nominal["electron_cutoff_keV"] == 10.0


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("nan"), float("inf"), 300.0])
def test_invalid_cutoff_fails_without_cuda(cutoff):
    bounds = runpy.run_path(str(HARNESS))["numerical_bounds"]
    with pytest.raises(ValueError, match="cutoff"):
        bounds(dict(electron_cutoff_keV=10.0, hard_photon_cutoff_eV=1000.0), 300.0, cutoff)


def test_table_only_preflight_checks_real_payloads_without_cuda(tmp_path, monkeypatch):
    from pyrite.perf import _benchmark_tables
    from pyrite.xsgen.bremslib import tables as bremslib
    from pyrite.xsgen.elsepa import catalog as elsepa
    from pyrite.xsgen.sbethe import catalog as sbethe

    entry = elastic(tmp_path / "elastic", energies=(1000.0, 10000.0, 1000000.0))
    stopping = store(
        TableRequest(
            "sbethe", "source-pin", MaterialTarget("Si", "identity"), "collision_stopping"
        ),
        dict(stopping_energy_eV=np.array([1000.0, 1000000.0])),
        root=tmp_path / "stop",
    )
    brem = store(
        TableRequest("bremslib", "source-pin", ElementTarget(14), "bremsstrahlung"),
        dict(t1_MeV=np.array([0.001, 1.0])),
        root=tmp_path / "brem",
    )
    prepared = SimpleNamespace(
        key=brem.key, digest=brem.digest, incident_energy_keV=np.array([1.0, 1000.0])
    )
    monkeypatch.setattr(elsepa, "resolve_layer_tables", lambda _: (entry,))
    monkeypatch.setattr(sbethe, "resolve_composition_table", lambda *_: stopping)
    monkeypatch.setattr(bremslib, "load_bremsstrahlung_tables", lambda _: {"Si": prepared})
    monkeypatch.setattr(_benchmark_tables, "resolve", lambda key: brem if key == brem.key else None)
    monkeypatch.setitem(sys.modules, "cupy", None)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(HARNESS),
            "si-300keV-100um",
            "--histories",
            "2",
            "--seed",
            "301",
            "--tables-only",
            "--moment-level",
            "3",
            "--electron-cutoff-kev",
            "1.25",
            "--output",
            str(output),
        ],
    )
    runpy.run_path(str(HARNESS))["main"]()
    record = json.loads(output.read_text())
    assert record["purpose"] == "accuracy-only" and record["numerical_gate"] == "unmeasured"
    assert record["moment_policy"]["rtol"] == 2.5e-5
    assert record["tables"]["sbethe"]["key"] == stopping.key
    assert record["tables"]["bremslib"][0]["coverage"]["nodes"] == 2
    assert record["bounds"]["electron_cutoff_keV"] == 1.25
    assert not output.with_suffix(".npz").exists()


@pytest.mark.parametrize(
    "override",
    [
        dict(elsepa_level=1),
        dict(sbethe_table=Path("local.npz")),
        dict(radiative_moments="exact"),
        dict(moment_level=0),
        dict(moment_level=3),
        dict(transport_precision="fp64"),
        dict(mixed_c1=0.025),
        dict(cone_energy_order=8),
        dict(capture_tracks=Path("new-track-capture")),
        dict(transport_core="per-electron"),
    ],
)
def test_numerical_settings_cannot_enter_frozen_nominal_timing(override):
    args = dict(
        accuracy_only=False,
        elsepa_level=0,
        elsepa_table=[],
        sbethe_table=None,
        radiative_moments="table",
        moment_level=None,
        max_de_frac=0.05,
        mixed_c1=None,
        inelastic_cutoff_ev=None,
        transport_precision="mixed",
        transport_core="cuda",
        cone_order=8,
        cone_rtol=1e-3,
    )
    args.update(override)
    study = runpy.run_path(str(HARNESS))["numerical_study"]
    assert study(SimpleNamespace(**args), {}, {})


def test_nominal_selection_preserves_original_request_and_arrays(tmp_path):
    native = elastic(tmp_path, energies=(1000, 10000, 100000))
    selected = select_elastic_tables((native,))
    assert selected[0].tables[0].key == native.tables[0].key
    for name, values in native.arrays.items():
        np.testing.assert_array_equal(selected[0].arrays[name], values)


def test_refinement_plan_keys_match_exact_admission_without_generating(tmp_path):
    from pyrite.perf._benchmark_tables import elastic_refinement_plan

    native = elastic(tmp_path / "native", energies=(1000, 10000, 100000), spin="preserved")
    before = native.tables[0].path.read_bytes()
    plan = elastic_refinement_plan((native, native))
    assert len(plan["baselines"]) == 1 and len(plan["requests"]) == 3
    assert not plan["generation_performed"] and plan["numerical_gate"] == "unmeasured"
    for record in plan["requests"]:
        model = record["request"]["model"]
        assert model["spin"] == "preserved"
        grid = model["energies_eV"]
        assert len(grid) == 2 ** record["level"] * 2 + 1
        assert grid[0] == 1000 and grid[-1] == 100000
        generated = elastic(tmp_path / f"level-{record['level']}", energies=grid, spin="preserved")
        assert generated.tables[0].key == record["key"]
        admitted = select_elastic_tables((native,), record["level"], [generated.tables[0].path])
        assert admitted[0].tables[0].key == record["key"]
        assert record["generated_payload_sha256"] is None
        assert record["generator_evidence_sha256"] is None
    assert native.tables[0].path.read_bytes() == before


@pytest.mark.parametrize("levels", [(), (0,), (4,), (True,), (1, 1)])
def test_refinement_plan_rejects_invalid_levels(tmp_path, levels):
    from pyrite.perf._benchmark_tables import elastic_refinement_plan

    with pytest.raises(ValueError, match="unique"):
        elastic_refinement_plan((elastic(tmp_path),), levels)


def test_retained_elsepa_plan_has_exact_nested_requests_and_no_generated_evidence():
    from pyrite._util.hashing import canonical_json_digest

    plan = json.loads(HARNESS.with_name("elsepa-refinement-plan-v1.json").read_text())
    assert not plan["generation_performed"] and not plan["submission_ready"]
    assert len(plan["scenes"]) == 6 and plan["materials"]["W"]["density_g_cm3"] == 19.3
    assert len(plan["baselines"]) == 3 and len(plan["requests"]) == 9
    native = {row["key"]: row for row in plan["baselines"]}
    assert {row["request"]["model"]["mode"] for row in native.values()} == {
        "free_atom",
        "muffin_tin",
    }
    for row in plan["requests"]:
        assert row["key"] == canonical_json_digest(row["request"])
        baseline = native[row["baseline_key"]]["request"]
        expected = baseline | dict(
            model=baseline["model"]
            | dict(
                energies_eV=list(nested_energy_grid(baseline["model"]["energies_eV"], row["level"]))
            )
        )
        assert row["request"] == expected
        assert row["generated_payload_sha256"] is None and row["generator_evidence_sha256"] is None
        assert row["status"] == "requires-independent-generation"


def test_energy_refinement_cannot_change_the_angular_mesh(tmp_path):
    native = elastic(tmp_path / "native")
    refined = elastic(tmp_path / "fine", nested_energy_grid([1000.0, 10000.0, 100000.0], 1))
    request = TableRequest(
        "elsepa",
        "source-pin",
        ElementTarget(14),
        "elastic_dcs",
        refined.tables[0].manifest["model"],
    )
    arrays = dict(refined.arrays, mu=np.array([0.0, 0.25, 1.0]))
    changed = store(request, arrays, root=tmp_path / "changed")
    with pytest.raises(ValueError, match="angular grid"):
        select_elastic_tables((native,), 1, [changed.path])


def test_muffin_tin_refinement_keeps_the_one_mev_crossover(tmp_path):
    native_free = elastic(tmp_path / "native-free", energies=(1000.0, 1e6, 1e7))
    fine_free = elastic(tmp_path / "fine-free", nested_energy_grid([1000.0, 1e6, 1e7], 1))
    native_grid = [1000.0, 1e5, 1e6]
    tables = []
    for name, grid in [("native-mt", native_grid), ("fine-mt", nested_energy_grid(native_grid, 1))]:
        request = TableRequest(
            "elsepa",
            "source-pin",
            MaterialTarget("Si", "identity"),
            "elastic_dcs",
            dict(mode="muffin_tin", energies_eV=list(grid)),
        )
        arrays = dict(
            energy_eV=np.asarray(grid),
            mu=np.array([0.0, 0.5, 1.0]),
            dcs_cm2_sr=np.full((len(grid), 3), 7.0),
            total_elastic_cm2=np.full(len(grid), 7.0),
            transport1_cm2=np.full(len(grid), 7.0),
        )
        tables.append(store(request, arrays, root=tmp_path / name))
    entry = replace(native_free, tables=(native_free.tables[0], tables[0]))
    selected = select_elastic_tables((entry,), 1, [fine_free.tables[0].path, tables[1].path])[0]
    low = selected.arrays["energy_eV"] <= 1e6
    assert selected.arrays["energy_eV"][low][-1] == 1e6
    assert np.all(selected.arrays["total_elastic_cm2"][low] == 7.0)
    assert np.all(selected.arrays["total_elastic_cm2"][~low] == 1.0)


@pytest.mark.parametrize("level", range(4))
def test_moment_ladder_has_frozen_component_budgets(level):
    policy = runpy.run_path(str(HARNESS))["numerical_moment_policy"](level, "table")
    assert policy.rtol == 2e-4 / 2**level
    assert policy.atol == (1e-10 / 2**level, 1e-14 / 2**level, 1e-10 / 2**level)
    assert policy.max_nodes == 32768 and policy.held_out_count == 8191


def test_moment_ladder_preserves_nominal_and_refuses_exact_override():
    factory = runpy.run_path(str(HARNESS))["numerical_moment_policy"]
    assert factory(None, "table") is None
    assert factory(None, "exact") is None
    with pytest.raises(ValueError, match="requires"):
        factory(1, "exact")


@pytest.mark.parametrize(
    "extra",
    [
        [],  # default mixed precision exists only on the CUDA core
        ["--transport-precision", "fp64", "--estimator", "continuous"],
        ["--transport-precision", "fp64", "--spectrum-step-ev", "25"],
    ],
)
def test_cpu_core_bench_is_limited_to_discrete_fp64_transport(monkeypatch, tmp_path, capsys, extra):
    argv = [str(HARNESS), "si-300keV-100um", "--histories", "2", "--seed", "301"]
    argv += ["--transport-core", "per-electron"]
    argv += ["--output", str(tmp_path / "x.json"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        runpy.run_path(str(HARNESS))["main"]()
    assert "per-electron supports discrete fp64" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()
