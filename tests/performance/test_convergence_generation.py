"""Retained generator evidence must come from fresh exact-request evaluations."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite._util.hashing import canonical_json_digest, file_sha256
from pyrite.perf._benchmark_tables import table_record
from pyrite.xsgen.elsepa.deck import ElsepaDeck
from pyrite.xsgen.store import ElementTarget, TableRequest, store

HERE = Path(__file__).resolve().parents[2] / "checks/geant4_performance"
RUNNER = runpy.run_path(str(HERE / "generate_convergence_table.py"))


def planned(tmp_path):
    deck = ElsepaDeck.free_atom(14, (100.0, 10000.0))
    request = TableRequest(
        "elsepa", "source-pin", ElementTarget(14), "elastic_dcs", deck.model_record()
    )
    table = store(
        request,
        {"energy_eV": np.array(deck.energies_ev), "mu": np.array([0.0, 1.0])},
        root=tmp_path / "native",
    )
    refined = request.record() | dict(
        model=request.model | dict(energies_eV=[100.0, 1000.0, 10000.0])
    )
    key = canonical_json_digest(refined)
    plan = dict(
        schema="pyrite.geant4-performance.elsepa-refinement-plan.v1",
        baselines=[table_record(table)],
        requests=[dict(key=key, request=refined, baseline_key=table.key, level=1)],
    )
    return plan, key, table


def test_admission_preserves_exact_nested_request(tmp_path):
    plan, key, _ = planned(tmp_path)
    request, deck, _ = RUNNER["admit"](plan, key)
    assert request.key == key
    assert deck.energies_ev == (100.0, 1000.0, 10000.0)
    assert deck.render() != ElsepaDeck.free_atom(14, (100.0, 10000.0)).render()


@pytest.mark.parametrize("damage", ["grid", "physics", "key", "duplicate", "baseline"])
def test_changed_plan_refused(tmp_path, damage):
    plan, key, _ = planned(tmp_path)
    row = plan["requests"][0]
    if damage == "grid":
        row["request"]["model"]["energies_eV"][1] = 2000.0
    elif damage == "physics":
        row["request"]["model"]["exchange_model"] = 0
    elif damage == "key":
        row["key"] = "changed"
    elif damage == "duplicate":
        plan["requests"].append(copy.deepcopy(row))
    else:
        plan["baselines"][0]["request"]["code_version"] = "changed"
    with pytest.raises(ValueError):
        RUNNER["admit"](plan, key)


def test_actual_frozen_rows_admit_including_muffin_tin():
    plan = json.loads((HERE / "elsepa-refinement-plan-v1.json").read_text())
    solids = 0
    for row in plan["requests"]:
        request, deck, _ = RUNNER["admit"](plan, row["key"])
        assert request.key == row["key"]
        assert deck.is_muffin_tin == (row["request"]["target"]["kind"] == "material")
        solids += deck.is_muffin_tin
    assert solids == 3


def muffin_row(plan):
    return next(r for r in plan["requests"] if r["request"]["model"]["mode"] == "muffin_tin")


@pytest.mark.parametrize("damage", ["radius", "composition", "orphan", "mode"])
def test_changed_muffin_tin_request_refused(damage):
    plan = json.loads((HERE / "elsepa-refinement-plan-v1.json").read_text())
    row = muffin_row(plan)
    if damage == "radius":
        for baseline in plan["baselines"]:
            if baseline["key"] == row["baseline_key"]:
                baseline["request"]["model"]["muffin_tin_radius_cm"] *= 1.01
    elif damage == "composition":
        for scene in plan["scenes"]:
            scene["composition"][0][1] *= 1.01
    elif damage == "orphan":
        for scene in plan["scenes"]:
            scene["native_keys"] = [k for k in scene["native_keys"] if k != row["baseline_key"]]
    else:
        row["request"]["model"]["mode"] = "free_atom"
    with pytest.raises(ValueError):
        RUNNER["admit"](plan, row["key"])


def setup_run(tmp_path, monkeypatch):
    plan, key, native = planned(tmp_path)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    # runpy functions retain their original globals; patch those explicitly.
    main = RUNNER["generate"].__globals__
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "elscata.f").write_text("retained generator source")
    database = source_root / "database"
    database.mkdir()
    (database / "atom.dat").write_text("retained atomic input")
    source = SimpleNamespace(
        root=source_root,
        origin="test",
        data_dirs={"database": database},
        spec=SimpleNamespace(
            digest_sources=["elscata.f"], programs={"elscata": object()}, upstream="test"
        ),
    )
    monkeypatch.setitem(main, "require", lambda _: native)
    monkeypatch.setitem(main, "resolve_source", lambda _: source)
    monkeypatch.setitem(main, "source_digest", lambda _: "source-pin")
    monkeypatch.setitem(main, "find_toolchain", lambda: SimpleNamespace(version="test compiler"))
    monkeypatch.setitem(main, "build_command", lambda s, p, t, dest: ["fake-compiler", str(dest)])

    def compile_fake(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"fresh binary")
        return SimpleNamespace(returncode=0, stdout=b"build log", stderr=b"")

    monkeypatch.setattr(main["subprocess"], "run", compile_fake)
    calls = []

    def run_fake(binary, **kwargs):
        calls.append(kwargs["stdin_text"])
        return SimpleNamespace(
            stdout="panel log",
            stderr="",
            outputs={name: b"fresh raw panel" for name in kwargs["outputs"]},
        )

    monkeypatch.setitem(main, "run_program", run_fake)
    energies = iter((100.0, 1000.0, 10000.0))
    monkeypatch.setitem(main, "parse_dcs", lambda raw: SimpleNamespace(energy_ev=next(energies)))
    monkeypatch.setitem(
        main,
        "table_arrays",
        lambda p, energies_ev: dict(
            energy_eV=np.array(energies_ev), mu=np.array([0.0, 1.0]), transport1_cm2=np.ones(3)
        ),
    )
    monkeypatch.setitem(main, "check_transport_scaling", lambda *a, **k: None)
    return path, key, calls, main


def test_success_retains_fresh_decks_raw_inputs_binary_and_exact_grid(tmp_path, monkeypatch):
    plan, key, calls, _ = setup_run(tmp_path, monkeypatch)
    out = tmp_path / "evidence"
    result = RUNNER["generate"](plan, key, out)
    assert len(calls) == len(set(calls)) == 3
    assert not result["headline"] and result["numerical_gate"] == "unmeasured"
    assert result["energy_nodes"] == 3
    assert result["actual_panel_energies_eV"] == [100.0, 1000.0, 10000.0]
    for name, digest in result["files_sha256"].items():
        assert file_sha256(out / name) == digest
    assert (out / "panels/0001/deck.in").read_text() == calls[1]
    assert (out / "source/elscata.f").is_file()
    assert (out / "data/database/atom.dat").is_file()
    assert (out / "generation.json").is_file()
    assert (out / "tables" / f"{key}.npz").is_file()
    with pytest.raises(FileExistsError):
        RUNNER["generate"](plan, key, out)
    assert len(calls) == 3


@pytest.mark.parametrize("failure", ["budget", "source", "panel", "angular", "build"])
def test_failures_never_publish_completion(tmp_path, monkeypatch, failure):
    plan, key, calls, main = setup_run(tmp_path, monkeypatch)
    out = tmp_path / "evidence"
    kwargs = {}
    if failure == "budget":
        kwargs["max_bytes"] = 1
    elif failure == "source":
        monkeypatch.setitem(main, "source_digest", lambda _: "wrong")
    elif failure == "build":
        monkeypatch.setattr(
            main["subprocess"],
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout=b"", stderr=b"compiler failed"),
        )
    elif failure == "panel":

        def fail(*a, **kw):
            raise RuntimeError("generator failed")

        monkeypatch.setitem(main, "run_program", fail)
    else:
        monkeypatch.setitem(
            main, "table_arrays", lambda *a, **k: dict(mu=np.array([0.0, 0.5, 1.0]))
        )
    with pytest.raises((ValueError, RuntimeError)):
        RUNNER["generate"](plan, key, out, **kwargs)
    assert not (out / "generation.json").exists()
    assert not (out / "tables").exists()
