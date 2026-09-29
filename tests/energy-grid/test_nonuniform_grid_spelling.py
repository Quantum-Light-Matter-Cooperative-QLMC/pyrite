"""Reporting paths for the catalog's nonuniform grid spellings (#153).

``materials/_catalog_decode.py`` has always accepted ``values``/``arange``/
``linspace``/``logspace``, but every reader in the CLI reporting path indexed
the uniform spelling: a ``logspace`` brem override resolved to ``None`` and
dropped out of its own report, and a ``values`` line row raised
``KeyError: 'linspace'``. These pin the honest rendering and, just as
importantly, that the uniform renderings did not move.
"""

import tomllib

import pytest

from pyrite.console import json as cli_json
from pyrite.energy_grid import apply
from tests.helpers.energy_grid_catalog import BASE_TOML

NONUNIFORM_TOML = """schema_version = 1

[profiles.standard]
energy_keV = { values = [30.0] }

[profiles.standard.overrides.graded]
E_grid_brem = { logspace = { start = 1.7, stop = 4.5, num = 5 } }

[profiles.standard.overrides.listed]
E_grid_brem = { values = [50.0, 80.0, 200.0, 900.0] }

[energy_grids.graded]
line_by_energy = [
  { energy_keV = 30.0, grid = { values = [10.0, 20.0, 45.0, 120.0] }, source = "derived" },
]

[materials.graded]
display_name = "Graded"

[materials.listed]
display_name = "Listed"
"""


@pytest.fixture
def nonuniform_catalog(tmp_path):
    path = tmp_path / "materials.toml"
    path.write_text(NONUNIFORM_TOML)
    return path


def test_logspace_brem_override_resolves_instead_of_vanishing():
    band = apply.effective_brem(tomllib.loads(NONUNIFORM_TOML), "graded")
    assert band is not None, "a logspace override used to resolve to None"
    assert band["kind"] == "logspace"
    # start/stop are exponents; the reported band is the resolved coordinates.
    assert band["nodes"].size == 5
    assert band["nodes"][0] == pytest.approx(10**1.7)
    assert band["nodes"][-1] == pytest.approx(10**4.5)


def test_values_brem_override_resolves_without_payload_keys():
    band = apply.effective_brem(tomllib.loads(NONUNIFORM_TOML), "listed")
    assert band["kind"] == "values"
    assert list(band["nodes"]) == [50.0, 80.0, 200.0, 900.0]


def test_show_reports_nonuniform_brem_and_line_grids(nonuniform_catalog):
    out = apply.show("graded", catalog_path=nonuniform_catalog)
    assert "brem grid: [50.1187, 31622.8] eV x 5 pts (logspace)" in out
    assert "line grid @ 30 keV: [10, 120] eV x 4 pts (values)" in out

    listed = apply.show("listed", "brem", catalog_path=nonuniform_catalog)
    assert "brem grid: [50, 900] eV x 4 pts (values)" in listed


def test_uniform_show_rendering_is_unchanged(tmp_path):
    path = tmp_path / "materials.toml"
    path.write_text(BASE_TOML)
    out = apply.show("hopg", catalog_path=path)
    assert "line grid @ 30 keV: [10, 2600] eV x 864 pts" in out
    assert "brem grid: [50, 136500] eV step 25" in out  # arange is still floored
    assert "(linspace)" not in out and "(arange)" not in out


def test_nonuniform_brem_is_not_floored_like_a_uniform_band(nonuniform_catalog):
    """``build_cases`` passes a nonuniform grid through; the report must agree.

    Flooring here would print a band no case is ever evaluated over.
    """
    _, _, brem_by_material, _ = apply.resolved_show_inputs(catalog_path=nonuniform_catalog)
    assert brem_by_material["listed"]["nodes"][0] == 50.0
    assert "start" not in brem_by_material["listed"]


def test_json_show_keeps_uniform_payloads_and_describes_graded_ones(nonuniform_catalog):
    raw, energy_grids, brem_by_material, refs = apply.resolved_show_inputs(
        catalog_path=nonuniform_catalog
    )
    result = cli_json.line_grid_show(
        raw["materials"], energy_grids, brem_by_material, {}, artifact_refs=refs
    )
    by_material = {item["material"]: item for item in result.payload["materials"]}
    assert result.errors == ()

    graded = by_material["graded"]
    assert graded["brem_grid"]["kind"] == "logspace"
    assert graded["brem_grid"]["points"] == 5
    # The descriptor is echoed verbatim so the report round-trips.
    assert graded["brem_grid"]["descriptor"] == {"start": 1.7, "stop": 4.5, "num": 5}
    line = graded["line_grids"][0]["grid"]
    assert line["kind"] == "values"
    assert line["descriptor"] == [10.0, 20.0, 45.0, 120.0]
    assert (line["start_eV"], line["stop_eV"], line["points"]) == (10.0, 120.0, 4)


def test_json_show_uniform_payload_shape_is_unchanged(tmp_path):
    path = tmp_path / "materials.toml"
    path.write_text(BASE_TOML)
    raw, energy_grids, brem_by_material, refs = apply.resolved_show_inputs(catalog_path=path)
    result = cli_json.line_grid_show(
        raw["materials"], energy_grids, brem_by_material, {}, artifact_refs=refs
    )
    entry = result.payload["materials"][0]
    assert entry["brem_grid"]["kind"] == "arange"
    assert "descriptor" not in entry["brem_grid"]
    assert entry["line_grids"][0]["grid"] == {
        "kind": "linspace",
        "start_eV": 10.0,
        "stop_eV": 2600.0,
        "points": 864,
        "endpoint": True,
    }


def test_merge_refuses_a_line_row_it_cannot_respell(nonuniform_catalog):
    """The writer stores linspace rows only, so say so by name."""
    combined = {
        "graded": {
            "line_rows": [{"energy_keV": 30.0, "start_eV": 10.0, "stop_eV": 900.0, "num": 300}],
            "brem": {"stop_eV": 40000.0, "step_eV": 25.0, "raw_eV": 30000.0},
        }
    }
    with pytest.raises(ValueError, match=r"line row @ 30 keV is a values grid"):
        apply.apply_bounds(NONUNIFORM_TOML, combined)


# --- write side -------------------------------------------------------------

WRITABLE_TOML = """schema_version = 1

[profiles.standard]
energy_keV = { values = [30.0] }
E_grid_brem = { arange = { start = 0.0, stop = 30000.0, step = 25.0 } }

[profiles.pinned]
energy_keV = { values = [30.0] }
energy_grid_refs = { graded = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" }

[materials.graded]
display_name = "Graded"
"""


@pytest.fixture
def writable_catalog(tmp_path, monkeypatch):
    path = tmp_path / "materials.toml"
    path.write_text(WRITABLE_TOML)
    monkeypatch.setattr(apply, "_validate_catalog_text", lambda *a, **k: None)
    monkeypatch.setattr(apply, "_warn_stale_golden", lambda *a, **k: None)
    monkeypatch.setattr(apply._provenance, "set_brem", lambda *a, **k: None)
    return path


def test_set_brem_geometric_writes_exact_nodes(writable_catalog, monkeypatch):
    """The stored row must decode to the builder's own coordinates.

    This is why the writer emits ``values`` and not ``logspace``: a logspace
    descriptor hands the floor endpoint back to ``np.logspace`` on decode and
    can round it below the floor, which ``geometric_continuum_grid`` refuses.
    """
    import numpy as np

    from pyrite.energy_grid import floor

    nodes = np.geomspace(40.0, 4000.0, 9)
    monkeypatch.setattr(floor, "geometric_continuum_grid", lambda *a, **k: nodes)

    band = apply.set_brem_geometric("graded", 4000.0, 9, catalog_path=writable_catalog)
    assert band == "[40, 4000] eV x 9 pts (values)"

    stored = tomllib.loads(writable_catalog.read_text())
    written = stored["profiles"]["standard"]["overrides"]["graded"]["E_grid_brem"]["values"]
    assert written == [float(node) for node in nodes]

    # and it round-trips through the reader added by the first slice
    resolved = apply.effective_brem(stored, "graded")
    assert np.array_equal(resolved["nodes"], nodes)


def test_set_brem_geometric_refuses_when_an_artifact_pins_the_material(writable_catalog):
    """An artifact would overwrite the row at load, so refuse instead of writing it."""
    with pytest.raises(ValueError, match=r"pins artifact sha256:0123"):
        apply.set_brem_geometric(
            "graded", 4000.0, 9, profile="pinned", catalog_path=writable_catalog
        )
    # nothing written
    assert "overrides" not in tomllib.loads(writable_catalog.read_text())["profiles"]["pinned"]


def test_set_brem_geometric_rejects_an_unknown_material(writable_catalog):
    with pytest.raises(ValueError, match="unknown material: nope"):
        apply.set_brem_geometric("nope", 4000.0, 9, catalog_path=writable_catalog)


def test_set_brem_geometric_first_node_is_the_derived_floor(writable_catalog, monkeypatch):
    """Not a step multiple: a geometric grid can sit exactly on the floor."""
    from pyrite.energy_grid import floor

    monkeypatch.setattr(floor, "photon_continuum_floor_eV", lambda *a, **k: 30.6614)
    band = apply.set_brem_geometric("graded", 40000.0, 12, catalog_path=writable_catalog)
    assert band.startswith("[30.6614, 40000] eV x 12 pts")
