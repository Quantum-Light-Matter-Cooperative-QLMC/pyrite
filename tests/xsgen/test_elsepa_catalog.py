"""Muffin-tin decks and the catalog elastic-table policy (issue #89)."""

import numpy as np
import pytest

from pyrite.xsgen import TableNotFoundError
from pyrite.xsgen.elsepa import ElsepaDeck
from pyrite.xsgen.elsepa.catalog import (
    ELASTIC_CEILING_EV,
    ELASTIC_FLOOR_EV,
    MUFFIN_TIN_CEILING_EV,
    PRODUCTION_ENERGIES_EV,
    elemental_solid,
    elemental_solid_for_composition,
    joined_arrays,
    nearest_neighbour_distance_ang,
    resolve_layer_tables,
)


def _fields(deck: ElsepaDeck) -> dict[str, str]:
    return {line[:6].strip(): line[7:] for line in deck.render().splitlines()}


def test_muffin_tin_deck_renders_radius_absorption_and_no_factorization():
    deck = ElsepaDeck.muffin_tin(14, [1.0e3], radius_cm=1.17582e-8)
    fields = _fields(deck)

    assert fields["MUFFIN"] == "1"
    assert float(fields["RMUF"]) == pytest.approx(1.17582e-8)
    assert fields["MABS"] == "2"
    assert float(fields["VABSA"]) == pytest.approx(0.75)
    assert "VABSD" not in fields  # ELSEPA's tabulated excitation energy
    assert fields["IHEF"] == "0"
    for line in deck.render().splitlines():
        assert line[6] == " " and len(line[7:]) <= 12


def test_muffin_tin_and_free_atom_records_differ_and_free_atom_keeps_its_spelling():
    free = ElsepaDeck.free_atom(14, [1.0e3]).model_record()
    muffin = ElsepaDeck.muffin_tin(14, [1.0e3], radius_cm=1.2e-8).model_record()

    assert free["mode"] == "free_atom"
    assert "muffin_tin_radius_cm" not in free
    assert muffin["mode"] == "muffin_tin"
    assert muffin["muffin_tin_radius_cm"] == pytest.approx(1.2e-8)
    assert muffin["high_energy_factorization"] == 0


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"muffin_tin_radius_cm": 0.0}, "radius"),
        ({"muffin_tin_radius_cm": 1e-8, "high_energy_factorization": 1}, "factorization"),
        ({"absorption_strength": 0.75}, "absorption model"),
    ],
)
def test_deck_rejects_inconsistent_muffin_tin_options(options, match):
    with pytest.raises(ValueError, match=match):
        ElsepaDeck(z=14, energies_ev=(1.0e3,), **options)


def test_production_grid_spans_the_declared_coverage_with_unique_output_names():
    deck = ElsepaDeck.free_atom(14, PRODUCTION_ENERGIES_EV)

    assert PRODUCTION_ENERGIES_EV[0] == ELASTIC_FLOOR_EV
    assert PRODUCTION_ENERGIES_EV[-1] == ELASTIC_CEILING_EV
    assert MUFFIN_TIN_CEILING_EV in PRODUCTION_ENERGIES_EV
    assert len(set(deck.output_names)) == len(PRODUCTION_ENERGIES_EV)


def test_nearest_neighbour_distance_matches_diamond_cubic_geometry():
    a = 5.4309
    basis = [
        ("Si", np.array(p))
        for p in [
            (0, 0, 0),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
            (0.25, 0.25, 0.25),
            (0.75, 0.75, 0.25),
            (0.75, 0.25, 0.75),
            (0.25, 0.75, 0.75),
        ]
    ]

    d = nearest_neighbour_distance_ang({"system": "cubic", "a": a}, basis)

    assert d == pytest.approx(a * np.sqrt(3.0) / 4.0, rel=1e-12)


@pytest.mark.parametrize(
    ("key", "element", "d_nn_ang"),
    [
        # ELSEPA's own tabulated DNNEL: Si 2.350 A, C (diamond) 1.540 A.
        ("silicon", "Si", 2.350),
        ("diamond", "C", 1.540),
        # Graphite in-plane C-C bond.
        ("hopg", "C", 1.421),
    ],
)
def test_elementary_catalog_crystals_get_half_the_nearest_neighbour_distance(
    key, element, d_nn_ang
):
    solid = elemental_solid(key)

    assert solid is not None
    assert solid.element == element
    assert 2.0 * solid.radius_cm * 1e8 == pytest.approx(d_nn_ang, abs=6e-3)


@pytest.mark.parametrize("key", ["mos2", "sapphire", "sio2", None, "not-a-material"])
def test_compounds_media_and_unnamed_layers_have_no_muffin_tin_table(key):
    assert elemental_solid(key) is None


def _table(energies, total, mu):
    energies = np.asarray(energies, dtype=float)
    return {
        "mu": mu,
        "energy_eV": energies,
        "total_elastic_cm2": np.asarray(total, dtype=float),
        "transport1_cm2": np.asarray(total, dtype=float) / 10.0,
        "dcs_cm2_sr": np.ones((energies.size, mu.size)),
    }


def test_join_takes_muffin_tin_rows_to_the_crossover_and_free_atom_rows_above():
    mu = np.linspace(0.0, 1.0, 5)
    free = _table([1e5, 1e6, 1e7], [3.0, 2.0, 1.0], mu)
    muffin = _table([1e5, 1e6], [2.1, 1.4], mu)

    joined = joined_arrays(free, muffin)

    assert joined["energy_eV"].tolist() == [1e5, 1e6, 1e7]
    assert joined["total_elastic_cm2"].tolist() == [2.1, 1.4, 1.0]
    assert joined["dcs_cm2_sr"].shape == (3, 5)


def test_join_refuses_tables_on_different_angular_grids():
    free = _table([1e7], [1.0], np.linspace(0.0, 1.0, 5))
    muffin = _table([1e6], [1.0], np.linspace(0.0, 1.0, 6))

    with pytest.raises(ValueError, match="angular grids"):
        joined_arrays(free, muffin)


def test_one_element_layers_match_their_crystal_by_number_density():
    from pyrite.materials import CATALOG

    for key in ("silicon", "diamond", "hopg"):
        composition = CATALOG.crystal(key).info.composition
        solid = elemental_solid_for_composition(composition)
        assert solid is not None and solid.key == key
    assert elemental_solid_for_composition([("Si", 0.03)]) is None
    assert elemental_solid_for_composition(CATALOG.crystal("mos2").info.composition) is None


def test_missing_tables_fail_with_the_fetch_command(monkeypatch):
    from pyrite.materials import CATALOG

    monkeypatch.setattr("pyrite.xsgen.elsepa.catalog.resolve", lambda key: None)

    with pytest.raises(TableNotFoundError, match=r"free atom Si .*pyrite tables fetch elsepa"):
        resolve_layer_tables([("Si", 0.03)])
    monkeypatch.setattr(
        "pyrite.xsgen.elsepa.catalog.resolve",
        lambda key: None if "muffin" in str(key) else _FakeTable(),
    )
    monkeypatch.setattr(
        "pyrite.xsgen.elsepa.catalog.muffin_tin_request",
        lambda *a, **k: (type("R", (), {"key": "muffin"})(), None),
    )
    with pytest.raises(TableNotFoundError, match="elementary solid 'silicon'.*fetch elsepa"):
        resolve_layer_tables(CATALOG.crystal("silicon").info.composition)


class _FakeTable:
    def arrays(self):
        mu = np.linspace(0.0, 1.0, 5)
        return _table([1e5, 1e7], [2.0, 1.0], mu)
