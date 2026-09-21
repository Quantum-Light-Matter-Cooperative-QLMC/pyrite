"""Table keying, provenance, and two-tier resolution.

The key is the invalidation rule: change the code, the deck, or the material
and the key changes, so a table generated under the old inputs is *not found*
rather than silently served. These tests freeze that, and freeze that a
shipped table and a user-generated one are indistinguishable to a consumer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyrite.xsgen._errors import TableNotFoundError
from pyrite.xsgen.store import (
    MODIFICATIONS_NOTE,
    ElementTarget,
    MaterialTarget,
    TableRequest,
    iter_stored,
    material_identity,
    require,
    resolve,
    store,
    user_table_dir,
)

pytestmark = pytest.mark.usefixtures("isolated_dirs")


def _request(**overrides) -> TableRequest:
    payload = {
        "code": "elsepa",
        "code_version": "a" * 64,
        "target": ElementTarget(z=29),
        "quantity": "elastic_dcs",
        "model": {"muffin": 1, "mabs": 2},
    }
    payload.update(overrides)
    return TableRequest(**payload)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "theta_deg": np.linspace(0.0, 180.0, 5),
        "dcs_cm2_sr": np.geomspace(1.0e-16, 1.0e-20, 5),
    }


# --- keying ---------------------------------------------------------------


def test_the_same_request_keys_the_same_table():
    assert _request().key == _request().key


def test_a_changed_deck_parameter_changes_the_key():
    """The case D5 calls the highest-risk one.

    ``MABS 2`` and ``MABS 0`` are different absorption models producing
    different numbers. If they shared a key, regenerating would serve
    checkpoints computed from the other one.
    """
    assert (
        _request(model={"muffin": 1, "mabs": 2}).key != _request(model={"muffin": 1, "mabs": 0}).key
    )


def test_a_patched_fortran_source_changes_the_key():
    assert _request().key != _request(code_version="b" * 64).key


def test_each_output_of_one_program_keys_separately():
    assert _request(quantity="elastic_dcs").key != _request(quantity="transport_xs").key


def test_a_different_element_keys_separately():
    assert _request().key != _request(target=ElementTarget(z=30)).key


def test_deck_hash_isolates_the_model_parameters():
    shared_deck = {"muffin": 1, "mabs": 2}
    elsepa = _request(model=shared_deck)
    other_code = _request(code="sbethe", model=shared_deck)
    assert elsepa.deck_hash == other_code.deck_hash
    assert elsepa.key != other_code.key


# --- material identity (finding F1) ---------------------------------------


def test_material_identity_is_independent_of_stoichiometry_spelling():
    """``MoS2`` written as counts and as fractions is one material."""
    counts = material_identity(composition={42: 1, 16: 2}, density_g_cm3=5.06)
    fractions = material_identity(composition={42: 1 / 3, 16: 2 / 3}, density_g_cm3=5.06)
    assert counts == fractions


def test_material_identity_tracks_composition_density_and_excitation():
    base = {"composition": {42: 1, 16: 2}, "density_g_cm3": 5.06}
    reference = material_identity(**base)
    assert material_identity(composition={42: 1, 16: 3}, density_g_cm3=5.06) != reference
    assert material_identity(**{**base, "density_g_cm3": 5.07}) != reference
    assert material_identity(**base, mean_excitation_eV=180.0) != reference


def test_material_identity_tracks_code_specific_material_scalars():
    """``RMUF`` and the band gap change what the generated table describes."""
    base = {"composition": {14: 1}, "density_g_cm3": 2.329}
    assert material_identity(**base, extra={"rmuf_a": 2.35}) != material_identity(
        **base, extra={"rmuf_a": 2.40}
    )


def test_a_material_target_keys_on_composition_not_on_its_catalogue_name():
    """Renaming a catalogue entry must not re-key its tables, and editing the
    composition behind an unchanged name must."""
    identity = material_identity(composition={42: 1, 16: 2}, density_g_cm3=5.06)
    renamed = _request(target=MaterialTarget(key="mos2_2h", identity=identity))
    original = _request(target=MaterialTarget(key="mos2", identity=identity))
    assert renamed.key == original.key

    edited = material_identity(composition={42: 1, 16: 2}, density_g_cm3=5.10)
    assert _request(target=MaterialTarget(key="mos2", identity=edited)).key != original.key


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"composition": {}, "density_g_cm3": 1.0}, "composition is empty"),
        ({"composition": {14: 0.0}, "density_g_cm3": 1.0}, "non-positive atom fraction"),
        ({"composition": {14: 1.0}, "density_g_cm3": 0.0}, "density must be positive"),
    ],
)
def test_material_identity_rejects_unphysical_records(kwargs, message):
    with pytest.raises(ValueError, match=message):
        material_identity(**kwargs)


# --- storage and resolution ----------------------------------------------


def test_stored_table_round_trips_its_arrays():
    request = _request()
    expected = _arrays()
    store(request, expected)

    found = resolve(request.key)
    assert found is not None
    loaded = found.arrays()
    assert set(loaded) == set(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(loaded[name], values)


def test_the_manifest_records_how_the_table_was_made():
    request = _request()
    stored = store(request, _arrays(), compiler="GNU Fortran 15.2.0", source_origin="vendored")
    manifest = json.loads((user_table_dir() / f"{request.key}.json").read_text(encoding="utf-8"))

    assert manifest["key"] == request.key
    assert manifest["code"] == "elsepa"
    assert manifest["code_version"] == request.code_version
    assert manifest["deck_hash"] == request.deck_hash
    assert manifest["compiler"] == "GNU Fortran 15.2.0"
    assert manifest["model"] == {"muffin": 1, "mabs": 2}
    assert manifest["arrays"]["theta_deg"]["shape"] == [5]
    assert manifest["manifest_sha256"] == stored.digest


def test_every_manifest_marks_the_table_as_an_adaptation():
    """CC BY, on all three codes, requires derived material to say so."""
    request = _request()
    store(request, _arrays())
    assert resolve(request.key).manifest["modifications"] == MODIFICATIONS_NOTE


def test_two_tables_differing_only_in_deck_do_not_overwrite_each_other():
    first = _request(model={"mabs": 0})
    second = _request(model={"mabs": 2})
    store(first, {"x": np.array([1.0])})
    store(second, {"x": np.array([2.0])})

    np.testing.assert_array_equal(resolve(first.key).arrays()["x"], [1.0])
    np.testing.assert_array_equal(resolve(second.key).arrays()["x"], [2.0])


def test_storing_over_an_existing_key_needs_an_explicit_overwrite():
    request = _request()
    store(request, _arrays())
    with pytest.raises(FileExistsError):
        store(request, _arrays())
    store(request, {"x": np.array([9.0])}, overwrite=True)
    np.testing.assert_array_equal(resolve(request.key).arrays()["x"], [9.0])


def test_an_empty_table_is_refused():
    with pytest.raises(ValueError, match="no arrays"):
        store(_request(), {})


def test_an_unknown_key_resolves_to_none():
    assert resolve("f" * 64) is None


def test_require_names_both_tiers_and_the_generate_command():
    with pytest.raises(TableNotFoundError) as excinfo:
        require("f" * 64)
    message = str(excinfo.value)
    assert str(user_table_dir()) in message
    assert "pyrite tables generate" in message


# --- two-tier resolution (D3) --------------------------------------------


def test_a_packaged_table_resolves_when_the_user_has_none(packaged_dir):
    request = _request()
    store(request, {"x": np.array([7.0])}, root=packaged_dir)

    found = resolve(request.key)
    assert found is not None
    assert found.tier == "packaged"
    np.testing.assert_array_equal(found.arrays()["x"], [7.0])


def test_a_user_table_overrides_a_shipped_one_without_deleting_it(packaged_dir):
    """A locally regenerated table wins, and the shipped bytes stay put."""
    request = _request()
    store(request, {"x": np.array([7.0])}, root=packaged_dir)
    store(request, {"x": np.array([8.0])})

    found = resolve(request.key)
    assert found.tier == "user"
    np.testing.assert_array_equal(found.arrays()["x"], [8.0])
    assert (packaged_dir / f"{request.key}.npz").is_file()


def test_iter_stored_lists_each_key_once_across_tiers(packaged_dir):
    shared = _request()
    packaged_only = _request(quantity="transport_xs")
    store(shared, {"x": np.array([7.0])}, root=packaged_dir)
    store(packaged_only, {"x": np.array([6.0])}, root=packaged_dir)
    store(shared, {"x": np.array([8.0])})

    listed = {table.key: table.tier for table in iter_stored()}
    assert listed == {shared.key: "user", packaged_only.key: "packaged"}


def test_a_table_without_its_manifest_is_not_served():
    """Half a table is not a table: provenance is what makes it usable."""
    request = _request()
    store(request, _arrays())
    (user_table_dir() / f"{request.key}.json").unlink()
    assert resolve(request.key) is None
    assert list(iter_stored()) == []
