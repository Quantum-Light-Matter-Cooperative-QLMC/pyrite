"""Generated tables reach run identity, and only when a run reads one.

Two properties, pulling against each other:

1. A run that reads a generated table must not resume into results computed
   from a different version of it. This is the failure D5 calls the
   highest-risk piece -- regenerating a table with different deck parameters
   and silently reusing the old checkpoints.
2. A run that reads no generated table must keep its digest bit-for-bit. No
   consumer resolves an xsgen table yet, so a marker that always applied would
   orphan every existing checkpoint in exchange for no change in the numbers.

Both identity surfaces are covered, because they gate different caches:
``dataset_identity`` gates the checkpoint stem, ``case_content_key`` gates the
content-addressable blob store.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import case_content_key, dataset_identity, variant_stem
from pyrite.xsgen.store import ElementTarget, TableRequest, identity_markers, store

pytestmark = pytest.mark.usefixtures("isolated_dirs")


def _identity(**kwargs):
    sweep = material_sweep("hopg")
    return dataset_identity("hopg", "full", default_settings(), sweep, **kwargs)


def _case() -> dict[str, object]:
    return {"material": "hopg", "E_keV": 100.0, "n_electrons": 1000, "seed": 7}


def _stored(deck: dict[str, object]):
    request = TableRequest(
        code="elsepa",
        code_version="a" * 64,
        target=ElementTarget(z=6),
        quantity="elastic_dcs",
        model=deck,
    )
    return store(request, {"dcs": np.array([1.0, 2.0])})


# --- property 2: runs that read no table are untouched --------------------


def test_a_run_reading_no_table_keeps_its_digest():
    baseline = _identity()["parameter_sha256"]
    assert _identity(xsgen_tables=None)["parameter_sha256"] == baseline
    assert _identity(xsgen_tables={})["parameter_sha256"] == baseline


def test_a_case_reading_no_table_keeps_its_content_key():
    baseline = case_content_key(_case())
    assert case_content_key(_case(), xsgen_tables=None) == baseline
    assert case_content_key(_case(), xsgen_tables={}) == baseline


def test_the_marker_is_absent_from_a_run_that_reads_no_table():
    """Absent, not empty: an empty mapping in the payload would re-key."""
    assert "xsgen_tables" not in _identity()["resolved_parameters"]


# --- property 1: a table change re-keys the run ---------------------------


def test_reading_a_table_forks_the_dataset_identity():
    markers = identity_markers([_stored({"mabs": 2})])
    assert _identity(xsgen_tables=markers)["parameter_sha256"] != _identity()["parameter_sha256"]


def test_regenerating_a_table_with_a_different_deck_forks_the_identity():
    """The D5 failure, frozen.

    ``MABS 2`` and ``MABS 0`` are different absorption models producing
    different cross sections. Sharing an identity would serve checkpoints
    computed from whichever was generated first.
    """
    first = _identity(xsgen_tables=identity_markers([_stored({"mabs": 2})]))
    second = _identity(xsgen_tables=identity_markers([_stored({"mabs": 0})]))

    assert first["parameter_sha256"] != second["parameter_sha256"]
    # The stem carries a 12-hex digest prefix, so the two land in different
    # checkpoint files rather than one overwriting the other.
    assert variant_stem(first) != variant_stem(second)


def test_regenerating_a_table_with_a_different_deck_forks_the_content_key():
    """The stem moving is not enough on its own.

    The content-addressable store is keyed separately, so without this a run
    whose stem had moved would still be handed a blob computed from the
    superseded table.
    """
    first = case_content_key(_case(), xsgen_tables=identity_markers([_stored({"mabs": 2})]))
    second = case_content_key(_case(), xsgen_tables=identity_markers([_stored({"mabs": 0})]))
    assert first != second


def test_an_unchanged_table_keeps_one_identity():
    """Resolving the same table twice must not re-key, or nothing ever resumes."""
    table = _stored({"mabs": 2})
    markers = identity_markers([table])
    assert (
        _identity(xsgen_tables=markers)["parameter_sha256"]
        == _identity(xsgen_tables=identity_markers([table]))["parameter_sha256"]
    )


def test_the_marker_tracks_the_manifest_not_the_key():
    """A table regenerated under the same key still re-keys the run.

    Same request, so the same table key; a fresh manifest, so a fresh digest.
    Keying the run on the table key alone would miss this.
    """
    first = _stored({"mabs": 2})
    first_markers = identity_markers([first])
    second = store(
        TableRequest(
            code="elsepa",
            code_version="a" * 64,
            target=ElementTarget(z=6),
            quantity="elastic_dcs",
            model={"mabs": 2},
        ),
        {"dcs": np.array([3.0, 4.0])},
        overwrite=True,
    )

    assert second.key == first.key
    assert identity_markers([second]) != first_markers
    assert (
        _identity(xsgen_tables=identity_markers([second]))["parameter_sha256"]
        != _identity(xsgen_tables=first_markers)["parameter_sha256"]
    )


def test_markers_map_every_table_a_run_reads():
    elastic = _stored({"mabs": 2})
    stopping = store(
        TableRequest(
            code="sbethe",
            code_version="b" * 64,
            target=ElementTarget(z=6),
            quantity="collisional_stopping",
            model={},
        ),
        {"stp": np.array([1.0])},
    )

    markers = identity_markers([elastic, stopping])
    assert markers == {elastic.key: elastic.digest, stopping.key: stopping.digest}
