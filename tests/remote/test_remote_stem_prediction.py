"""Queue stem prediction agrees with the runner on canonical versus hashed (#361)."""

import argparse
import warnings
from dataclasses import replace

import pytest

from pyrite import Precision, materials
from pyrite.campaign.profiles import canonical_profile_run, named_profile_stem
from pyrite.remote import _queue_scripts as scripts
from pyrite.runs import scan


def _runner_stem(material):
    args = argparse.Namespace(fidelity="full", catalog_profile="standard")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return scan._resolved_run(args, material)[3]


def test_unmodified_standard_profile_keeps_the_bare_canonical_stem():
    assert canonical_profile_run("standard", "full") is True
    assert scripts._stems(["hopg"], False) == ["hopg"]


@pytest.mark.parametrize(
    "field, value",
    [
        (
            "profile_precisions",
            Precision(
                target_rse=0.1, min_electrons=20, max_electrons=60, block_electrons=20
            ).to_dict(),
        ),
        ("profile_emissions", "both"),
        ("profile_transport_numerics", {"straggling": True}),
    ],
)
def test_modified_standard_profile_never_predicts_the_canonical_stem(monkeypatch, field, value):
    catalog = materials.CATALOG
    monkeypatch.setattr(
        materials,
        "CATALOG",
        replace(catalog, **{field: {**getattr(catalog, field), "standard": value}}),
    )

    assert canonical_profile_run("standard", "full") is False
    predicted = scripts._stems(["hopg"], False)
    assert predicted == [named_profile_stem("hopg")]
    assert predicted[0].startswith("hopg@full-")
    assert _runner_stem("hopg").startswith("hopg@full-")
