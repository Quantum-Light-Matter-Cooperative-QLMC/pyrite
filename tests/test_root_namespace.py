"""Root migration compatibility, discovery, warning filters, and object identity."""

import importlib
import re
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import pytest

import pyrite
from pyrite._api_deprecations import API_DEPRECATIONS, ROOT_API_REPLACEMENTS


@pytest.mark.parametrize("spelling,replacement", ROOT_API_REPLACEMENTS.items())
def test_deprecated_alias_warns_on_repeated_access_and_keeps_identity(spelling, replacement):
    name = spelling.removeprefix("pyrite.")
    module, attribute = replacement.rsplit(".", 1)
    canonical = getattr(importlib.import_module(module), attribute)
    entry = API_DEPRECATIONS[spelling]
    assert entry.deprecated_in == "0.20.5"
    assert entry.remove_in == "0.22.0"
    for _ in range(2):
        with pytest.warns(DeprecationWarning, match=entry.remove_in) as records:
            assert getattr(pyrite, name) is canonical
        assert replacement in str(records[0].message)
        assert records[0].filename == __file__
        assert name not in vars(pyrite)


def test_warning_filter_changes_are_respected_after_first_access():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        first = pyrite.FixedSampling
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match="pyrite.config.FixedSampling"):
            _ = pyrite.FixedSampling
    from pyrite.config import FixedSampling

    assert first is FixedSampling


def test_root_discovery_and_metadata_do_not_resolve_domain_exports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pyrite; "
            "before = set(sys.modules); "
            "assert set(pyrite._PUBLIC_OBJECTS) <= set(dir(pyrite)); "
            "assert set(pyrite._PUBLIC_OBJECTS) <= set(pyrite.__all__); "
            "assert set(pyrite._ROOT_API) <= set(pyrite.__all__); "
            "assert pyrite.__version__ and pyrite.DATA_DIR; "
            "assert before == set(sys.modules); "
            "assert not {'numpy', 'matplotlib', 'cupy', 'pyrite.api'} & before",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""


def test_star_import_and_states_module_remain_compatible():
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec("from pyrite import *", namespace)
    assert set(pyrite.__all__) <= namespace.keys()
    assert namespace["states"] is importlib.import_module("pyrite.states")
    assert namespace["Beam"] is pyrite.Beam


def test_unknown_root_attribute_keeps_attribute_error():
    with pytest.raises(AttributeError):
        _ = pyrite.no_such_public_object


def test_retained_and_deferred_root_exports_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for name in pyrite._PUBLIC_OBJECTS:
            if f"pyrite.{name}" not in ROOT_API_REPLACEMENTS:
                assert getattr(pyrite, name) is not None


def test_documented_root_classification_matches_governed_inventory():
    root = Path(__file__).resolve().parents[1]
    policies = tomllib.loads((root / "api-surface.toml").read_text())["root_policy"]
    documented = {
        name: (classification.lower(), canonical)
        for name, classification, canonical in re.findall(
            r"^\| `([\w]+)` \| (Retained|Deprecated|Deferred) \| `([\w.]+)` \|",
            (root / "docs/api.md").read_text(),
            re.M,
        )
    }
    assert documented == {
        name: (row["classification"], row["canonical"].replace(":", "."))
        for name, row in policies.items()
    }
