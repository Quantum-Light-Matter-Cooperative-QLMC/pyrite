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

REMOVED_ROOT_NAMES = (
    "Acquisition",
    "AdaptiveSampling",
    "AngularAcceptance",
    "BlazedGrooves",
    "BunchFrontTilt",
    "BunchLength",
    "CrystalCut",
    "CxrReflections",
    "EnergyBins",
    "Filter",
    "FilterPlate",
    "FixedSampling",
    "Footprint",
    "GdfSource",
    "IdealPhotonCounter",
    "Layer",
    "NativeAcquisition",
    "Offsets",
    "PixelGrid",
    "PixelScorer",
    "PlanarDetector",
    "PlanarPose",
    "PointBunch",
    "PointSpot",
    "Precision",
    "RadiationNumerics",
    "RectangularBody",
    "ResolvedObservation",
    "SensorAcceptance",
    "Slab",
    "Spot",
    "Stack",
    "SubstanceRef",
    "TransportNumerics",
    "TransverseSpot",
    "directional_source",
    "directional_sources",
    "load_numerics",
    "load_physics",
)


@pytest.mark.parametrize("name", REMOVED_ROOT_NAMES)
def test_due_root_alias_is_no_longer_exported(name):
    with pytest.raises(AttributeError):
        getattr(pyrite, name)


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
            assert getattr(pyrite, name) is not None


def test_documented_root_classification_matches_governed_inventory():
    root = Path(__file__).resolve().parents[1]
    policies = tomllib.loads((root / "api-surface.toml").read_text())["root_policy"]
    documented = {
        name: (classification.lower(), canonical)
        for name, classification, canonical in re.findall(
            r"^\| `([\w]+)` \| (Retained|Deprecated|Deferred) \| `([\w.]+)` \|",
            (root / "docs/reference/python-api/compatibility.md").read_text(),
            re.M,
        )
    }
    assert documented == {
        name: (row["classification"], row["canonical"].replace(":", "."))
        for name, row in policies.items()
    }
