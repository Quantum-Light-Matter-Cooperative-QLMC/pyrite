"""Package-layout and public-surface guards."""

from importlib import import_module, util
from importlib.metadata import distribution
from pathlib import Path

import pyrite


def test_distribution_identity_exposes_only_canonical_namespace() -> None:
    installed = distribution("pyrite-mc")

    assert installed.metadata["Name"] == "pyrite-mc"
    assert installed.version == pyrite.__version__
    assert util.find_spec("pyrite") is not None
    assert util.find_spec("cxr_mc") is None
    assert util.find_spec("pyrite_mc") is None


def test_root_exports_stay_frozen() -> None:
    assert pyrite.__all__ == [
        "DATA_DIR",
        "__version__",
        "Analysis",
        "Acquisition",
        "NativeAcquisition",
        "AcquisitionBatch",
        "Beam",
        "BunchLength",
        "GdfSource",
        "Offsets",
        "PointBunch",
        "PointSpot",
        "Spot",
        "TransverseSpot",
        "BunchFrontTilt",
        "BlazedGrooves",
        "CrystalCut",
        "Physics",
        "CxrReflections",
        "CxrSpectra",
        "AngularAcceptance",
        "SensorAcceptance",
        "Detector",
        "EnergyBins",
        "Footprint",
        "Filter",
        "FilterPlate",
        "IdealPhotonCounter",
        "Layer",
        "Numerics",
        "Precision",
        "PixelGrid",
        "PixelScorer",
        "PlanarDetector",
        "PlanarPose",
        "PixelMetadata",
        "PixelRayMap",
        "Provenance",
        "RectangularBody",
        "Result",
        "ResolvedObservation",
        "Scene",
        "Slab",
        "Stack",
        "SpatialResult",
        "SpectralDensity",
        "SpectralFactors",
        "Sweep",
        "SubstanceRef",
        "Target",
        "fetch_tables",
        "photon_energies",
        "simulate",
    ]


def test_remote_facade_is_the_public_package() -> None:
    remote = import_module("pyrite.remote")

    assert Path(remote.__file__).name == "__init__.py"
    assert import_module("pyrite.remote.lifecycle") is remote.lifecycle


def test_source_modules_stay_near_the_1200_line_budget() -> None:
    """Keep issue #64's module-size acceptance criterion enforceable."""
    from pyrite.devtools.source_checks import oversized_source_modules

    assert oversized_source_modules(Path(pyrite.__file__).parent) == {}
