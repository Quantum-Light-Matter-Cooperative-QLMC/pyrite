"""Package-layout and public-surface guards."""

from importlib import import_module, util
from importlib.metadata import distribution
from pathlib import Path

from packaging.requirements import Requirement

import pyrite


def test_distribution_identity_exposes_only_canonical_namespace() -> None:
    installed = distribution("pyrite-mc")

    assert installed.metadata["Name"] == "pyrite-mc"
    assert installed.version == pyrite.__version__
    assert util.find_spec("pyrite") is not None
    assert util.find_spec("cxr_mc") is None
    assert util.find_spec("pyrite_mc") is None


def test_pint_is_a_core_dependency_and_units_extra_remains_compatible() -> None:
    installed = distribution("pyrite-mc")
    requirements = [Requirement(value) for value in installed.requires or []]
    pint_requirements = [value for value in requirements if value.name.lower() == "pint"]
    assert len(pint_requirements) == 1
    assert pint_requirements[0].marker is None
    assert "0.26.2" in pint_requirements[0].specifier
    assert "0.27" not in pint_requirements[0].specifier
    assert "units" in installed.metadata.get_all("Provides-Extra", [])


def test_root_exports_stay_frozen() -> None:
    assert pyrite.__all__ == [
        "DATA_DIR",
        "__version__",
        "states",
        "Analysis",
        "AcquisitionBatch",
        "Beam",
        "Physics",
        "PhotonTransport",
        "ExecutionConfig",
        "KernelTuning",
        "QuantumRegion",
        "QuantumDynamics",
        "QuantumNumerics",
        "QuantumSetup",
        "TransverseStateSpec",
        "CxrSpectra",
        "Detector",
        "Numerics",
        "PixelMetadata",
        "PixelRayMap",
        "Provenance",
        "Result",
        "Scene",
        "SpatialResult",
        "SpectralDensity",
        "SpectralFactors",
        "Sweep",
        "Target",
        "fetch_tables",
        "photon_energies",
        "simulate",
        "DirectionalSpectrum",
    ]


def test_remote_facade_is_the_public_package() -> None:
    remote = import_module("pyrite.remote")

    assert Path(remote.__file__).name == "__init__.py"
    assert import_module("pyrite.remote.lifecycle") is remote.lifecycle


def test_source_modules_stay_near_the_1200_line_budget() -> None:
    """Keep issue #64's module-size acceptance criterion enforceable."""
    from pyrite.devtools.source_checks import oversized_source_modules

    assert oversized_source_modules(Path(pyrite.__file__).parent) == {}
