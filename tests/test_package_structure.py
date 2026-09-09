"""Package-layout and public-surface guards."""

from importlib import import_module, util
from importlib.metadata import distribution
from pathlib import Path

import pytest

import pyrite

_COMPAT_MODULES = {
    "analyze": "apps.analyze",
    "archive": "checkpoints.archive",
    "beam_metrics": "campaign.beam_metrics",
    "blaze": "runs.blaze",
    "campaign_lock": "checkpoints.campaign_lock",
    "check": "apps.check",
    "check_config": "cli.commands.check_config",
    "checkpoint_cleanup": "checkpoints.checkpoint_cleanup",
    "config": "campaign.config",
    "export": "apps.export",
    "longitudinal": "campaign.longitudinal",
    "performance_analysis": "perf.performance_analysis",
    "performance_profile": "perf.performance_profile",
    "profiles": "campaign.profiles",
    "recompute": "checkpoints.recompute",
    "recompute_defaults": "checkpoints.recompute_defaults",
    "run": "runs.run",
    "scan": "runs.scan",
    "slim": "checkpoints.slim",
    "sweep": "campaign.sweep",
    "transverse": "montecarlo.transverse",
    "validation_background": "validation.validation_background",
    "validation_oracles": "validation.validation_oracles",
    "viewer": "apps.viewer",
}


def test_distribution_identity_exposes_only_canonical_namespace() -> None:
    installed = distribution("pyrite-xray")

    assert installed.metadata["Name"] == "pyrite-xray"
    assert installed.version == pyrite.__version__
    assert util.find_spec("pyrite") is not None
    assert util.find_spec("cxr_mc") is None
    assert util.find_spec("pyrite_xray") is None


def test_root_exports_stay_frozen() -> None:
    assert pyrite.__all__ == [
        "DATA_DIR",
        "__version__",
        "Analysis",
        "Beam",
        "BlazedGrooves",
        "Convergence",
        "Detector",
        "EnergyBins",
        "Footprint",
        "FilterPlate",
        "Layer",
        "Numerics",
        "PixelGrid",
        "PixelScorer",
        "PlanarDetector",
        "PlanarPose",
        "PixelRayMap",
        "Result",
        "Scene",
        "Slab",
        "Stack",
        "SpatialResult",
        "SpectralFactors",
        "Sweep",
        "simulate",
    ]


@pytest.mark.parametrize(("legacy_name", "canonical_name"), _COMPAT_MODULES.items())
def test_root_module_reexports(legacy_name: str, canonical_name: str) -> None:
    legacy = import_module(f"pyrite.{legacy_name}")
    canonical = import_module(f"pyrite.{canonical_name}")
    exports = getattr(
        canonical,
        "__all__",
        [name for name in vars(canonical) if not name.startswith("_")],
    )
    stable_exports = [name for name in exports if callable(getattr(canonical, name))]

    assert stable_exports
    assert all(getattr(legacy, name) is getattr(canonical, name) for name in stable_exports)
    # The canonical module must live in the package its dotted path names, so a
    # shim cannot quietly start pointing somewhere else in the tree.
    assert Path(canonical.__file__).parent.name == canonical_name.rsplit(".", 2)[-2]


def test_remote_facade_is_the_public_package() -> None:
    remote = import_module("pyrite.remote")

    assert Path(remote.__file__).name == "__init__.py"
    assert import_module("pyrite.remote.lifecycle") is remote.lifecycle


def test_source_modules_stay_near_the_1200_line_budget() -> None:
    """Keep issue #64's module-size acceptance criterion enforceable."""
    package_root = Path(pyrite.__file__).parent
    exempt = {
        Path("montecarlo/spectrum/lines.py"),  # issue #65
        Path("montecarlo/transport/cores.py"),  # issue #66
    }
    oversized = {
        path.relative_to(package_root): len(path.read_text().splitlines())
        for path in package_root.rglob("*.py")
        if path.relative_to(package_root) not in exempt
        and len(path.read_text().splitlines()) > 1_250
    }

    assert oversized == {}
