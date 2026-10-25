"""Public facade coverage: discoverability, identity and import-light behavior."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from pyrite.devtools import api_surface


@pytest.mark.parametrize(
    ("domain", "name"),
    [
        ("beams", "Beam"),
        ("beams", "TransverseSpot"),
        ("geometry", "Slab"),
        ("geometry", "Target"),
        ("config", "Physics"),
        ("config", "Numerics"),
        ("config", "FixedSampling"),
        ("config", "load_physics"),
        ("detectors", "Detector"),
        ("detectors", "AngularAcceptance"),
        ("detectors", "PlanarDetector"),
        ("detectors", "PixelGrid"),
        ("detectors", "PlanarPose"),
        ("detectors", "FilterPlate"),
        ("observations", "Acquisition"),
        ("observations", "NativeAcquisition"),
        ("observations", "AcquisitionBatch"),
        ("observations", "PixelScorer"),
        ("observations", "ResolvedObservation"),
    ],
)
def test_canonical_facade_preserves_root_object_identity(domain: str, name: str) -> None:
    """Re-export exactly the same type/function, not wrapper subclasses."""
    import pyrite

    facade = importlib.import_module(f"pyrite.{domain}")
    assert getattr(facade, name) is getattr(pyrite, name)


def test_canonical_names_match_reviewed_manifest() -> None:
    root = Path(__file__).resolve().parents[2]
    exports, errors = api_surface.read_manifest(root / "api-surface.toml")
    assert not errors
    domains, errors = api_surface.read_facade_domains(root / "api-surface.toml")
    assert not errors
    assert set(domains) == {"beams", "geometry", "config", "detectors", "observations"}
    assert not api_surface.check_facades(root, exports, domains)


def test_namespaces_import_without_eager_heavy_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import sys
from importlib.abc import MetaPathFinder

OPTIONAL = {'cupy', 'matplotlib', 'dask', 'distributed', 'optuna'}

class DenyOptional(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in OPTIONAL:
            raise AssertionError(f'eager optional dependency import: {fullname}')
        return None

sys.meta_path.insert(0, DenyOptional())
import pyrite
assert 'pyrite.montecarlo' not in sys.modules
assert 'pyrite.campaign.model' not in sys.modules
import pyrite.beams
import pyrite.geometry
import pyrite.config
import pyrite.detectors
import pyrite.observations
from pyrite.beams import Beam
from pyrite.config import Physics, FixedSampling
from pyrite.detectors import Detector
from pyrite import Scene, Result, Target
assert Beam.__name__ == 'Beam'
assert Physics.__name__ == 'Physics'
assert Detector.__name__ == 'Detector'
assert FixedSampling.__name__ == 'FixedSampling'
assert Scene.__name__ == 'Scene'
assert Result.__name__ == 'Result'
assert Target.__name__ == 'Target'
for name in OPTIONAL:
    assert name not in sys.modules, name
assert 'Beam' in dir(pyrite.beams)
assert 'Slab' in dir(pyrite.geometry)
assert 'Physics' in dir(pyrite.config)
assert 'PlanarDetector' in dir(pyrite.detectors)
assert 'Acquisition' in dir(pyrite.observations)
assert 'pyrite.montecarlo.runner' not in sys.modules
assert 'pyrite.observations.plan' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_observation_legacy_exports_are_unchanged() -> None:
    """Existing observation inventory/plan/store paths continue to resolve."""
    from pyrite.observations import ObservationPlan, ObservationStore, PixelSampling
    from pyrite.observations.plan import ObservationPlan as OriginalPlan
    from pyrite.observations.plan import PixelSampling as OriginalSampling
    from pyrite.observations.store import ObservationStore as OriginalStore

    assert ObservationPlan is OriginalPlan
    assert PixelSampling is OriginalSampling
    assert ObservationStore is OriginalStore


def test_canonical_detector_and_acquisition_values_are_interoperable() -> None:
    """Physical detector/scorer/acquisition contracts survive import relocation."""
    from pyrite.detectors import IdealPhotonCounter, PixelGrid, PlanarDetector, PlanarPose
    from pyrite.observations import Acquisition, PixelScorer

    pixels = PixelGrid(shape=(2, 2), pitch_mm=(0.1, 0.1))
    detector = PlanarDetector(
        pose=PlanarPose(center_mm=(0.0, 0.0, 100.0)),
        pixels=pixels,
        response=IdealPhotonCounter(),
    )
    request = Acquisition(
        exposure_s=1.0,
        measured_edges_eV=(100.0, 200.0, 300.0),
    )
    assert isinstance(detector.scorer, PixelScorer)
    assert detector.pixels is pixels
    assert request.exposure_s == 1.0


def test_mixed_facade_requires_canonical_names_in_all(tmp_path: Path) -> None:
    """Preexisting extra exports are valid, but declared canonical ones must appear."""
    source = tmp_path / "__init__.py"
    source.write_text('_EXPORTS = {"New": ("pyrite.instrument", "New")}\n__all__ = ["Previous"]\n')
    mapping, errors = api_surface.read_facade_exports(source)
    assert mapping == {"New": ("pyrite.instrument", "New")}
    assert any("__all__ omits canonical exports" in error for error in errors)


@pytest.mark.parametrize(
    "all_expression",
    [
        '__all__ = ["New", 1]',
        '__all__ = ["New", "New"]',
        '__all__ = ["New", None]',
    ],
)
def test_literal_facade_all_requires_unique_strings(tmp_path: Path, all_expression: str) -> None:
    """Literal AST constants may be non-strings; the checker must reject them."""
    source = tmp_path / "__init__.py"
    source.write_text('_EXPORTS = {"New": ("pyrite.instrument", "New")}\n' + all_expression)
    mapping, errors = api_surface.read_facade_exports(source)
    assert mapping == {"New": ("pyrite.instrument", "New")}
    assert any("unique string list" in error for error in errors)


def test_literal_facade_all_with_strings_is_valid(tmp_path: Path) -> None:
    """A legacy package may have additional, ungoverned public exports."""
    source = tmp_path / "__init__.py"
    source.write_text(
        '_EXPORTS = {"New": ("pyrite.instrument", "New")}\n__all__ = ["Existing", "New"]\n'
    )
    mapping, errors = api_surface.read_facade_exports(source)
    assert mapping == {"New": ("pyrite.instrument", "New")}
    assert not errors
