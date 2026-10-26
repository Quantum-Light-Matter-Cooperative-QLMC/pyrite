"""Public facade coverage: discoverability, identity and import-light behavior."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from pyrite.devtools import api_surface

_BEAM_NAMES = (
    "Beam",
    "BunchFrontTilt",
    "BunchLength",
    "GdfSource",
    "Offsets",
    "PointBunch",
    "PointSpot",
    "Spot",
    "TransverseSpot",
)
_GEOMETRY_OWNERS = {
    "BlazedGrooves": "pyrite.campaign.geometry",
    "CrystalCut": "pyrite.campaign.geometry",
    "Filter": "pyrite.instrument",
    "Footprint": "pyrite.campaign.geometry",
    "Layer": "pyrite.campaign.geometry",
    "RectangularBody": "pyrite.bodies",
    "Slab": "pyrite.campaign.geometry",
    "Stack": "pyrite.campaign.geometry",
    "SubstanceRef": "pyrite.bodies",
    "Target": "pyrite.campaign.target",
}


@pytest.mark.parametrize(
    ("domain", "name", "owner"),
    [("beams", name, "pyrite.campaign.beam") for name in _BEAM_NAMES]
    + [("geometry", name, owner) for name, owner in _GEOMETRY_OWNERS.items()],
)
def test_physical_facades_preserve_implementation_and_pickle_identity(domain, name, owner):
    """All physical exports preserve class metadata and legacy pickle bindings."""
    import pyrite

    canonical = getattr(importlib.import_module(f"pyrite.{domain}"), name)
    original = getattr(importlib.import_module(owner), name)
    assert canonical is original is getattr(pyrite, name)
    # Class identity also preserves constructor signatures and dataclass Fields.
    assert pickle.dumps(canonical) == pickle.dumps(original)
    assert pickle.loads(pickle.dumps(original)) is canonical


@pytest.mark.parametrize(
    ("domain", "names"),
    [("beams", _BEAM_NAMES), ("geometry", tuple(_GEOMETRY_OWNERS))],
)
def test_physical_facade_inventories_are_deliberate(domain, names):
    facade = importlib.import_module(f"pyrite.{domain}")
    assert sorted(facade.__all__) == sorted(names)
    assert len(facade.__all__) == len(set(facade.__all__))
    assert set(names) <= set(dir(facade))


@pytest.mark.parametrize("gdf", [False, True], ids=["analytic", "gdf"])
def test_canonical_physical_inputs_compose_layered_scenes_and_sweeps(gdf):
    """Composition/serialization require neither GDF file I/O nor transport."""
    from dataclasses import asdict

    from pyrite import Scene, Sweep
    from pyrite.beams import Beam, BunchLength, GdfSource, PointBunch, PointSpot
    from pyrite.geometry import Footprint, Layer, Stack

    beam = (
        Beam.from_gdf("source.gdf", z_origin_m=0.0, time_s=1e-9, energy_keV=30.0)
        if gdf
        else Beam(energy_keV=30.0, transverse=PointSpot(), longitudinal=BunchLength(100.0))
    )
    target = Stack(
        layers=(Layer("hopg", 1e4), Layer("sio2", 2850.0), Layer("silicon", 5e6)),
        footprint=Footprint(5.0, 5.0),
    )
    scene = Scene(beam=beam, target=target)
    expanded = [value for _, value in Sweep(scene, {"beam.energy_keV": [30.0, 50.0]}).expand()]
    assert [value.beam.energy_keV for value in expanded] == [30.0, 50.0]
    assert all(value.target == target for value in expanded)
    assert Beam.from_spec(beam.to_spec()) == beam
    assert asdict(pickle.loads(pickle.dumps(scene))) == asdict(scene)
    if gdf:
        assert beam.gdf == GdfSource("source.gdf", 0.0, time_s=1e-9)
        assert all(value.beam.gdf == beam.gdf for value in expanded)
        assert isinstance(beam.longitudinal, PointBunch)
    else:
        from pyrite import api
        from pyrite.campaign.beam import Beam as OriginalBeam
        from pyrite.campaign.geometry import Footprint as OriginalFootprint
        from pyrite.campaign.geometry import Layer as OriginalLayer
        from pyrite.campaign.geometry import Stack as OriginalStack

        legacy_scene = Scene(
            beam=OriginalBeam.from_spec(beam.to_spec()),
            target=OriginalStack(
                layers=tuple(
                    OriginalLayer(layer.material, layer.thickness_ang) for layer in target.layers
                ),
                footprint=OriginalFootprint(5.0, 5.0),
            ),
        )
        assert api.build_case(scene) == api.build_case(legacy_scene)


def test_canonical_body_has_independent_target_and_filter_roles():
    from pyrite.detectors import PlanarPose
    from pyrite.geometry import Filter, RectangularBody, SubstanceRef, Target

    body = RectangularBody(SubstanceRef("crystal", "silicon"), (5.0, 5.0), 0.1)
    target = Target(body, tilt_deg=30.0)
    window = Filter(body, PlanarPose((0.0, 0.0, 50.0)), name="window")
    assert target.body is window.body is body
    assert target.to_slab().thickness_ang == 1_000_000.0
    assert window.to_plate().thickness_mm == 0.1
    for value in (body, target, window):
        assert pickle.loads(pickle.dumps(value)) == value


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
assert 'pyrite.campaign.beam' not in sys.modules
assert 'pyrite.campaign.geometry' not in sys.modules
assert 'pyrite.bodies' not in sys.modules
for facade in (pyrite.beams, pyrite.geometry):
    for name in facade.__all__:
        getattr(facade, name)
import pyrite.config
import pyrite.detectors
import pyrite.observations
from pyrite.beams import Beam
from pyrite.config import Physics, FixedSampling
from pyrite.detectors import Detector
from pyrite.observations import Observation
from pyrite import Scene, Result, Target
assert Beam.__name__ == 'Beam'
assert Physics.__name__ == 'Physics'
assert Detector.__name__ == 'Detector'
assert FixedSampling.__name__ == 'FixedSampling'
assert Observation.__name__ == 'Observation'
assert Observation().detector == 'detector'
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
