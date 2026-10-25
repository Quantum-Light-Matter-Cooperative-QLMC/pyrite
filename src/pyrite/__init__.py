"""pyrite -- crystal X-ray radiation (PXR + coherent bremsstrahlung) from
table-top electron beams in crystals.

The package is intentionally light at import time: ``import pyrite`` pulls in
no heavy dependencies (matplotlib / cupy / the MC pipeline). Use the curated root for primary simulation objects and canonical domain
imports for specialized types, e.g. ``from pyrite.beams import TransverseSpot``
and ``from pyrite.geometry import Slab``. Historical root aliases remain
available through their documented compatibility windows.

See the README for the scientific overview, ``docs/api.md`` for the supported
surface, and ``AGENTS.md`` for repository working conventions.
"""

import logging
from typing import TYPE_CHECKING, Any

from ._env import env_value
from .paths import data_dir

__version__ = "0.20.4"

# Packaged data (catalog/, cifs/,
# eaglexo_qe.csv, and the other entries ADR-0014 classifies as in-wheel).
# Resolved relative to this file so it works installed (wheel) or from a source checkout.
DATA_DIR = data_dir()

# Package logger. Library convention: attach a NullHandler so a plain `import
# pyrite` (and every `pyrite` CLI invocation, incl. each ProcessPoolExecutor
# worker in montecarlo.runner) stays silent -- submodules log routine
# noise (for example the GPU/CPU backend probe) at DEBUG,
# which propagates nowhere by default. Set PYRITE_MC_DEBUG=1 to see it: this
# attaches our own StreamHandler at DEBUG on just the "pyrite" logger,
# without touching the caller's root logging config.
logger = logging.getLogger("pyrite")
logger.addHandler(logging.NullHandler())
if env_value("PYRITE_MC_DEBUG"):
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

if TYPE_CHECKING:
    # Static declarations retain compatibility while recommending canonical owners.
    from pyrite._numerics import Physics as Physics
    from pyrite._numerics_groups import ExecutionConfig as ExecutionConfig
    from pyrite._photon_config import PhotonTransport as PhotonTransport
    from pyrite.api import simulate as simulate
    from pyrite.beams import (
        BunchFrontTilt as BunchFrontTilt,
    )
    from pyrite.beams import (
        BunchLength as BunchLength,
    )
    from pyrite.beams import (
        GdfSource as GdfSource,
    )
    from pyrite.beams import (
        Offsets as Offsets,
    )
    from pyrite.beams import (
        PointBunch as PointBunch,
    )
    from pyrite.beams import (
        PointSpot as PointSpot,
    )
    from pyrite.beams import (
        Spot as Spot,
    )
    from pyrite.beams import (
        TransverseSpot as TransverseSpot,
    )
    from pyrite.campaign.beam import Beam as Beam
    from pyrite.campaign.model import (
        Analysis as Analysis,
    )
    from pyrite.campaign.model import (
        Numerics as Numerics,
    )
    from pyrite.campaign.model import (
        Scene as Scene,
    )
    from pyrite.campaign.model import (
        Sweep as Sweep,
    )
    from pyrite.campaign.target import Target as Target
    from pyrite.config import (
        AdaptiveSampling as AdaptiveSampling,
    )
    from pyrite.config import (
        CxrReflections as CxrReflections,
    )
    from pyrite.config import (
        FixedSampling as FixedSampling,
    )
    from pyrite.config import (
        Precision as Precision,
    )
    from pyrite.config import (
        RadiationNumerics as RadiationNumerics,
    )
    from pyrite.config import (
        TransportNumerics as TransportNumerics,
    )
    from pyrite.config import (
        load_numerics as load_numerics,
    )
    from pyrite.config import (
        load_physics as load_physics,
    )
    from pyrite.detectors import (
        AngularAcceptance as AngularAcceptance,
    )
    from pyrite.detectors import (
        Detector as Detector,
    )
    from pyrite.detectors import (
        EnergyBins as EnergyBins,
    )
    from pyrite.detectors import (
        FilterPlate as FilterPlate,
    )
    from pyrite.detectors import (
        IdealPhotonCounter as IdealPhotonCounter,
    )
    from pyrite.detectors import (
        PixelGrid as PixelGrid,
    )
    from pyrite.detectors import (
        PlanarDetector as PlanarDetector,
    )
    from pyrite.detectors import (
        PlanarPose as PlanarPose,
    )
    from pyrite.detectors import (
        SensorAcceptance as SensorAcceptance,
    )
    from pyrite.directional import (
        directional_source as directional_source,
    )
    from pyrite.directional import (
        directional_sources as directional_sources,
    )
    from pyrite.geometry import (
        BlazedGrooves as BlazedGrooves,
    )
    from pyrite.geometry import (
        CrystalCut as CrystalCut,
    )
    from pyrite.geometry import (
        Filter as Filter,
    )
    from pyrite.geometry import (
        Footprint as Footprint,
    )
    from pyrite.geometry import (
        Layer as Layer,
    )
    from pyrite.geometry import (
        RectangularBody as RectangularBody,
    )
    from pyrite.geometry import (
        Slab as Slab,
    )
    from pyrite.geometry import (
        Stack as Stack,
    )
    from pyrite.geometry import (
        SubstanceRef as SubstanceRef,
    )
    from pyrite.materials.emission import photon_energies as photon_energies
    from pyrite.observations import (
        Acquisition as Acquisition,
    )
    from pyrite.observations import (
        AcquisitionBatch as AcquisitionBatch,
    )
    from pyrite.observations import (
        NativeAcquisition as NativeAcquisition,
    )
    from pyrite.observations import (
        PixelScorer as PixelScorer,
    )
    from pyrite.observations import (
        ResolvedObservation as ResolvedObservation,
    )
    from pyrite.quantum import (
        QuantumDynamics as QuantumDynamics,
    )
    from pyrite.quantum import (
        QuantumNumerics as QuantumNumerics,
    )
    from pyrite.quantum import (
        QuantumRegion as QuantumRegion,
    )
    from pyrite.quantum import (
        QuantumSetup as QuantumSetup,
    )
    from pyrite.quantum_state import TransverseStateSpec as TransverseStateSpec
    from pyrite.results.directional import DirectionalSpectrum as DirectionalSpectrum
    from pyrite.results.model import (
        CxrSpectra as CxrSpectra,
    )
    from pyrite.results.model import (
        PixelMetadata as PixelMetadata,
    )
    from pyrite.results.model import (
        PixelRayMap as PixelRayMap,
    )
    from pyrite.results.model import (
        Result as Result,
    )
    from pyrite.results.model import (
        SpatialResult as SpatialResult,
    )
    from pyrite.results.model import (
        SpectralDensity as SpectralDensity,
    )
    from pyrite.results.model import (
        SpectralFactors as SpectralFactors,
    )
    from pyrite.results.provenance import Provenance as Provenance
    from pyrite.xsgen.install import fetch_tables as fetch_tables

_PUBLIC_OBJECTS = {
    "Analysis": ("pyrite.campaign.model", "Analysis"),
    "Acquisition": ("pyrite.instrument", "Acquisition"),
    "NativeAcquisition": ("pyrite.instrument", "NativeAcquisition"),
    "AcquisitionBatch": ("pyrite.instrument", "AcquisitionBatch"),
    "Beam": ("pyrite.campaign.beam", "Beam"),
    "BunchLength": ("pyrite.campaign.beam", "BunchLength"),
    "GdfSource": ("pyrite.campaign.beam", "GdfSource"),
    "Offsets": ("pyrite.campaign.beam", "Offsets"),
    "PointBunch": ("pyrite.campaign.beam", "PointBunch"),
    "PointSpot": ("pyrite.campaign.beam", "PointSpot"),
    "Spot": ("pyrite.campaign.beam", "Spot"),
    "TransverseSpot": ("pyrite.campaign.beam", "TransverseSpot"),
    "BunchFrontTilt": ("pyrite.campaign.beam", "BunchFrontTilt"),
    "BlazedGrooves": ("pyrite.campaign.geometry", "BlazedGrooves"),
    "CrystalCut": ("pyrite.campaign.geometry", "CrystalCut"),
    "Physics": ("pyrite._numerics", "Physics"),
    "PhotonTransport": ("pyrite._photon_config", "PhotonTransport"),
    "TransportNumerics": ("pyrite._numerics_groups", "TransportNumerics"),
    "RadiationNumerics": ("pyrite._numerics_groups", "RadiationNumerics"),
    "ExecutionConfig": ("pyrite._numerics_groups", "ExecutionConfig"),
    "QuantumRegion": ("pyrite.quantum", "QuantumRegion"),
    "QuantumDynamics": ("pyrite.quantum", "QuantumDynamics"),
    "QuantumNumerics": ("pyrite.quantum", "QuantumNumerics"),
    "QuantumSetup": ("pyrite.quantum", "QuantumSetup"),
    "TransverseStateSpec": ("pyrite.quantum_state", "TransverseStateSpec"),
    "CxrReflections": ("pyrite._numerics", "CxrReflections"),
    "CxrSpectra": ("pyrite.results.model", "CxrSpectra"),
    "AngularAcceptance": ("pyrite.detectors", "AngularAcceptance"),
    "SensorAcceptance": ("pyrite.detectors", "SensorAcceptance"),
    "Detector": ("pyrite.detectors", "Detector"),
    "EnergyBins": ("pyrite.detectors", "EnergyBins"),
    "Footprint": ("pyrite.campaign.geometry", "Footprint"),
    "Filter": ("pyrite.instrument", "Filter"),
    "FilterPlate": ("pyrite.instrument", "FilterPlate"),
    "IdealPhotonCounter": ("pyrite.detectors", "IdealPhotonCounter"),
    "Layer": ("pyrite.campaign.geometry", "Layer"),
    "Numerics": ("pyrite.campaign.model", "Numerics"),
    "Precision": ("pyrite._precision", "Precision"),
    "FixedSampling": ("pyrite._sampling", "FixedSampling"),
    "AdaptiveSampling": ("pyrite._sampling", "AdaptiveSampling"),
    "PixelGrid": ("pyrite.instrument", "PixelGrid"),
    "PixelScorer": ("pyrite.instrument", "PixelScorer"),
    "PlanarDetector": ("pyrite.instrument", "PlanarDetector"),
    "PlanarPose": ("pyrite.instrument", "PlanarPose"),
    "PixelMetadata": ("pyrite.results.model", "PixelMetadata"),
    "PixelRayMap": ("pyrite.results.model", "PixelRayMap"),
    "Provenance": ("pyrite.results.provenance", "Provenance"),
    "RectangularBody": ("pyrite.bodies", "RectangularBody"),
    "Result": ("pyrite.results.model", "Result"),
    "ResolvedObservation": ("pyrite.instrument", "ResolvedObservation"),
    "Scene": ("pyrite.campaign.model", "Scene"),
    "Slab": ("pyrite.campaign.geometry", "Slab"),
    "Stack": ("pyrite.campaign.geometry", "Stack"),
    "SpatialResult": ("pyrite.results.model", "SpatialResult"),
    "SpectralDensity": ("pyrite.results.model", "SpectralDensity"),
    "SpectralFactors": ("pyrite.results.model", "SpectralFactors"),
    "Sweep": ("pyrite.campaign.model", "Sweep"),
    "SubstanceRef": ("pyrite.bodies", "SubstanceRef"),
    "Target": ("pyrite.campaign.target", "Target"),
    "fetch_tables": ("pyrite.xsgen.install", "fetch_tables"),
    "load_numerics": ("pyrite._config_presets", "load_numerics"),
    "load_physics": ("pyrite._config_presets", "load_physics"),
    "photon_energies": ("pyrite.materials.emission", "photon_energies"),
    "simulate": ("pyrite.api", "simulate"),
    "directional_source": ("pyrite.directional", "directional_source"),
    "directional_sources": ("pyrite.directional", "directional_sources"),
    "DirectionalSpectrum": ("pyrite.results.directional", "DirectionalSpectrum"),
}


def __getattr__(name: str) -> Any:
    """Resolve the public simulation surface lazily."""
    # import_module keeps these resolutions dynamic: a static ``from ... import``
    # here would make every ``import pyrite`` depend on the whole surface for the
    # import-linter layer contracts.
    from importlib import import_module

    if name == "states":
        module = import_module("pyrite.states")
        globals()[name] = module
        return module

    try:
        module_name, attribute = _PUBLIC_OBJECTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    from ._api_deprecations import ROOT_API_REPLACEMENTS, warn_api

    spelling = f"pyrite.{name}"
    if spelling in ROOT_API_REPLACEMENTS:
        # Do not cache compatibility aliases: every access must reach the warning
        # hook, including after callers change their warning filters.
        warn_api(spelling)
        return getattr(import_module(module_name), attribute)

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = ["DATA_DIR", "__version__", "states", *_PUBLIC_OBJECTS]


# Forward-looking domain contract. __all__ also preserves historical star imports
# throughout the support window; dir() lists both without resolving any object.
_ROOT_API = (
    "Beam",
    "Target",
    "Detector",
    "Scene",
    "Sweep",
    "Physics",
    "Numerics",
    "simulate",
    "Result",
    "photon_energies",
)


def __dir__() -> list[str]:
    """Discover retained and compatibility exports without importing their owners."""
    return sorted(set(globals()) | set(__all__))
