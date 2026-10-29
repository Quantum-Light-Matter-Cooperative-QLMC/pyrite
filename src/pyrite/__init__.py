"""pyrite -- crystal X-ray radiation (PXR + coherent bremsstrahlung) from
table-top electron beams in crystals.

The package is intentionally light at import time: ``import pyrite`` pulls in
no heavy dependencies (matplotlib / cupy / the MC pipeline). Use the curated
root for primary simulation objects and canonical domain namespaces for
specialized types, e.g. ``pyrite.beams.TransverseSpot`` and
``pyrite.geometry.Slab``. Canonical namespaces are imported lazily.

See the README for the scientific overview, ``docs/api.md`` for the supported
surface, and ``AGENTS.md`` for repository working conventions.
"""

import logging
from typing import TYPE_CHECKING, Any

from ._env import env_value
from .paths import data_dir

__version__ = "0.24.0"

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
    # Static declarations retain the approved remaining root objects.
    from pyrite._numerics import Physics as Physics
    from pyrite._numerics_groups import ExecutionConfig as ExecutionConfig
    from pyrite._numerics_groups import KernelTuning as KernelTuning
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

    # Domain module handles are part of the dot-access API, e.g. pr.beams.Beam.
    # These imports are static-only; normal `import pyrite` remains light.
    from . import beams as beams
    from . import campaign as campaign
    from . import config as config
    from . import datasets as datasets
    from . import detectors as detectors
    from . import execution as execution
    from . import experimental as experimental
    from . import geometry as geometry
    from . import materials as materials
    from . import observations as observations
    from . import optimization as optimization
    from . import plots as plots
    from . import quantum as quantum
    from . import radiation as radiation
    from . import results as results
    from . import storage as storage
    from . import transport as transport
    from . import units as units

_PUBLIC_OBJECTS = {
    "Analysis": ("pyrite.campaign.model", "Analysis"),
    "AcquisitionBatch": ("pyrite.instrument", "AcquisitionBatch"),
    "Beam": ("pyrite.campaign.beam", "Beam"),
    "Physics": ("pyrite._numerics", "Physics"),
    "PhotonTransport": ("pyrite._photon_config", "PhotonTransport"),
    "ExecutionConfig": ("pyrite._numerics_groups", "ExecutionConfig"),
    "KernelTuning": ("pyrite._numerics_groups", "KernelTuning"),
    "QuantumRegion": ("pyrite.quantum", "QuantumRegion"),
    "QuantumDynamics": ("pyrite.quantum", "QuantumDynamics"),
    "QuantumNumerics": ("pyrite.quantum", "QuantumNumerics"),
    "QuantumSetup": ("pyrite.quantum", "QuantumSetup"),
    "TransverseStateSpec": ("pyrite.quantum_state", "TransverseStateSpec"),
    "CxrSpectra": ("pyrite.results.model", "CxrSpectra"),
    "Detector": ("pyrite.detectors", "Detector"),
    "Numerics": ("pyrite.campaign.model", "Numerics"),
    "PixelMetadata": ("pyrite.results.model", "PixelMetadata"),
    "PixelRayMap": ("pyrite.results.model", "PixelRayMap"),
    "Provenance": ("pyrite.results.provenance", "Provenance"),
    "Result": ("pyrite.results.model", "Result"),
    "Scene": ("pyrite.campaign.model", "Scene"),
    "SpatialResult": ("pyrite.results.model", "SpatialResult"),
    "SpectralDensity": ("pyrite.results.model", "SpectralDensity"),
    "SpectralFactors": ("pyrite.results.model", "SpectralFactors"),
    "Sweep": ("pyrite.campaign.model", "Sweep"),
    "Target": ("pyrite.campaign.target", "Target"),
    "fetch_tables": ("pyrite.xsgen.install", "fetch_tables"),
    "photon_energies": ("pyrite.materials.emission", "photon_energies"),
    "simulate": ("pyrite.api", "simulate"),
    "DirectionalSpectrum": ("pyrite.results.directional", "DirectionalSpectrum"),
}

# Public domain *modules* are separate from the curated root object API.
# Listing a module here does not restore any removed root class aliases.
# Keep these lazy to preserve fast, low-dependency `import pyrite`.
_PUBLIC_MODULES = frozenset(
    {
        "beams",
        "campaign",
        "config",
        "datasets",
        "detectors",
        "execution",
        "experimental",
        "geometry",
        "materials",
        "observations",
        "optimization",
        "plots",
        "quantum",
        "radiation",
        "results",
        "storage",
        "transport",
        "units",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve curated root objects and approved submodules lazily."""
    from importlib import import_module

    if name == "states" or name in _PUBLIC_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    try:
        module_name, attribute = _PUBLIC_OBJECTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


# __all__ stays unchanged: star imports retain the curated 0.22.0 root object
# surface. Domain-module dot access does not reintroduce deprecated aliases.
__all__ = ["DATA_DIR", "__version__", "states", *_PUBLIC_OBJECTS]


# Forward-looking domain contract for the curated root object surface.
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
    """Discover root objects and public domain namespaces without importing them."""
    return sorted(set(globals()) | set(__all__) | _PUBLIC_MODULES)
