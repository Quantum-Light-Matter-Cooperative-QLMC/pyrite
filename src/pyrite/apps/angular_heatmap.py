"""Cache, form layout and plotting for the standalone directional-source app."""

import io
import json
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from .._numerics import CxrReflections, Numerics
from ..campaign.beam import Beam, PointBunch, PointSpot
from ..campaign.geometry import Footprint, Slab
from ..directional import DIRECTIONAL_COMPONENTS, directional_sources, source_line_ceiling_eV
from .angular_maps import AngularPanel

#: App labels and units of each directional component.
COMPONENT_LABELS = {
    "pxr-cbs": "PXR/CBS (incoherent)",
    "characteristic": "Characteristic lines",
    "bremsstrahlung": "Bremsstrahlung continuum",
}

#: Largest retained (tilt, angle, energy) density array, in float64 elements
#: (about 1.6 GB). A memory guard only; scoring work itself is uncapped.
MAX_RESULT_ELEMENTS = 200_000_000


@dataclass(frozen=True)
class MapConfig:
    """Submitted transport/scoring controls; display current is deliberately absent.

    ``stop_eV=None`` resolves the photon stop to the scene's kinematic PXR/CBS
    line ceiling, so relativistic forward lines are not truncated by the grid.
    ``width_mm=None`` selects the laterally infinite slab (limiting case only).
    The default 5 x 5 mm footprint is the project's historical bounded crystal.
    """

    material: str = "silicon"
    beam_keV: float = 30.0
    thickness_ang: float = 2000.0
    width_mm: float | None = 5.0
    height_mm: float | None = 5.0
    tilts_deg: tuple[float, ...] = (30.0, 45.0)
    tilt_azimuth_deg: float = 0.0
    plane_azimuth_deg: float = 0.0
    start_eV: float = 100.0
    stop_eV: float | None = None
    energy_points: int = 1025
    log_energy: bool = False
    angle_points: int = 72
    electrons: int = 128
    seed: int = 1
    max_reflections: int = 4
    transport_core: str = "auto"
    direction_batch: int = 16
    components: tuple[str, ...] = ("pxr-cbs",)

    def __post_init__(self):
        if not self.components or not set(self.components) <= set(DIRECTIONAL_COMPONENTS):
            raise ValueError(f"select components from {', '.join(DIRECTIONAL_COMPONENTS)}")
        tilts = np.asarray(self.tilts_deg, dtype=float)
        if not len(tilts) or not np.all(np.isfinite(tilts)):
            raise ValueError("supply at least one finite target tilt")
        if len(set(tilts)) != len(tilts):
            raise ValueError("target tilts must be distinct")
        if not self.thickness_ang > 0:
            raise ValueError("target thickness must be positive")
        dimensions = (self.width_mm, self.height_mm)
        if dimensions != (None, None) and not all(d is not None and d > 0 for d in dimensions):
            raise ValueError("footprint needs positive width and height, or neither (infinite)")
        stop = self.beam_keV * 1000 if self.stop_eV is None else self.stop_eV
        if not (10 <= self.start_eV < stop <= self.beam_keV * 1000):
            raise ValueError("photon bounds must satisfy 10 <= start < stop <= beam energy [eV]")
        if self.energy_points < 2 or self.angle_points < 4:
            raise ValueError("use at least 2 energy points and 4 periodic angle samples")
        if min(self.electrons, self.max_reflections, self.direction_batch) < 1 or self.seed < 0:
            raise ValueError("electrons, reflections and batch must be positive; seed nonnegative")
        if self.result_elements > MAX_RESULT_ELEMENTS:
            raise ValueError(
                f"result arrays need {8 * self.result_elements / 1e9:.1f} GB; "
                "reduce tilts, components, angles or energy points"
            )

    @property
    def result_elements(self) -> int:
        return len(self.tilts_deg) * len(self.components) * self.angle_points * self.energy_points

    @property
    def scoring_work(self) -> int:
        """Electron x component x direction x energy evaluations (resource feedback)."""
        return self.electrons * self.result_elements

    def beam(self) -> Beam:
        return Beam(self.beam_keV, transverse=PointSpot(), longitudinal=PointBunch())

    def target(self, tilt_deg: float) -> Slab:
        width, height = self.width_mm, self.height_mm
        footprint = None if width is None or height is None else Footprint(width, height)
        return Slab(
            self.material,
            thickness_ang=self.thickness_ang,
            tilt_deg=tilt_deg,
            tilt_azim_deg=self.tilt_azimuth_deg,
            footprint=footprint,
        )

    def numerics(self) -> Numerics:
        return Numerics(
            cxr_trials=self.electrons,
            brem_trials=self.electrons,
            transport_core=self.transport_core,
            cxr_reflections=CxrReflections(max_reflections=self.max_reflections),
        )

    def line_ceiling_eV(self) -> float:
        """Kinematic line ceiling over every tilt (no transport)."""
        return max(
            source_line_ceiling_eV(self.beam(), self.target(tilt), numerics=self.numerics())
            for tilt in self.tilts_deg
        )

    def energy_grid(self) -> np.ndarray:
        stop = self.line_ceiling_eV() if self.stop_eV is None else self.stop_eV
        stop = min(stop, self.beam_keV * 1000)
        if not stop > self.start_eV:
            raise ValueError(f"photon start must lie below the resolved stop {stop:g} eV")
        spacing = np.geomspace if self.log_energy else np.linspace
        return spacing(self.start_eV, stop, self.energy_points)


@lru_cache(maxsize=4)
def calculate_maps(config: MapConfig):
    """One transport per tilt; cache source arrays separately from plots/current.

    Returns one ``{component: DirectionalSpectrum}`` per tilt, in tilt order.
    """
    energy = config.energy_grid()
    return tuple(
        directional_sources(
            config.beam(),
            config.target(tilt),
            energy_eV=energy,
            angle_deg=np.linspace(-180, 180, config.angle_points, endpoint=False),
            plane_azimuth_deg=config.plane_azimuth_deg,
            numerics=config.numerics(),
            seed=config.seed,
            direction_batch=config.direction_batch,
            components=config.components,
        )
        for tilt in config.tilts_deg
    )


def calculate_maps_reporting_cache(config: MapConfig):
    """:func:`calculate_maps` plus whether the result came from the cache."""
    hits = calculate_maps.cache_info().hits
    maps = calculate_maps(config)
    return maps, calculate_maps.cache_info().hits > hits


def tilt_panels(maps, component: str, *, current_na: float):
    """Rate panels (ph/s/eV/sr) of one component for the given tilt maps."""
    return [
        AngularPanel(
            f"{COMPONENT_LABELS[component]} · tilt {by_component[component].case['tilt_deg']:g}°",
            by_component[component].angle_deg,
            by_component[component].energy_eV,
            by_component[component].rate(current_na),
        )
        for by_component in maps
    ]


def maps_npz(config: MapConfig, maps, *, current_na: float) -> bytes:
    """Compressed ``.npz`` of every computed tilt and component, for download.

    Per tilt and component ``t<i>_<component>_*``: ``density``
    (ph/incident-electron/eV/sr), ``rate`` (ph/s/eV/sr at ``current_na``);
    per tilt ``t<i>_supported``, ``t<i>_direction_lab`` and
    ``t<i>_direction_sample``; shared ``angle_deg``, ``energy_eV`` and
    ``tilt_deg``. ``metadata`` is JSON: the submitted config, current,
    units, line ceilings and each tilt's provenance identity digest.
    """
    first = next(iter(maps[0].values()))
    arrays = {
        "angle_deg": first.angle_deg,
        "energy_eV": first.energy_eV,
        "tilt_deg": np.array([next(iter(m.values())).case["tilt_deg"] for m in maps]),
    }
    for index, by_component in enumerate(maps):
        result = next(iter(by_component.values()))
        arrays |= {
            f"t{index}_supported": result.supported,
            f"t{index}_direction_lab": result.direction_lab,
            f"t{index}_direction_sample": result.direction_sample,
        }
        for name, spectrum in by_component.items():
            key = name.replace("-", "_")
            arrays |= {
                f"t{index}_{key}_density": spectrum.density,
                f"t{index}_{key}_rate": spectrum.rate(current_na),
            }
    metadata = {
        "config": asdict(config),
        "current_na": current_na,
        "units": {"density": first.units, "rate": "ph/s/eV/sr"},
        "components": {name: s.component for name, s in maps[0].items()},
        "line_window_ceiling_eV": [
            next(iter(m.values())).provenance.get("line_window_ceiling_eV") for m in maps
        ],
        "identity_digest": [next(iter(m.values())).provenance.get("identity_digest") for m in maps],
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, metadata=np.array(json.dumps(metadata, default=str)), **arrays)
    return buffer.getvalue()


def parse_tilts(text: str) -> tuple[float, ...]:
    """Tilts from a comma list and/or inclusive ``start:stop:step`` ranges [deg]."""
    tilts = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        try:
            fields = [float(field) for field in item.split(":")]
        except ValueError:
            raise ValueError(f"cannot read tilt {item!r}; use numbers or start:stop:step") from None
        if len(fields) == 1:
            tilts.append(fields[0])
        elif len(fields) == 3 and fields[2] > 0 and fields[1] >= fields[0]:
            count = int(np.floor((fields[1] - fields[0]) / fields[2] + 1e-9)) + 1
            tilts.extend(fields[0] + fields[2] * np.arange(count))
        else:
            raise ValueError(f"tilt range {item!r} must be start:stop:step with step > 0")
    return tuple(float(t) for t in tilts)
