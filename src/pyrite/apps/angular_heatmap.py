"""Cache, form layout and plotting for the standalone directional-source app."""

from dataclasses import dataclass
from functools import lru_cache
from html import escape

import numpy as np

from .._numerics import CxrReflections, Numerics
from ..campaign.beam import Beam, PointBunch, PointSpot
from ..campaign.geometry import Footprint, Slab
from ..directional import directional_source, source_line_ceiling_eV

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

    def __post_init__(self):
        tilts = np.asarray(self.tilts_deg, dtype=float)
        if not len(tilts) or not np.all(np.isfinite(tilts)):
            raise ValueError("supply at least one finite target tilt")
        if len(set(np.round(tilts * 2) / 2)) != len(tilts):
            raise ValueError("target tilts must be distinct after half-degree resolution")
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
                "reduce tilts, angles or energy points"
            )

    @property
    def result_elements(self) -> int:
        return len(self.tilts_deg) * self.angle_points * self.energy_points

    @property
    def scoring_work(self) -> int:
        """Electron x direction x energy evaluations per run (resource feedback)."""
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
    """One transport per tilt; cache source arrays separately from plots/current."""
    energy = config.energy_grid()
    return tuple(
        directional_source(
            config.beam(),
            config.target(tilt),
            energy_eV=energy,
            angle_deg=np.linspace(-180, 180, config.angle_points, endpoint=False),
            plane_azimuth_deg=config.plane_azimuth_deg,
            numerics=config.numerics(),
            seed=config.seed,
            direction_batch=config.direction_batch,
        )
        for tilt in config.tilts_deg
    )


def form_template(groups: dict[str, tuple[str, ...]]) -> str:
    """HTML batch template: one labeled column per control group.

    Values name the batch keys rendered, in order, inside each group; the grid
    wraps to fewer columns on narrow screens.
    """
    columns = "".join(
        '<div class="pyrite-group" style="display:flex;flex-direction:column;gap:.55rem">'
        f'<div class="pyrite-shell pyrite-group__label">{escape(label)}</div>'
        + "".join(f"<div>{{{key}}}</div>" for key in keys)
        + "</div>"
        for label, keys in groups.items()
    )
    return (
        '<div style="display:grid;gap:1.25rem 1.5rem;'
        'grid-template-columns:repeat(auto-fit,minmax(19rem,1fr))">'
        f"{columns}</div>"
    )


def heatmap_figure(maps, *, current_na=5.0, logarithmic=True):
    """Draw one panel per selected tilt; no transport or scoring is performed.

    Panels share both axes and one color scale, so only the leftmost axis and
    the single right-hand colorbar carry labels. Gray marks unsupported
    (infinite-slab face-parallel) directions; zeros take the lowest color.
    """
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.figure import Figure

    rates = [result.rate(current_na) for result in maps]
    positive = np.concatenate([rate[np.isfinite(rate) & (rate > 0)] for rate in rates])
    norm = (
        LogNorm(
            vmin=max(float(positive.min()), float(positive.max()) * 1e-8),
            vmax=float(positive.max()) * (1 + 1e-12),
        )
        if logarithmic and len(positive)
        else Normalize(vmin=0, vmax=max(1.0, max(float(np.nanmax(rate)) for rate in rates)))
    )
    figure = Figure(figsize=(1.0 + 5.2 * len(maps), 4.8), layout="constrained")
    axes = figure.subplots(1, len(maps), squeeze=False, sharex=True, sharey=True)[0]
    log_energy = _geometric(maps[0].energy_eV)
    for axis, result, rate in zip(axes, maps, rates, strict=True):
        # Duplicate the first periodic sample only for drawing circle closure.
        angle = np.append(result.angle_deg, result.angle_deg[0] + 360)
        value = np.concatenate((rate, rate[:1]), axis=0)
        # Gray (masked) means unsupported only: zero and underflowed escape
        # share the log floor color with anything below it.
        if isinstance(norm, LogNorm):
            value = np.clip(value, norm.vmin, None)
        colors = np.ma.masked_invalid(value.T)
        image = axis.pcolormesh(
            angle, result.energy_eV, colors, shading="auto", norm=norm, cmap="inferno"
        )
        axis.set_title(f"Target tilt {result.case['tilt_deg']:g}°")
        axis.set_xticks(np.arange(-180, 181, 90))
        axis.set_facecolor("#d8d8d8")
        if log_energy:
            axis.set_yscale("log")
    axes[0].set(xlim=(-180, 180), ylabel="Emitted photon energy (eV)")
    figure.supxlabel("Signed observation angle (deg)")
    figure.colorbar(
        image, ax=list(axes), label=f"Photon intensity at {current_na:g} nA (ph/s/eV/sr)"
    )
    return figure


def _geometric(grid: np.ndarray) -> bool:
    ratio = grid[1:] / grid[:-1]
    return len(grid) > 2 and bool(np.allclose(ratio, ratio[0])) and ratio[0] > 1.01
