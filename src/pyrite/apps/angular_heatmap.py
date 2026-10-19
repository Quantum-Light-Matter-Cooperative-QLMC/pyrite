"""Bounded cache and plotting for the standalone directional-source app."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .._numerics import CxrReflections, Numerics
from ..campaign.beam import Beam, PointBunch, PointSpot
from ..campaign.geometry import Slab
from ..directional import directional_source


@dataclass(frozen=True)
class MapConfig:
    """Submitted transport/scoring controls; display current is deliberately absent."""

    material: str = "silicon"
    beam_keV: float = 30.0
    thickness_ang: float = 20.0
    tilts_deg: tuple[float, ...] = (30.0, 45.0)
    tilt_azimuth_deg: float = 0.0
    plane_azimuth_deg: float = 0.0
    start_eV: float = 100.0
    stop_eV: float = 5000.0
    energy_points: int = 129
    angle_points: int = 72
    electrons: int = 8
    seed: int = 1

    def __post_init__(self):
        if not self.tilts_deg or len(self.tilts_deg) > 4:
            raise ValueError("supply one to four target tilts")
        if not np.all(np.isfinite(self.tilts_deg)) or len(
            set(np.round(np.asarray(self.tilts_deg) * 2) / 2)
        ) != len(self.tilts_deg):
            raise ValueError(
                "target tilts must be finite and distinct after half-degree resolution"
            )
        if (
            self.electrons * self.angle_points * self.energy_points * len(self.tilts_deg)
            > 4_000_000
        ):
            raise ValueError(
                "quick local app scoring budget exceeded; reduce counts or use the API remotely"
            )
        if not (10 <= self.start_eV < self.stop_eV <= self.beam_keV * 1000):
            raise ValueError("photon bounds must satisfy 10 <= start < stop <= beam energy [eV]")
        if not 2 <= self.energy_points <= 2049 or not 8 <= self.angle_points <= 720:
            raise ValueError("use 2–2049 energy points and 8–720 periodic angle samples")
        if not 1 <= self.electrons <= 256 or not 0 < self.thickness_ang <= 2000:
            raise ValueError(
                "quick local app supports 1–256 electrons and at most 2000 Å; use remote for heavy studies"
            )


@lru_cache(maxsize=4)
def calculate_maps(config: MapConfig):
    """One transport per tilt; cache source arrays separately from plots/current."""
    beam = Beam(config.beam_keV, transverse=PointSpot(), longitudinal=PointBunch())
    return tuple(
        directional_source(
            beam,
            Slab(
                config.material,
                thickness_ang=config.thickness_ang,
                tilt_deg=tilt,
                tilt_azim_deg=config.tilt_azimuth_deg,
                footprint=None,
            ),
            energy_eV=np.linspace(config.start_eV, config.stop_eV, config.energy_points),
            angle_deg=np.linspace(-180, 180, config.angle_points, endpoint=False),
            plane_azimuth_deg=config.plane_azimuth_deg,
            numerics=Numerics(
                cxr_trials=config.electrons,
                brem_trials=config.electrons,
                cxr_reflections=CxrReflections(max_reflections=4),
            ),
            seed=config.seed,
        )
        for tilt in config.tilts_deg
    )


def heatmap_figure(maps, *, current_na=5.0, logarithmic=True):
    """Draw one panel per selected tilt; no transport or scoring is performed."""
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
    figure = Figure(figsize=(7 * len(maps), 5), layout="constrained")
    axes = figure.subplots(1, len(maps), squeeze=False)[0]
    for axis, result, rate in zip(axes, maps, rates, strict=True):
        # Duplicate the first periodic sample only for drawing circle closure.
        angle = np.append(result.angle_deg, result.angle_deg[0] + 360)
        value = np.concatenate((rate, rate[:1]), axis=0)
        colors = np.ma.masked_invalid(value.T)
        if logarithmic:
            colors = np.ma.masked_less_equal(colors, 0)
        image = axis.pcolormesh(
            angle,
            result.energy_eV,
            colors,
            shading="auto",
            norm=norm,
            cmap="inferno",
        )
        axis.set(
            xlim=(-180, 180),
            xlabel="Signed observation angle (deg)",
            ylabel="Emitted photon energy (eV)",
            title=f"Target tilt {result.case['tilt_deg']:g}° · {current_na:g} nA",
        )
        axis.set_facecolor("#d8d8d8")
        figure.colorbar(image, ax=axis, label="Photon intensity (ph/s/eV/sr)")
    return figure
