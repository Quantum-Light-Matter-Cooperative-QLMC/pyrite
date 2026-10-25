"""Internal bulk Cherenkov maps; no interface or escaped-radiation observable."""

import io
import json
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from ..montecarlo.spectrum.cherenkov import (
    CHERENKOV_MODEL,
    ChargedPassage,
    ConstantIndex,
    frank_tamm_yield,
)
from ..montecarlo.spectrum.transition_radiation import beta_from_kinetic_eV
from .angular_maps import AngularPanel, form_template

VALUE_LABEL = "Internal ph/electron/eV/angular bin"


@dataclass(frozen=True)
class CherenkovConfig:
    """Straight passage in an explicitly declared constant-index dielectric.

    The plotted values integrate over each full-azimuth polar bin and over
    ``path_um``; they are not pointwise directional densities.
    """

    beam_keV: float = 300.0
    refractive_index: float = 1.46
    path_um: float = 1.0
    start_eV: float = 1.0
    stop_eV: float = 5.0
    energy_points: int = 81
    angle_bins: int = 180

    def __post_init__(self):
        values = (self.beam_keV, self.refractive_index, self.path_um, self.start_eV, self.stop_eV)
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError(
                "Cherenkov energies, index and path length must be finite and positive"
            )
        if self.start_eV >= self.stop_eV:
            raise ValueError("Cherenkov photon bounds must satisfy start < stop")
        if self.stop_eV >= self.beam_keV * 1000:
            raise ValueError("Cherenkov photons must lie below the beam kinetic energy")
        if self.energy_points < 2 or self.angle_bins < 2:
            raise ValueError("use at least two photon energies and polar bins")
        if self.energy_points * self.angle_bins > 4_000_000:
            raise ValueError("Cherenkov map exceeds 4 million bins; reduce grid dimensions")

    def energy_grid(self):
        return np.linspace(self.start_eV, self.stop_eV, self.energy_points)


@lru_cache(maxsize=4)
def calculate_cherenkov(config: CherenkovConfig):
    """Reuse the bulk source and its exact cone binning without transport.

    Validation: frank-tamm-homogeneous-dielectric
    """
    energy = config.energy_grid()
    edges = np.linspace(0, np.pi, config.angle_bins + 1)
    passage = ChargedPassage(
        beta_from_kinetic_eV(config.beam_keV * 1000), path_length_m=config.path_um * 1e-6
    )
    source = frank_tamm_yield(passage, energy, ConstantIndex(config.refractive_index))
    values = source.angular_histogram(edges).T * (config.path_um * 1e-6)
    panel = AngularPanel(
        f"Cherenkov internal source · n = {config.refractive_index:g}",
        np.rad2deg((edges[:-1] + edges[1:]) / 2),
        energy,
        values,
    )
    return panel, edges, source


def cherenkov_npz(config: CherenkovConfig) -> bytes:
    """Export internal per-bin yields, coordinates and declared source settings."""
    panel, edges, source = calculate_cherenkov(config)
    metadata = {
        "config": asdict(config),
        "model": CHERENKOV_MODEL,
        "observable": "internal homogeneous bulk source; no escape or interfaces",
        "units": VALUE_LABEL,
        "angular_normalization": "integrated over full-azimuth polar bin, not per sr",
        "dielectric": {"kind": "constant-index", "epsilon": config.refractive_index**2},
        "beta": source.passage.beta,
    }
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        energy_eV=panel.energy_eV,
        angle_deg=panel.angle_deg,
        angle_edges_rad=edges,
        yield_per_electron_per_eV_per_bin=panel.values,
        yield_per_m_per_eV=source.yield_per_m_per_eV,
        above_threshold=source.above_threshold,
        metadata=np.array(json.dumps(metadata)),
    )
    return buffer.getvalue()


def cherenkov_form(mo):
    """Small submitted form, independent of transported-source controls."""
    defaults = CherenkovConfig()
    return (
        mo.Html(
            form_template(
                {
                    "Passage": ("beam", "index", "path"),
                    "Photon band": ("start", "stop", "points", "bins"),
                }
            )
        )
        .batch(
            beam=mo.ui.number(0.1, 1e8, value=defaults.beam_keV, label="Beam kinetic energy (keV)"),
            index=mo.ui.number(
                0.01, 100, value=defaults.refractive_index, label="Declared index n"
            ),
            path=mo.ui.number(1e-6, 1e6, value=defaults.path_um, label="Straight path (µm)"),
            start=mo.ui.number(1e-6, 1e6, value=defaults.start_eV, label="Photon start (eV)"),
            stop=mo.ui.number(1e-6, 1e6, value=defaults.stop_eV, label="Photon stop (eV)"),
            points=mo.ui.number(2, 1_000_000, value=defaults.energy_points, label="Photon points"),
            bins=mo.ui.number(2, 36000, value=defaults.angle_bins, label="Polar bins (0–180°)"),
        )
        .form(submit_button_label="Calculate Cherenkov source", bordered=True)
    )


def submitted_config(value):
    """Translate the app's submitted form, preserving a bounded initial example."""
    if value is None:
        return CherenkovConfig()
    return CherenkovConfig(
        beam_keV=value["beam"],
        refractive_index=value["index"],
        path_um=value["path"],
        start_eV=value["start"],
        stop_eV=value["stop"],
        energy_points=int(value["points"]),
        angle_bins=int(value["bins"]),
    )
