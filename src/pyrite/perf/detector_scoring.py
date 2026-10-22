"""Per-primary cone-detector yields from the production coupled bremsstrahlung estimator.

Opt-in benchmark scoring for the #510 uncollided-detector arm. Every value is
the production expected-value estimator
:func:`pyrite.montecarlo.spectrum.brem_events.mc_coupled_brem_spectrum`
(BremsLib DDCS by track length along coupled tracks) in its per-electron band
mode, so the segments of one primary are summed before any moment is formed
and primaries without emission keep a zero score. Nothing here changes a
production default or physics model.

Two stages share one estimator call and differ only in the absorber:

``source``
    photons emitted at creation into the cone: the absorbing layer has zero
    number density, so the escape transmission is exactly one.
``escape``
    the same photons after uncollided (Beer--Lambert, EPDL2025 narrow-beam)
    escape through the slab and its finite footprint along the straight ray.

The cone integral uses :func:`pyrite.perf.comparison.converged_cone_scores`.
Photon energy is integrated per band with composite Gauss--Legendre panels
split at every EPDL2025 photoelectric edge of the slab, refined jointly with
the angular order, so convergence evidence covers both quadratures.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .comparison import (
    DETECTOR_BANDS_KEV,
    DETECTOR_STAGES,
    cone_solid_angle,
    converged_cone_scores,
    detector_field,
)

AZIMUTH_CONVENTION = "detector axis in the x-z plane of incidence, +x azimuth"
# Energy refinement is cheap (more nodes per estimator call) and angular
# refinement is not (one call per direction), so energy gets more nodes per
# level: ENERGY_NODES_PER_ORDER * order Gauss nodes on panels no wider than
# MAX_PANEL_KEV.
ENERGY_NODES_PER_ORDER = 2
MAX_PANEL_KEV = 100.0


def detector_axis(center_deg: float) -> NDArray[np.float64]:
    """Unit axis at polar angle ``center_deg`` from the slab normal ``+z``, in x-z."""
    theta = math.radians(center_deg)
    return np.array([math.sin(theta), 0.0, math.cos(theta)])


def band_panels(
    low_keV: float, high_keV: float, edges_keV: Sequence[float]
) -> list[tuple[float, float]]:
    """Split ``[low, high]`` at every interior edge, so each panel is smooth in mu.

    Panels wider than :data:`MAX_PANEL_KEV` are further divided evenly.
    """
    cuts = sorted({low_keV, high_keV, *(e for e in edges_keV if low_keV < e < high_keV)})
    panels = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        pieces = max(1, math.ceil((b - a) / MAX_PANEL_KEV))
        bounds = np.linspace(a, b, pieces + 1)
        panels.extend(zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True))
    return panels


def band_rule(
    panels: Sequence[tuple[float, float]], order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Composite Gauss-Legendre nodes and weights [eV] over the panels."""
    x, w = np.polynomial.legendre.leggauss(order)
    nodes, weights = [], []
    for a, b in panels:
        nodes.append(1e3 * (0.5 * (b - a) * x + 0.5 * (b + a)))
        weights.append(1e3 * 0.5 * (b - a) * w)
    return np.concatenate(nodes), np.concatenate(weights)


def absorption_edges_keV(composition: Sequence[tuple[str, float]]) -> list[float]:
    """EPDL2025 photoelectric edges [keV] of every element in the slab."""
    from pyrite.materials.photon_cross_sections import photoelectric_edges

    return sorted(
        {e / 1e3 for element, _ in composition for e, _jump in photoelectric_edges(element)}
    )


def cone_detector_scores(
    segments: Mapping[str, Any],
    *,
    composition: Sequence[tuple[str, float]],
    energy_keV: float,
    center_deg: float,
    half_angle_deg: float,
    bremslib_tables: Mapping[str, Any],
    rtol: float,
    start: int = 2,
    max_order: int = 32,
) -> dict[str, Any]:
    """Return per-primary cone yields [photons per incident primary] and evidence.

    ``segments`` is one coupled ``simulate_trajectories`` result
    (``radiative_model="bremslib-soft-hard"``) for a single-layer slab. The
    result maps every :func:`~pyrite.perf.comparison.detector_field` (bands
    and their 10 keV totals, both stages) to a ``(Ne,)`` array, plus an
    ``acceptance`` record of the converged quadrature.
    """
    from pyrite.montecarlo.spectrum.brem_events import mc_coupled_brem_spectrum

    radiative = segments.get("radiative") or {}
    if radiative.get("model") != "bremslib-soft-hard" or int(segments.get("n_layers", 1)) != 1:
        raise ValueError("the detector arm needs coupled single-layer transport rows")
    comp = [(str(element), float(n)) for element, n in composition]
    thickness = float(segments["thickness_ang"])
    absorbers = {
        "source": [(0.0, thickness, [(element, 0.0) for element, _ in comp])],
        "escape": [(0.0, thickness, comp)],
    }
    edges = absorption_edges_keV(comp)
    bands = [
        (band, band_panels(low, min(high, energy_keV), edges) if low < energy_keV else None)
        for band, low, high in DETECTOR_BANDS_KEV
    ]
    n_primaries = int(segments["Ne"])
    cutoff_eV = float(radiative["cutoff_eV"])

    def integrand(directions: NDArray[np.float64], order: int) -> NDArray[np.float64]:
        out = np.zeros((directions.shape[0], n_primaries, len(DETECTOR_STAGES) * len(bands)))
        n_energy = ENERGY_NODES_PER_ORDER * order
        rules = [None if p is None else band_rule(p, n_energy) for _b, p in bands]
        for i, n_hat in enumerate(directions):
            for s, stage in enumerate(DETECTOR_STAGES):
                for b, rule in enumerate(rules):
                    if rule is None:
                        continue
                    out[i, :, s * len(bands) + b] = mc_coupled_brem_spectrum(
                        segments,
                        rule[0],
                        cutoff_eV=cutoff_eV,
                        bremslib_tables=bremslib_tables,
                        composition=comp,
                        n_hat=n_hat,
                        layers=absorbers[stage],
                        electron_band_weights=rule[1],
                    )
        return out

    axis = detector_axis(center_deg)
    half_angle = math.radians(half_angle_deg)
    names = [detector_field(stage, band) for stage in DETECTOR_STAGES for band, _p in bands]
    exact_zero = [panels is None for _stage in DETECTOR_STAGES for _band, panels in bands]
    result = converged_cone_scores(
        integrand,
        axis,
        half_angle,
        rtol=rtol,
        exact_zero=exact_zero,
        start=start,
        max_order=max_order,
    )
    scores = {name: result["scores"][:, j] for j, name in enumerate(names)}
    for stage in DETECTOR_STAGES:
        scores[detector_field(stage, "ge10kev")] = sum(
            scores[detector_field(stage, band)] for band, _p in bands
        )
    acceptance = {
        "center_deg": float(center_deg),
        "half_angle_deg": float(half_angle_deg),
        "axis": axis.tolist(),
        "azimuth_convention": AZIMUTH_CONVENTION,
        "solid_angle_sr": cone_solid_angle(half_angle),
        "angular_rule": "Gauss-Legendre in cos(theta) x 2*order periodic azimuth",
        "energy_rule": f"composite Gauss-Legendre, {ENERGY_NODES_PER_ORDER}*order nodes per "
        f"panel, panels split at EPDL2025 photoelectric edges and at most {MAX_PANEL_KEV:g} keV "
        "wide; refined jointly with the angular order",
        "edges_keV": edges,
        "rtol": rtol,
        "order": result["order"],
        "converged": result["converged"],
        "fields": dict(zip(names, result["fields"], strict=False)),
        "estimator": "mc_coupled_brem_spectrum per-electron band weights (BremsLib DDCS)",
        "stages": {
            "source": "at creation into the cone, unit transmission",
            "escape": "uncollided straight-ray Beer-Lambert escape through slab and footprint",
        },
    }
    return {"scores": scores, "acceptance": acceptance}
