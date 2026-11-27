"""Mosaic route resolution and its case/identity keys (#1185, #1207).

``Sweep.mosaic_route="auto"`` (the default) resolves per run: the sampled
crystallite average for incoherent emission on a mosaic crystal, the analytic
detector-stage width otherwise. Split from ``campaign.sweep`` to keep that
module within its line budget; ``campaign.sweep`` re-exports the names.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from ..materials import CATALOG

if TYPE_CHECKING:
    from .sweep import Sweep


#: Default mosaic route model (#1207): ``mosaic_route="auto"`` scores an
#: incoherent mosaic crystal with per-history sampled crystallites at
#: ``MOSAIC_SAMPLED_DRAWS_DEFAULT`` draws, qualified against a K=256 x 4
#: reference on HOPG 0.8 deg (docs/validation/materials/mosaic-mc-sampled.md).
#: Stamped as the case's ``mosaic_route_model`` only on cases that route
#: resolves to the sampled average, so other cases keep their identity.
MOSAIC_ROUTE_DEFAULT_MODEL = "mc-sampled-k256-v1"
MOSAIC_SAMPLED_DRAWS_DEFAULT = 256
MOSAIC_GH_NODES_DEFAULT = 5


def _sweep_mosaic_fwhm_rad(sweep: Sweep, crystal: str) -> float | None:
    """The run's mosaic FWHM [rad], or None for a perfect crystal."""
    assert sweep.target is not None
    if not sweep.target.mosaic:
        return None
    mosaic_deg = (
        sweep.mosaic_fwhm_deg
        if sweep.mosaic_fwhm_deg is not None
        else CATALOG.crystal(crystal).mosaic_fwhm_deg
    )
    return float(np.deg2rad(mosaic_deg)) if mosaic_deg else None


def project_mosaic_identity(
    sweep_payload: dict[str, Any], sweep: Sweep, crystal: str, coherent_emission: bool
) -> None:
    """Hash the mosaic route its cases run, so ``"auto"`` moves only sampled digests.

    A run that ``"auto"`` resolves to the analytic route records the historical
    ``"analytic"`` and 5 nodes, exactly what it ran before #1207. A run it
    resolves to the sampled average records that route, its draws and
    ``MOSAIC_ROUTE_DEFAULT_MODEL``, as its cases do.
    """
    mosaic = _sweep_mosaic_fwhm_rad(sweep, crystal) is not None
    route = _resolved_mosaic_route(sweep, mosaic, coherent_emission, warn=False)
    sampled = route == "mc-sampled" and mosaic
    if sweep_payload.get("mosaic_nodes") is None:
        sweep_payload["mosaic_nodes"] = (
            MOSAIC_SAMPLED_DRAWS_DEFAULT if sampled else MOSAIC_GH_NODES_DEFAULT
        )
    if sweep.mosaic_route == "auto":
        sweep_payload["mosaic_route"] = route
        if sampled:
            sweep_payload["mosaic_route_model"] = MOSAIC_ROUTE_DEFAULT_MODEL


def _resolved_mosaic_route(
    sweep: Sweep, mosaic: bool, coherent_emission: bool, *, warn: bool = True
) -> str:
    """The concrete route for ``sweep.mosaic_route``; ``"auto"`` follows #1207.

    Full-fiber runs keep ``"analytic"``, which they refuse with a mosaic, so an
    explicit ``"mc"`` stays their only mosaic route. Coherent emission cannot
    sum different crystallites' fields across histories, so it keeps the
    analytic detector-stage width and warns that its source spectrum is not
    mosaic-broadened.
    """
    route = sweep.mosaic_route
    if route not in ("auto", "analytic", "mc", "mc-sampled"):
        raise ValueError(f"mosaic_route must be auto, analytic, mc or mc-sampled: {route!r}")
    if route != "auto":
        return route
    if not mosaic or sweep.cxr_fiber is not None:
        return "analytic"
    if coherent_emission:
        if not warn:
            return "analytic"
        warnings.warn(
            "mosaic_route='auto' keeps the analytic detector-stage mosaic width for "
            "coherent emission: the source spectrum is not mosaic-broadened (#1207)",
            UserWarning,
            stacklevel=4,
        )
        return "analytic"
    return "mc-sampled"


def mosaic_case_keys(sweep: Sweep, cp: dict, coherent_emission: bool = False) -> dict[str, Any]:
    """Case mosaic keys: the mosaic FWHM on at most one route, plus its rule.

    Crystal mosaicity is None unless the run enables it AND the crystal has a
    mosaic_fwhm_deg (or the Sweep overrides it); None is a perfect crystal, so
    store_result adds no mosaic term (exact no-op). The exact MC routes do the
    broadening INSIDE mc_cxr_spectrum and turn the analytic store_result term
    OFF (mutually exclusive -- applying both double-counts). Validation: mosaic-mc-sampled
    """
    mosaic_fwhm_rad = _sweep_mosaic_fwhm_rad(sweep, cp["crystal"])
    route = _resolved_mosaic_route(sweep, mosaic_fwhm_rad is not None, coherent_emission)
    nodes = sweep.mosaic_nodes
    if nodes is None:
        sampled = route == "mc-sampled" and mosaic_fwhm_rad is not None
        nodes = MOSAIC_SAMPLED_DRAWS_DEFAULT if sampled else MOSAIC_GH_NODES_DEFAULT
    keys: dict[str, Any] = {"mosaic_mc_nodes": nodes}
    if mosaic_fwhm_rad is None or route == "analytic":
        return keys | {"mosaic_fwhm_rad": mosaic_fwhm_rad, "mosaic_mc_fwhm_rad": None}
    keys |= {"mosaic_fwhm_rad": None, "mosaic_mc_fwhm_rad": mosaic_fwhm_rad}
    if route == "mc-sampled":  # Gauss-Hermite case keys stay unchanged
        keys["mosaic_mc_sampling"] = "history"
        if sweep.mosaic_route == "auto":
            keys["mosaic_route_model"] = MOSAIC_ROUTE_DEFAULT_MODEL
    return keys
