"""
materials.registry
============

Single source of truth for material scan grids and crystallographic scan
defaults.

This module is intentionally leaf-like: it imports NumPy for array literals but
does not import :mod:`cxr_mc.sweep` or :mod:`cxr_mc.config`. Driver modules take
typed projections from :data:`MATERIAL_CONFIGS` so per-material geometry grids
and orientation defaults live together without an import cycle.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NotRequired, TypedDict, cast

import numpy as np

ScalarOrSeq = float | Sequence[float] | np.ndarray


@dataclass(frozen=True)
class Layer:
    """One substrate-side layer of a stack.

    The film itself is represented by a sweep's material and thickness; stack
    layers are fixed beneath it for a run.
    """

    material: str
    thickness_ang: float
    beam_uvw: tuple | None = None
    azimuth_deg: float = 0.0


class CrystalParamsGrid(TypedDict):
    """Fixed per-material crystallography used by ``sweep.crystal_params``.

    ``hkl_list`` is present only for materials where the automatic dominant
    reflection family search is intentionally bypassed. ``hkl_list_reason``
    records why that bypass exists so pinned-vs-derived planes are visible in
    the registry.
    """

    B_ang2: float
    beam_uvw: tuple[int, int, int]
    E_grid: np.ndarray
    hkl_list: NotRequired[list[tuple[int, ...]]]
    hkl_list_reason: NotRequired[str]


class MaterialGrid(TypedDict):
    """The per-material scan grid spread into :class:`sweep.Sweep`.

    A registry key can also name a full stack. In that case ``crystal`` in the
    combined material row names the film crystal and ``stack`` holds the fixed
    substrate-side :class:`Layer` list.
    """

    thickness_ang: ScalarOrSeq
    energy_keV: ScalarOrSeq
    tilt_deg: ScalarOrSeq
    tilt_azim_deg: ScalarOrSeq
    E_grid_line: np.ndarray
    E_grid_brem: np.ndarray
    substrate: NotRequired[str]
    stack: NotRequired[tuple[Layer, ...]]


class MaterialConfig(TypedDict, total=False):
    """Combined material row.

    Scan-capable rows carry the :class:`MaterialGrid` fields. Crystal-capable
    rows carry the :class:`CrystalParamsGrid` fields. Named stacks use
    ``crystal`` to point at their film crystal while keeping their own scan grid.
    """

    label: str
    crystal: str
    B_ang2: float
    beam_uvw: tuple[int, int, int]
    E_grid: np.ndarray
    hkl_list: list[tuple[int, ...]]
    hkl_list_reason: str
    thickness_ang: ScalarOrSeq
    energy_keV: ScalarOrSeq
    tilt_deg: ScalarOrSeq
    tilt_azim_deg: ScalarOrSeq
    E_grid_line: np.ndarray
    E_grid_brem: np.ndarray
    substrate: str
    stack: tuple[Layer, ...]


def pm(*hkls: tuple[int, ...]) -> list[tuple[int, ...]]:
    """A list of reflections together with their negatives."""
    out = []
    for hkl in hkls:
        out += [tuple(hkl), tuple(-x for x in hkl)]
    return out


# Product target: few-layer 2H-MoTe2 with c = 13.41 A (two layers per cell).
_MOTE2_PRODUCT_LAYER_PITCH_ANG = 13.41 / 2.0

# Few-layer 2H-MoS2: c = 12.294 A (crystal_structures.toml), two layers per cell.
_MOS2_LAYER_PITCH_ANG = 12.294 / 2.0


MATERIAL_CONFIGS: dict[str, MaterialConfig] = {
    # Thickness study: total flux + CXR/brem ratio vs thickness at a few key
    # tilts (positive tilt_deg = reciprocal vector tilted toward the detector,
    # Zhai's convention = high flux). Single azimuth (pitch plane) and single
    # energy so plot_metric_vs(x="thickness_ang", hue="tilt_deg") has nothing
    # to silently collapse -- one clean curve per tilt.
    # For an energy comparison instead, add 40 to energy_keV and use hue="E0_keV".
    "hopg": {
        "label": "HOPG",
        "B_ang2": 0.8,
        "beam_uvw": (0, 0, 1),
        "E_grid": np.arange(100.0, 5000.0, 3.0),
        "hkl_list": pm((0, 0, 2), (0, 0, 4)),
        "hkl_list_reason": (
            "HOPG is fiber-textured, so only the basal (00l) c-axis reflections "
            "are coherent; skip the automatic dominant-reflections search."
        ),
        "thickness_ang": 1e4,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 10, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    # Layered h-BN: c-axis normal, so start with the basal 00l family like HOPG.
    "hbn": {
        "label": "h-BN",
        "B_ang2": 0.6,
        "beam_uvw": (0, 0, 1),
        "E_grid": np.arange(100.0, 5000.0, 3.0),
        "hkl_list": pm((0, 0, 2), (0, 0, 4)),
        "hkl_list_reason": (
            "Layered h-BN is scanned with the c-axis normal; keep the basal "
            "(00l) family explicit to match the HOPG orientation model."
        ),
        "thickness_ang": np.concat([np.logspace(2, 5, 6), np.logspace(5, 6, 2, endpoint=False)]),
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(80, 89, 10, endpoint=True),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(5.0, 1000.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    "diamond": {
        "label": "diamond",
        "B_ang2": 0.21,
        "beam_uvw": (4, 0, 0),
        "E_grid": np.arange(100.0, 5000.0, 2.0),
        "thickness_ang": 10e4,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    "silicon": {
        "label": "silicon",
        "B_ang2": 0.46,
        "beam_uvw": (4, 4, 0),
        "E_grid": np.arange(100.0, 5000.0, 3.0),
        "thickness_ang": 10e4,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    "mose2": {
        "label": "MoSe2",
        "B_ang2": 0.6,
        "beam_uvw": (0, 0, 2),
        "E_grid": np.arange(350.0, 1750.0, 3.0),
        "thickness_ang": 10e4,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    # isostructural with MoSe2; W has no NIST Mott table so transport falls back to
    # analytic screened-Rutherford screening for W (see montecarlo).
    "wse2": {
        "label": "WSe2",
        "B_ang2": 0.6,
        "beam_uvw": (0, 0, 2),
        "E_grid": np.arange(350.0, 2500.0, 3.0),
        "thickness_ang": 10e4,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    # 2H-MoTe2 (alpha) bulk, isostructural with MoSe2. Te has no NIST Mott table ->
    # transport falls back to analytic screened-Rutherford screening for Te (see
    # montecarlo), as for W/S/Pt/Hf/Zr.
    "mote2": {
        "label": "MoTe2",
        "B_ang2": 0.6,
        "beam_uvw": (0, 0, 2),
        "E_grid": np.arange(350.0, 2500.0, 3.0),
        "thickness_ang": 10e4,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    # 2H-MoTe2 product-page variant (few-layer, on sapphire substrate). Same
    # phonon/crystal params as bulk; only the lattice differs (crystal_structures.toml).
    "mote2_product": {
        "label": "MoTe2 (product)",
        "B_ang2": 0.6,
        "beam_uvw": (0, 0, 2),
        "E_grid": np.arange(350.0, 2500.0, 3.0),
        "thickness_ang": _MOTE2_PRODUCT_LAYER_PITCH_ANG * np.arange(3, 7),
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "substrate": "sapphire",
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    # 1T (CdI2-type): heavy metal at the ORIGIN -> every (00l) stays strong, so the
    # bright basal series marches up in energy with the tight c. These metals
    # (Pt/Hf/Zr) have no NIST Mott table -> analytic SR screening.
    "ptse2": {
        "label": "PtSe2",
        "B_ang2": 0.6,
        "beam_uvw": (0, 0, 1),
        "E_grid": np.arange(350.0, 3500.0, 3.0),
        "thickness_ang": 10e4,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    "hfse2": {
        "label": "HfSe2",
        "B_ang2": 0.6,
        "beam_uvw": (0, 0, 1),
        "E_grid": np.arange(350.0, 3500.0, 3.0),
        "thickness_ang": 10e4,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    "zrse2": {
        "label": "ZrSe2",
        "B_ang2": 0.6,
        "beam_uvw": (0, 0, 1),
        "E_grid": np.arange(350.0, 3500.0, 3.0),
        "thickness_ang": 10e4,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    # 2H disulfides, isostructural with WSe2/MoSe2 (small in-plane a -> bright). S has
    # no NIST Mott table -> analytic SR screening fallback.
    "ws2": {
        "label": "WS2",
        "B_ang2": 0.6,
        "beam_uvw": (0, 0, 2),
        "E_grid": np.arange(350.0, 2500.0, 3.0),
        "thickness_ang": 10e4,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(50.0, 3500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    "mos2": {
        "label": "MoS2",
        "B_ang2": 0.6,
        "beam_uvw": (0, 0, 2),
        "E_grid": np.arange(350.0, 2500.0, 3.0),
        "thickness_ang": _MOS2_LAYER_PITCH_ANG * 3,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "substrate": "sapphire",
        "E_grid_line": np.arange(50.0, 4500.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    # Named device stack: few-layer 2H-MoS2 on a thin thermal a-SiO2 (285 nm,
    # the common device oxide -- adjust to the actual wafer) over thick
    # crystalline Si. Run as `cxr scan mos2-on-sio2-si`.
    "mos2-on-sio2-si": {
        "label": "MoS2 on SiO2/Si",
        "crystal": "mos2",
        "thickness_ang": _MOS2_LAYER_PITCH_ANG * np.arange(3, 7),
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "stack": (Layer("sio2", 2850.0), Layer("silicon", 5e6)),
        "E_grid_line": np.arange(50.0, 4000.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    # c-cut sapphire: c-axis normal to the film.
    "sapphire": {
        "label": "sapphire",
        "B_ang2": 0.25,
        "beam_uvw": (0, 0, 1),
        "E_grid": np.arange(100.0, 5000.0, 3.0),
        "thickness_ang": 5e6,
        "energy_keV": [25, 30, 35],
        "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
        "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
        "E_grid_line": np.arange(50.0, 4000.0, 3.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
}


def _has_scan_grid(row: MaterialConfig) -> bool:
    return all(
        key in row
        for key in (
            "thickness_ang",
            "energy_keV",
            "tilt_deg",
            "tilt_azim_deg",
            "E_grid_line",
            "E_grid_brem",
        )
    )


def _has_crystal_params(row: MaterialConfig) -> bool:
    return "B_ang2" in row and "beam_uvw" in row and "E_grid" in row


def _scan_grid(row: MaterialConfig) -> MaterialGrid:
    scan = cast(MaterialGrid, row)
    grid: MaterialGrid = {
        "thickness_ang": scan["thickness_ang"],
        "energy_keV": scan["energy_keV"],
        "tilt_deg": scan["tilt_deg"],
        "tilt_azim_deg": scan["tilt_azim_deg"],
        "E_grid_line": scan["E_grid_line"],
        "E_grid_brem": scan["E_grid_brem"],
    }
    if "substrate" in scan:
        grid["substrate"] = scan["substrate"]
    if "stack" in scan:
        grid["stack"] = scan["stack"]
    return grid


def _crystal_grid(row: MaterialConfig) -> CrystalParamsGrid:
    crystal = cast(CrystalParamsGrid, row)
    grid: CrystalParamsGrid = {
        "B_ang2": crystal["B_ang2"],
        "beam_uvw": crystal["beam_uvw"],
        "E_grid": crystal["E_grid"],
    }
    if "hkl_list" in crystal:
        grid["hkl_list"] = crystal["hkl_list"]
    if "hkl_list_reason" in crystal:
        grid["hkl_list_reason"] = crystal["hkl_list_reason"]
    return grid


MATERIAL_GRIDS: dict[str, MaterialGrid] = {
    material: _scan_grid(row) for material, row in MATERIAL_CONFIGS.items() if _has_scan_grid(row)
}

CRYSTAL_PARAMS: dict[str, CrystalParamsGrid] = {
    material: _crystal_grid(row)
    for material, row in MATERIAL_CONFIGS.items()
    if _has_crystal_params(row)
}

MATERIAL_LABELS = {
    material: cast(str, row["label"])
    for material, row in MATERIAL_CONFIGS.items()
    if "label" in row
}

MATERIALS = tuple(MATERIAL_GRIDS)


def material_crystal_key(material: str) -> str:
    """Return the film/crystal key for a scan material or named stack."""
    if material not in MATERIAL_GRIDS:
        raise ValueError(f"unknown material {material!r} (have {list(MATERIAL_GRIDS)})")
    return MATERIAL_CONFIGS[material].get("crystal", material)


def material_scan_grid(material: str) -> MaterialGrid:
    """Return the raw scan grid for ``material``."""
    if material not in MATERIAL_GRIDS:
        raise ValueError(f"unknown material {material!r} (have {list(MATERIAL_GRIDS)})")
    return MATERIAL_GRIDS[material]


def crystal_config(material: str) -> CrystalParamsGrid:
    """Return fixed crystallographic scan defaults for a crystal material."""
    if material not in CRYSTAL_PARAMS:
        raise ValueError(f"unknown material {material!r} (have {list(CRYSTAL_PARAMS)})")
    return CRYSTAL_PARAMS[material]
