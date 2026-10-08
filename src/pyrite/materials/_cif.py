"""CIF loading adapters shared by the material catalog and crystal API."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from gemmi import SmallStructure


class _CrystalsAtomLike(Protocol):
    element: str
    coords_fractional: object
    occupancy: float


class _CrystalsCrystalLike(Protocol):
    lattice_parameters: tuple[float, float, float, float, float, float]
    volume: float

    def __iter__(self) -> Iterator[_CrystalsAtomLike]: ...


def crystals_crystal_to_crystal_info(
    crystal: _CrystalsCrystalLike, mosaic_fwhm_deg: float | None = None
) -> dict[str, object]:
    """Convert a ``crystals.Crystal``-like object to a CRYSTALS entry.

    Compatibility adapter for callers supplying an already expanded structure;
    it neither imports nor requires the former ``crystals`` dependency.
    Source: lattice parameters, fractional coordinates, and cell volume.
    The adapter is structural only; PyRITE owns X-ray scattering physics. It assumes each
    expanded site is fully occupied and rejects partial occupancy instead of
    silently treating it as a whole atom. In the P1 limiting case, sites are
    copied one-for-one; callers must expand higher symmetry before conversion.

    Validation: crystals-cif-adapter
    """
    a, b, c, alpha, beta, gamma = crystal.lattice_parameters
    lattice = {
        "system": "general",
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
    }
    basis = []
    for atom in crystal:
        occupancy = float(getattr(atom, "occupancy", 1.0))
        if not np.isclose(occupancy, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"PyRITE CIF imports require full occupancy; found {occupancy:g} for {atom.element}"
            )
        coords = np.mod(np.asarray(atom.coords_fractional, dtype=float), 1.0)
        basis.append((str(atom.element), coords))
    basis.sort(key=lambda site: (site[0], *site[1].tolist()))

    return {
        "lattice": lattice,
        "basis": basis,
        "V_cell": float(crystal.volume),
        "mosaic_fwhm_deg": mosaic_fwhm_deg,
    }


def load_crystal_from_cif(
    path: str | Path, mosaic_fwhm_deg: float | None = None
) -> dict[str, object]:
    """Load a CIF with Gemmi and return a CRYSTALS-compatible entry.

    Source: Gemmi's small-structure CIF API (https://gemmi.readthedocs.io/en/stable/chemistry.html).
    Gemmi's affine symmetry operations expand fractional sites modulo integers.
    Assumes fully occupied sites; partial occupancy is rejected. P1 copies
    sites one-for-one; special positions occur once per unit cell. The canonical
    adapter then retains the expanded fractional basis, cell parameters, and
    volume while PyRITE retains ownership of X-ray scattering physics.

    Validation: crystals-cif-adapter
    """
    import gemmi

    crystal = gemmi.read_small_structure(str(Path(path)))
    return gemmi_structure_to_crystal_info(crystal, mosaic_fwhm_deg=mosaic_fwhm_deg)


def gemmi_structure_to_crystal_info(
    crystal: SmallStructure, mosaic_fwhm_deg: float | None = None
) -> dict[str, object]:
    """Convert Gemmi's small structure to the internal crystal mapping.

    Source: Gemmi small-structure API, ``cell.parameters``, ``cell.volume``,
    ``Site.fract``, ``Site.occ``, and ``Op.apply_to_xyz``. Lengths are
    angstroms, angles degrees, volume cubic angstroms, coordinates fractional.
    Assumes fully occupied sites. P1 preserves the input basis modulo lattice
    translations; symmetry-equivalent special positions are emitted once.

    Validation: crystals-cif-adapter
    """
    import gemmi

    for site in crystal.sites:
        if not np.isclose(site.occ, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"PyRITE CIF imports require full occupancy; found {site.occ:g} for {site.element.name}"
            )
    if crystal.symops:
        operations = [gemmi.Op(triplet) for triplet in crystal.symops]
    elif crystal.spacegroup is not None:
        operations = list(crystal.spacegroup.operations())
    else:
        raise ValueError(
            "CIF must specify a recognized space group or explicit symmetry operations"
        )
    basis = []
    identity = gemmi.Op("x,y,z")
    for site in crystal.sites:
        orbit = []
        fractional = [site.fract.x, site.fract.y, site.fract.z]
        for operation in operations:
            # Preserve literal P1 coordinates without Gemmi's denominator round trip.
            coords = np.mod(
                np.asarray(
                    fractional if operation == identity else operation.apply_to_xyz(fractional)
                ),
                1.0,
            )
            # Compare modulo integer translations, including the 0/1 boundary.
            if any(
                np.all(np.abs((coords - previous + 0.5) % 1.0 - 0.5) <= 1e-12) for previous in orbit
            ):
                continue
            orbit.append(coords)
            basis.append((site.element.name, coords))
    basis.sort(key=lambda site: (site[0], *site[1].tolist()))
    return {
        "lattice": {
            "system": "general",
            **dict(
                zip(("a", "b", "c", "alpha", "beta", "gamma"), crystal.cell.parameters, strict=True)
            ),
        },
        "basis": basis,
        "V_cell": crystal.cell.volume,
        "mosaic_fwhm_deg": mosaic_fwhm_deg,
    }


__all__ = [
    "crystals_crystal_to_crystal_info",
    "gemmi_structure_to_crystal_info",
    "load_crystal_from_cif",
]
