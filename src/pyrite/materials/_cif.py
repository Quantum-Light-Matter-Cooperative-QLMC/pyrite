"""CIF loading adapters shared by the material catalog and crystal API."""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np


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

    Source: ``Crystal.lattice_parameters``, ``Atom.coords_fractional``, and
    ``Crystal.volume`` from crystals 1.7. The adapter is structural only, so
    PyRITE remains the source of X-ray scattering physics. It assumes each
    expanded site is fully occupied and rejects partial occupancy instead of
    silently treating it as a whole atom. In the P1 limiting case, sites are
    copied one-for-one; for higher symmetry, ``Crystal.from_cif`` expands the
    asymmetric unit before this deterministic conversion.

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
    """Load a CIF with crystals 1.7 and return a CRYSTALS-compatible entry.

    ``Crystal.from_cif`` performs the CIF symmetry expansion. The canonical
    adapter then retains the expanded fractional basis, cell parameters, and
    volume while PyRITE retains ownership of X-ray scattering physics.

    Validation: crystals-cif-adapter
    """
    from crystals import Crystal

    crystal = Crystal.from_cif(Path(path))
    return crystals_crystal_to_crystal_info(crystal, mosaic_fwhm_deg=mosaic_fwhm_deg)


__all__ = ["crystals_crystal_to_crystal_info", "load_crystal_from_cif"]
