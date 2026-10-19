"""
materials.photon_differential

Per-atom EPDL2025 data that photon interaction sampling needs beyond the
integrated channels of :mod:`pyrite.materials.photon_cross_sections`, for
Z=1--100 (EPICS2025, D. E. Cullen, IAEA ``NDS-IAEA-225``):

  subshell photoionization  MF=23 MT=534--599, with binding energy ``EPE``
  coherent form factor      MF=27 MT=502, ``F(x, Z)``
  incoherent scattering fn  MF=27 MT=504, ``S(x, Z)``

``x = sin(theta/2) / lambda`` is the momentum-transfer variable in 1/Angstrom,
tabulated from 0 to 1e9. A subshell designator is ``MT - 533``, the ENDF-6
convention EADL relaxation records use (1 = K, 2 = L1, ...), so photoionized
vacancies index :mod:`pyrite.montecarlo.eadl_relaxation` directly; EPDL2025
subshell sets and binding energies are identical to EADL2025's for Z=1--100.

The table ``epdl2025_differential.npz`` (``pyrite tables fetch
epdl-differential``; see :mod:`pyrite.datasets`) is derived from the
SHA-256-pinned upstream tape by ``scripts/release_epdl_table.py --table
differential`` with the same lin-lin knot thinning to ``5e-4`` relative as the
attenuation table. Evaluation is lin-lin, the upstream law, and
right-continuous at repeated knots. A subshell is zero below its first knot;
outside ``[EPDL_E_MIN_EV, EPDL_E_MAX_EV]`` (energies) or ``[0, X_MAX_INV_ANG]``
(momentum transfer) values are NaN.

Standalone CPU interaction samplers consume these tables (issue #341); no
production photon tracker is enabled. The sum of the subshells is the MT=522
photoelectric total within
the two tables' thinning tolerances; the attenuation table stays the owner of
channel competition.
"""

import hashlib
from dataclasses import dataclass
from functools import cache

import numpy as np

from .photon_cross_sections import EPDL_E_MAX_EV, EPDL_E_MIN_EV, _atomic_number, _interpolate

EPDL_DIFFERENTIAL_SHA256 = "d36a50d1add5c6f0b3df80a108e83aa8db8a4822445ca192534f6bc0a55bb166"

#: Largest tabulated momentum transfer [1/Angstrom] of every MF=27 section.
X_MAX_INV_ANG = 1.0e9

_SUBSHELL_MT_OFFSET = 533
_FORM_FACTOR_MT = 502
_SCATTERING_FUNCTION_MT = 504
_BARN_TO_ANG2 = 1.0e-8


@dataclass(frozen=True, slots=True)
class PhotoionizationSubshell:
    """One EPDL2025 subshell photoionization table (read-only arrays)."""

    designator: int
    binding_energy_eV: float
    energy_eV: np.ndarray
    cross_section_ang2: np.ndarray


@dataclass(frozen=True, slots=True)
class _ElementTables:
    subshells: tuple[PhotoionizationSubshell, ...]
    form_factor: tuple[np.ndarray, np.ndarray]
    scattering_function: tuple[np.ndarray, np.ndarray]


def _readonly(values: np.ndarray) -> np.ndarray:
    values.flags.writeable = False
    return values


@cache
def _table() -> dict[int, _ElementTables]:
    """Verify and unpack the table into per-element subshell and MF=27 tables."""
    from ..datasets import require_dataset

    path = require_dataset("epdl-differential")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EPDL_DIFFERENTIAL_SHA256:
        raise ValueError(
            f"EPDL differential table {path} checksum mismatch: "
            f"expected {EPDL_DIFFERENTIAL_SHA256}, got {actual}"
        )
    with np.load(path) as archive:
        z = archive["z"].astype(int)
        mf = archive["mf"].astype(int)
        mt = archive["mt"].astype(int)
        binding = archive["binding_eV"].astype(float)
        offsets = archive["offsets"]
        x = archive["x"].astype(float)
        y = archive["y"].astype(float)
    shells: dict[int, list[PhotoionizationSubshell]] = {}
    scattering: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for k in range(z.size):
        x_k = _readonly(x[offsets[k] : offsets[k + 1]])
        y_k = y[offsets[k] : offsets[k + 1]]
        z_k = int(z[k])
        if mf[k] == 23:
            shells.setdefault(z_k, []).append(
                PhotoionizationSubshell(
                    designator=int(mt[k]) - _SUBSHELL_MT_OFFSET,
                    binding_energy_eV=float(binding[k]),
                    energy_eV=x_k,
                    cross_section_ang2=_readonly(y_k * _BARN_TO_ANG2),
                )
            )
        else:
            scattering.setdefault(z_k, {})[int(mt[k])] = (x_k, _readonly(y_k))
    return {
        z_k: _ElementTables(
            subshells=tuple(shells[z_k]),
            form_factor=scattering[z_k][_FORM_FACTOR_MT],
            scattering_function=scattering[z_k][_SCATTERING_FUNCTION_MT],
        )
        for z_k in sorted(shells)
    }


def _evaluate(knots: tuple[np.ndarray, np.ndarray], values: object, lo: float, hi: float):
    """Lin-lin evaluation of one section at ``values``, NaN outside ``[lo, hi]``."""
    query = np.asarray(values, dtype=float)
    flat = np.atleast_1d(query).ravel()
    inside = (flat >= lo) & (flat <= hi)
    out = np.full(flat.shape, np.nan)
    out[inside] = _interpolate(*knots, flat[inside])
    return out.reshape(query.shape)


def photoionization_subshells(element: str) -> tuple[PhotoionizationSubshell, ...]:
    """EPDL2025 subshell photoionization tables of ``element``, by designator.

    Raises
    ------
    KeyError
        If ``element`` is unknown or outside Z=1--100.

    Validation: epdl-photon-differential-data
    """
    return _table()[_atomic_number(element)].subshells


def subshell_photoionization_cross_sections_ang2(
    element: str, E_eV: object
) -> tuple[tuple[int, ...], np.ndarray]:
    """Per-atom photoionization cross section of each subshell, in Angstrom^2.

    Returns
    -------
    tuple[tuple[int, ...], numpy.ndarray]
        Subshell designators and an array shaped ``(n_subshells, *E.shape)``;
        zero below a subshell's binding energy, NaN outside
        ``[EPDL_E_MIN_EV, EPDL_E_MAX_EV]``.

    Validation: epdl-photon-differential-data
    """
    subshells = photoionization_subshells(element)
    rows = [
        _evaluate((shell.energy_eV, shell.cross_section_ang2), E_eV, EPDL_E_MIN_EV, EPDL_E_MAX_EV)
        for shell in subshells
    ]
    return tuple(shell.designator for shell in subshells), np.stack(rows)


def coherent_form_factor(element: str, x_inv_ang: object) -> np.ndarray:
    """EPDL2025 atomic form factor ``F(x, Z)``; ``F(0) = Z``.

    ``x = sin(theta/2) / lambda`` in 1/Angstrom; NaN outside
    ``[0, X_MAX_INV_ANG]``.

    Validation: epdl-photon-differential-data
    """
    return _evaluate(_table()[_atomic_number(element)].form_factor, x_inv_ang, 0.0, X_MAX_INV_ANG)


def coherent_form_factor_table(element: str) -> tuple[np.ndarray, np.ndarray]:
    """Read-only lin-lin knots ``(x [Å⁻¹], F)`` for bounded rejection sampling.

    These are the same pinned knots used by ``coherent_form_factor``.
    Validation: epdl-photon-differential-data
    """
    return _table()[_atomic_number(element)].form_factor


def incoherent_scattering_function(element: str, x_inv_ang: object) -> np.ndarray:
    """EPDL2025 incoherent scattering function ``S(x, Z)``; ``S(0) = 0``, ``S -> Z``.

    ``x = sin(theta/2) / lambda`` in 1/Angstrom; NaN outside
    ``[0, X_MAX_INV_ANG]``.

    Validation: epdl-photon-differential-data
    """
    return _evaluate(
        _table()[_atomic_number(element)].scattering_function, x_inv_ang, 0.0, X_MAX_INV_ANG
    )


def incoherent_scattering_function_table(element: str) -> tuple[np.ndarray, np.ndarray]:
    """Read-only lin-lin knots ``(x [Å⁻¹], S)`` for bounded rejection sampling.

    These are the same pinned knots used by ``incoherent_scattering_function``.
    Validation: epdl-photon-differential-data
    """
    return _table()[_atomic_number(element)].scattering_function
