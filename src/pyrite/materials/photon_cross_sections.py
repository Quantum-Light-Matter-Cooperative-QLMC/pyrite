"""
materials.photon_cross_sections

Per-atom photon interaction cross sections from the Livermore Evaluated Photon
Data Library, EPDL2025 (EPICS2025, D. E. Cullen, IAEA ``NDS-IAEA-225``), for
Z=1--100 between 1 eV and 100 GeV.

Five integrated MF=23 channels are packaged:

  photoelectric   MT=522  total photoionization (all subshells)
  coherent        MT=502  Rayleigh scattering
  incoherent      MT=504  Compton scattering (bound, with S(x, Z))
  pair_nuclear    MT=517  pair production in the nuclear field  (> 2 m_e c^2)
  pair_electron   MT=515  triplet production in the electron field (> 4 m_e c^2)

Their sum is EPDL's MT=501 total, the per-atom cross section of the
narrow-beam (good-geometry) attenuation coefficient. Photonuclear absorption is
not in EPDL; it is a sub-percent giant-resonance term near 10--30 MeV and is
not modelled.

The table ``epdl2025_mf23.npz`` (a fetched dataset, ``pyrite tables fetch
epdl``; see :mod:`pyrite.datasets`) is derived from
the SHA-256-pinned upstream tape by ``scripts/release_epdl_table.py``. The only
modification is knot thinning under the upstream lin-lin law (ENDF
interpolation law 2) that reproduces every upstream node to the stored relative
``tolerance`` (5e-4). Evaluation is lin-lin, as declared upstream, and
right-continuous at photoionization edges, which the table stores as repeated
energies. Outside ``[EPDL_E_MIN_EV,
EPDL_E_MAX_EV]`` every channel is NaN: the data hold nothing there, and NaN
puts the band under the callers' out-of-domain policy.

This is atomic data access, not a physical model: a caller multiplies by a
number density to form an attenuation coefficient
(``materials.attenuation._mu_total_inv_ang``). Coherent couplings and the
refractive index keep the Chantler/Waasmaier--Kirfel factors of
:mod:`pyrite.materials.atomic`.
"""

import hashlib
from collections import OrderedDict
from functools import cache

import numpy as np

from .atomic import Z_TABLE

EPDL_TABLE_SHA256 = "fcc2f00c5bb969e99bc84cac16762f13e939f071d585c433a5c0f420913fcfc9"

#: Generation marker for the narrow-beam attenuation data, hashed into the
#: dataset and case-content identities like ``characteristic_model``: the
#: packaged table's digest prefix, so a regenerated table orphans results once.
ATTENUATION_MODEL = f"epdl2025-mf23-{EPDL_TABLE_SHA256[:12]}"

#: Energy band [eV] tabulated by every EPDL2025 MF=23 section.
EPDL_E_MIN_EV = 1.0
EPDL_E_MAX_EV = 1.0e11

#: Channel name -> ENDF MF=23 MT number, in summation order.
PHOTON_CHANNELS: dict[str, int] = {
    "photoelectric": 522,
    "coherent": 502,
    "incoherent": 504,
    "pair_nuclear": 517,
    "pair_electron": 515,
}

_BARN_TO_ANG2 = 1.0e-8

_TOTAL_MEMO: OrderedDict[tuple, np.ndarray] = OrderedDict()
_TOTAL_MEMO_MAX = 256


@cache
def _table() -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Verify and unpack the table into ``{Z: {MT: (E_eV, sigma_ang2)}}``."""
    from ..datasets import require_dataset

    path = require_dataset("epdl")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != EPDL_TABLE_SHA256:
        raise ValueError(
            f"EPDL table {path} checksum mismatch: expected {EPDL_TABLE_SHA256}, got {actual}"
        )
    with np.load(path) as archive:
        z = archive["z"].astype(int)
        mt = archive["mt"].astype(int)
        offsets = archive["offsets"]
        energy = archive["energy_eV"].astype(float)
        sigma = archive["sigma_barn"].astype(float) * _BARN_TO_ANG2
    out: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for k, (z_k, mt_k) in enumerate(zip(z, mt, strict=True)):
        e_k = energy[offsets[k] : offsets[k + 1]]
        s_k = sigma[offsets[k] : offsets[k + 1]]
        e_k.flags.writeable = False
        s_k.flags.writeable = False
        out.setdefault(int(z_k), {})[int(mt_k)] = (e_k, s_k)
    return out


def _atomic_number(element: str) -> int:
    try:
        z = Z_TABLE[element]
    except KeyError:
        raise KeyError(f"Unknown element symbol '{element}'.") from None
    if not 1 <= z <= 100:
        raise KeyError(f"EPDL2025 covers Z=1..100; '{element}' has Z={z}.")
    return z


def _interpolate(knots_e: np.ndarray, knots_s: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Evaluate one channel at in-band ``energy`` by the upstream lin-lin law.

    Below the first knot the channel is zero (it starts at a threshold). The
    interval index uses ``side="right"``, so an energy equal to an edge selects
    the interval that starts at the edge's upper (post-edge) value.

    Validation: narrow-beam-total-attenuation
    """
    out = np.zeros_like(energy)
    inside = energy >= knots_e[0]
    x = energy[inside]
    i = np.clip(np.searchsorted(knots_e, x, side="right") - 1, 0, knots_e.size - 2)
    e0, e1 = knots_e[i], knots_e[i + 1]
    s0, s1 = knots_s[i], knots_s[i + 1]
    out[inside] = s0 + (x - e0) / (e1 - e0) * (s1 - s0)
    return out


def photon_cross_sections_ang2(element: str, E_eV: object) -> dict[str, np.ndarray]:
    """Per-atom EPDL2025 cross section of each channel, in Angstrom^2.

    Parameters
    ----------
    element
        Element symbol with Z=1--100.
    E_eV
        Scalar or array photon energies in eV.

    Returns
    -------
    dict[str, numpy.ndarray]
        One array per :data:`PHOTON_CHANNELS` key, shaped like ``E_eV``; NaN
        outside ``[EPDL_E_MIN_EV, EPDL_E_MAX_EV]``.

    Raises
    ------
    KeyError
        If ``element`` is unknown or outside Z=1--100.

    Validation: narrow-beam-total-attenuation
    """
    tables = _table()[_atomic_number(element)]
    energy = np.asarray(E_eV, dtype=float)
    flat = np.atleast_1d(energy).ravel()
    in_band = (flat >= EPDL_E_MIN_EV) & (flat <= EPDL_E_MAX_EV)
    out = {}
    for name, mt in PHOTON_CHANNELS.items():
        values = np.full(flat.shape, np.nan)
        values[in_band] = _interpolate(*tables[mt], flat[in_band])
        out[name] = values.reshape(energy.shape)
    return out


def total_photon_cross_section_ang2(element: str, E_eV: object) -> np.ndarray:
    """Per-atom EPDL2025 total (sum of all five channels), in Angstrom^2.

    Memoized on the exact energy bytes: transport callers re-request the same
    element on the same grid once per layer or reflection.

    Returns
    -------
    numpy.ndarray
        Read-only cross sections shaped like ``E_eV``; NaN outside
        ``[EPDL_E_MIN_EV, EPDL_E_MAX_EV]``.

    Validation: narrow-beam-total-attenuation
    """
    energy = np.asarray(E_eV, dtype=float)
    key = (
        element,
        energy.shape,
        hashlib.blake2b(np.ascontiguousarray(energy).tobytes()).digest(),
    )
    cached = _TOTAL_MEMO.get(key)
    if cached is not None:
        _TOTAL_MEMO.move_to_end(key)
        return cached
    channels = photon_cross_sections_ang2(element, energy)
    total = sum(channels[name] for name in PHOTON_CHANNELS)
    total = np.asarray(total, dtype=float)
    total.flags.writeable = False
    _TOTAL_MEMO[key] = total
    if len(_TOTAL_MEMO) > _TOTAL_MEMO_MAX:
        _TOTAL_MEMO.popitem(last=False)
    return total


def photoelectric_edges(element: str) -> tuple[tuple[float, float], ...]:
    """EPDL2025 photoionization edges of ``element`` as ``(energy_eV, jump_ratio)``.

    An edge is a repeated energy in the MT=522 table: the cross section is
    discontinuous there, ``sigma(E_edge^-)`` below and ``sigma(E_edge)`` (the
    right-continuous value) above. ``jump_ratio`` is above/below; the
    first-ionization onset, where the value below is zero, is not an edge.

    Validation: narrow-beam-total-attenuation
    """
    energy, sigma = _table()[_atomic_number(element)][PHOTON_CHANNELS["photoelectric"]]
    out = []
    for k in np.flatnonzero(np.diff(energy) == 0.0):
        if sigma[k] > 0.0:
            out.append((float(energy[k]), float(sigma[k + 1] / sigma[k])))
    return tuple(out)
