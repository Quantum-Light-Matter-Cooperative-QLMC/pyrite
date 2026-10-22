"""Inputs and per-history output of the opt-in PenRed (PENELOPE-2018) benchmark arm.

The scene is one rectangular slab in vacuum: ``z`` in ``[0, t]`` with the
lateral faces at ``x, y = +-w/2`` [cm], so a primary that leaves it is never
re-entered, as in the PyRITE first-exit transport. The pencil beam starts just
upstream of the origin with direction ``(sin a, 0, cos a)``.

The custom ``PYRITE_PRIMARY`` tally (``checks/equal_precision/penred``) writes
``T <hist> <terminal> <E_eV>`` for the primary's end and ``G <hist> <E_eV>``
for each hard bremsstrahlung photon of the primary at creation. Photons are
stacked because the material photon ``EABS`` is below the scoring threshold,
then absorbed before any transport by a body photon ``EABS`` above the beam
energy, so their creation energies are exact and no photon is transported.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..materials._transport_data import TRANSPORT_ELEMENTS
from .benchmark_matrix import SOURCE_BANDS_KEV
from .precision import grouped_scores

TERMINAL_CODES = {"transmitted": 5, "backscattered": 4, "side_exited": 6, "stopped": 7}
# Starting distance of the source upstream of the entrance face [cm].
SOURCE_STANDOFF_CM = 1e-7
# Material photon absorption energy [eV]: below the 10 keV score threshold, so
# every scored photon is stacked and reaches the tally.
MATERIAL_PHOTON_EABS_EV = 5e3
# Field count of each tally record type, including its tag.
RECORD_FIELDS = {"T": 4, "G": 3, "S": 4}


def mass_fractions(composition: Sequence[tuple[str, float]]) -> list[tuple[int, float]]:
    """``(Z, mass fraction)`` from elemental number densities, by CIAAW weights."""
    masses = [
        (TRANSPORT_ELEMENTS[el]["A"] * n, int(TRANSPORT_ELEMENTS[el]["Z"])) for el, n in composition
    ]
    total = sum(m for m, _z in masses)
    return [(z, m / total) for m, z in masses]


def _surface(label: int, name: str, transform: Sequence[tuple[str, float, str]]) -> str:
    lines = [
        "0" * 64,
        f"SURFACE ({label:4d})   {name}",
        "INDICES=( 0, 0, 0, 1, 0)",
    ]
    for key, value, unit in transform:
        lines.append(f"{key:>7s}=({value:+.15E},   0){unit}")
    return "\n".join(lines)


def geometry_text(thickness_cm: float, width_cm: float) -> str:
    """PENELOPE quadric file of the slab: z planes, then x and y planes by rotation."""
    if not thickness_cm > 0.0 or not width_cm > 0.0:
        raise ValueError("slab thickness and width must be positive")
    half = 0.5 * width_cm
    x_plane = [("THETA", 90.0, " DEG")]
    y_plane = [("THETA", 90.0, " DEG"), ("PHI", 90.0, " DEG")]
    surfaces = [
        (1, "Plane Z=0", [], +1),
        (2, "Plane Z=t", [("Z-SHIFT", thickness_cm, "")], -1),
        (3, "Plane X=-w/2", [*x_plane, ("X-SHIFT", -half, "")], +1),
        (4, "Plane X=+w/2", [*x_plane, ("X-SHIFT", half, "")], -1),
        (5, "Plane Y=-w/2", [*y_plane, ("Y-SHIFT", -half, "")], +1),
        (6, "Plane Y=+w/2", [*y_plane, ("Y-SHIFT", half, "")], -1),
    ]
    body = [
        "0" * 64,
        "MODULE  (   1)   Slab",
        "MATERIAL(   1)",
        *(f"SURFACE ({label:4d}), SIDE POINTER=({side:+d})" for label, _n, _t, side in surfaces),
        "0" * 64,
        "END      " + "0" * 55,
    ]
    blocks = [_surface(label, name, t) for label, name, t, _side in surfaces]
    return "PyRITE #510 benchmark slab\n\n\n" + "\n".join([*blocks, *body]) + "\n"


def config_text(
    *,
    composition: Sequence[tuple[str, float]],
    density_g_cm3: float,
    energy_keV: float,
    thickness_cm: float,
    width_cm: float,
    angle_deg: float,
    histories: int,
    seed_pair: int,
    electron_eabs_eV: float,
    wcr_eV: float,
    c1: float = 0.05,
    c2: float = 0.05,
    dsmax_cm: float | None = None,
) -> str:
    """``pen_main`` configuration of one single-threaded benchmark request.

    ``WCC`` equals the beam energy, so no hard inelastic collision (delta ray)
    occurs and all inelastic loss is soft. The photon tally is described in the
    module docstring. ``dsmax_cm`` defaults to a tenth of the slab thickness.
    """
    if histories < 2 or seed_pair < 0:
        raise ValueError("need at least two histories and a nonnegative seed pair")
    energy_eV = energy_keV * 1e3
    if not 0.0 < wcr_eV < MATERIAL_PHOTON_EABS_EV * 2 or electron_eabs_eV >= energy_eV:
        raise ValueError("WCR or electron EABS outside the benchmark range")
    angle = math.radians(angle_deg)
    dsmax = 0.1 * thickness_cm if dsmax_cm is None else dsmax_cm
    lines = [
        "simulation/threads 1",
        f"simulation/seedPair {seed_pair}",
        f"sources/generic/source1/nhist {histories}",
        'sources/generic/source1/kpar "electron"',
        'sources/generic/source1/direction/type "CONE"',
        f"sources/generic/source1/direction/theta {angle_deg!r}",
        "sources/generic/source1/direction/phi 0.0",
        "sources/generic/source1/direction/alpha 0.0",
        'sources/generic/source1/energy/type "MONOENERGETIC"',
        f"sources/generic/source1/energy/energy {energy_eV!r}",
        'sources/generic/source1/spatial/type "POINT"',
        f"sources/generic/source1/spatial/position/x {-math.tan(angle) * SOURCE_STANDOFF_CM!r}",
        "sources/generic/source1/spatial/position/y 0.0",
        f"sources/generic/source1/spatial/position/z {-SOURCE_STANDOFF_CM!r}",
        'tallies/primary/type "PYRITE_PRIMARY"',
        "tallies/primary/zmin 0.0",
        f"tallies/primary/zmax {thickness_cm!r}",
        f"tallies/primary/tolerance {1e-3 * thickness_cm!r}",
        'geometry/type "PEN_QUADRIC"',
        'geometry/input-file "slab.geo"',
        'geometry/processed-geo-file "report.geo"',
        f"geometry/dsmax/1 {dsmax!r}",
        f"geometry/eabs/1/gamma {2.0 * energy_eV!r}",
        "materials/slab/number 1",
        'materials/slab/filename "slab.mat"',
        "materials/slab/force-creation true",
        f"materials/slab/density {density_g_cm3!r}",
        *(f"materials/slab/elements/{z} {w!r}" for z, w in mass_fractions(composition)),
        f"materials/slab/eabs/electron {electron_eabs_eV!r}",
        f"materials/slab/eabs/positron {electron_eabs_eV!r}",
        f"materials/slab/eabs/gamma {MATERIAL_PHOTON_EABS_EV!r}",
        f"materials/slab/C1 {c1!r}",
        f"materials/slab/C2 {c2!r}",
        f"materials/slab/WCC {energy_eV!r}",
        f"materials/slab/WCR {wcr_eV!r}",
    ]
    if thickness_cm <= 0.0 or width_cm <= 0.0 or not dsmax > 0.0:
        raise ValueError("invalid slab or step length")
    return "\n".join(lines) + "\n"


def parse_tally(text: str, n_primaries: int) -> dict[str, Any]:
    """Primary terminals, transmitted energies and photon energies from the tally file.

    History numbers are made zero-based from the smallest terminal record;
    every primary must end exactly once and every photon must belong to one.
    Returns ``ids``/``terminal``/``energy_keV`` per primary, ``photon_ids``,
    ``photon_keV`` and the count of unscored secondaries.
    """
    records: dict[str, list[list[str]]] = {kind: [] for kind in RECORD_FIELDS}
    for line in text.splitlines():
        fields = line.split()
        if fields and len(fields) == RECORD_FIELDS.get(fields[0]):
            records[fields[0]].append(fields[1:])
        elif fields:
            raise ValueError(f"invalid PenRed tally record {line!r}")
    ends = np.array(records["T"], dtype=np.float64).reshape(-1, 3)
    photons = np.array(records["G"], dtype=np.float64).reshape(-1, 2)
    if n_primaries < 2 or len(ends) != n_primaries:
        raise ValueError("incomplete or duplicate primary terminal records")
    first = ends[:, 0].min()
    ids = (ends[:, 0] - first).astype(np.int64)
    if not np.array_equal(np.sort(ids), np.arange(n_primaries)):
        raise ValueError("incomplete or duplicate primary terminal records")
    terminal = ends[:, 1].astype(np.int64)
    energy = ends[:, 2] * 1e-3
    photon_ids = (photons[:, 0] - first).astype(np.int64)
    photon_keV = photons[:, 1] * 1e-3
    valid_ends = np.isin(terminal, list(TERMINAL_CODES.values())) & (energy >= 0.0)
    valid_photons = (photon_ids >= 0) & (photon_ids < n_primaries) & (photon_keV > 0.0)
    finite = np.all(np.isfinite(energy)) and np.all(np.isfinite(photon_keV))
    if not (finite and valid_ends.all() and valid_photons.all()):
        raise ValueError("invalid terminal state or photon primary/energy")
    return {
        "ids": ids,
        "terminal": terminal,
        "energy_keV": energy,
        "photon_ids": photon_ids,
        "photon_keV": photon_keV,
        "unscored_secondaries": len(records["S"]),
    }


def primary_scores(
    parsed: Mapping[str, Any], n_primaries: int, *, kind: str
) -> dict[str, NDArray[np.float64]]:
    """Per-primary ``source`` or ``transport`` scores of the comparison schema."""
    if kind not in ("source", "transport"):
        raise ValueError(f"PenRed arm has no {kind!r} record kind")
    ids, terminal = parsed["ids"], parsed["terminal"]
    scores = {
        name: grouped_scores(
            ids[terminal == code], np.ones(np.count_nonzero(terminal == code)), n_primaries
        )
        for name, code in TERMINAL_CODES.items()
    }
    forward = terminal == TERMINAL_CODES["transmitted"]
    scores["transmitted_energy_keV_per_primary"] = grouped_scores(
        ids[forward], parsed["energy_keV"][forward], n_primaries
    )
    if kind == "source":
        k = parsed["photon_keV"]
        bands = [
            ("photons_ge10kev", 10.0, math.inf),
            *SOURCE_BANDS_KEV,
            ("photons_ge100kev", 100.0, math.inf),
        ]
        for name, low, high in bands:
            mask = (k >= low) & (k < high)
            scores[name] = grouped_scores(
                parsed["photon_ids"][mask], np.ones(np.count_nonzero(mask)), n_primaries
            )
    return scores
