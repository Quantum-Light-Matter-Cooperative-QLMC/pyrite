"""Production ELSEPA elastic tables for catalog materials.

Coverage policy (issue #89):

* Every element uses a free-atom table (``MUFFIN 0``, high-energy
  factorization) over :data:`PRODUCTION_ENERGIES_EV`, 100 eV to 100 MeV.
* An *elementary* solid -- a catalog crystal holding one element -- replaces
  the rows at or below :data:`MUFFIN_TIN_CEILING_EV` with a muffin-tin table
  (``MUFFIN 1``, LDA-II absorption) whose radius is half the crystal's
  nearest-neighbour distance. ``elscata`` disables the high-energy
  factorization in muffin-tin mode, and its partial-wave series loses
  accuracy above about 1 MeV (round-off warnings; relative DCS errors of a few
  percent for W at 10 MeV), so the solid-state table stops there.
* Compounds use free-atom tables for every element. ELSEPA's muffin-tin model
  is defined for elementary solids only (one Z, ``NELEC = IZ``).

The muffin-tin model removes the long-range tail of the atomic potential, so
the *total* elastic cross section drops by about 30% relative to the free
atom (Si, 0.1-10 MeV) while the first transport cross section agrees to 1-2%.
The joined table therefore steps in total cross section between the last
muffin-tin node (1 MeV) and the first free-atom node above it; energy
interpolation turns the step into a ramp over that one grid interval. This is
an accepted, documented discontinuity, not a fitted blend.

Below :data:`ELASTIC_FLOOR_EV` the DHFS/muffin-tin treatment is not a
supported physical model, so tables start there and the sampler refuses
energies outside ``[ELASTIC_FLOOR_EV, ELASTIC_CEILING_EV]`` rather than
extrapolating.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.constants import Avogadro

from ...materials import CATALOG
from ...materials._transport_data import TRANSPORT_ELEMENTS
from ...materials.crystal import _direct_lattice_vectors
from .._errors import TableNotFoundError
from ..store import StoredTable, resolve
from .generate import (
    GenerationResult,
    element_request,
    generate_element,
    generate_muffin_tin,
    muffin_tin_request,
)

#: Lowest tabulated energy: the documented floor of the DHFS/muffin-tin model.
ELASTIC_FLOOR_EV = 100.0
#: Highest energy served by a muffin-tin table.
MUFFIN_TIN_CEILING_EV = 1.0e6
#: Highest tabulated energy (delta rays from the ~100 MeV beams of #13).
ELASTIC_CEILING_EV = 1.0e8

_DECADE_MANTISSAS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)

#: Production energy grid: ten log-spaced nodes per decade, 100 eV-100 MeV.
PRODUCTION_ENERGIES_EV = tuple(
    [m * 10.0**k for k in range(2, 8) for m in _DECADE_MANTISSAS] + [ELASTIC_CEILING_EV]
)

#: Arrays a transport consumer needs from a joined table.
_SAMPLER_FIELDS = ("mu", "energy_eV", "total_elastic_cm2", "transport1_cm2", "dcs_cm2_sr")


def nearest_neighbour_distance_ang(lattice, basis) -> float:
    """Return the shortest interatomic distance in a periodic crystal.

    Searches every pair of basis sites over the 5x5x5 block of neighbouring
    cells, which bounds the nearest neighbour for any cell whose shortest
    lattice vector is not shorter than half its longest -- true of every
    catalog crystal.
    """
    a1, a2, a3 = _direct_lattice_vectors(lattice)
    cell = np.vstack([a1, a2, a3])
    sites = np.asarray([np.asarray(frac, dtype=float) for _, frac in basis]) @ cell
    shifts = np.asarray(list(product(range(-2, 3), repeat=3)), dtype=float) @ cell
    best = np.inf
    for i, site in enumerate(sites):
        separations = sites[None, :, :] + shifts[:, None, :] - site
        distances = np.linalg.norm(separations, axis=-1)
        distances[shifts.shape[0] // 2, i] = np.inf  # the site itself
        best = min(best, float(distances.min()))
    if not np.isfinite(best) or best <= 0.0:
        raise ValueError("crystal has no positive nearest-neighbour distance")
    return best


@dataclass(frozen=True)
class ElementalSolid:
    """Inputs to one muffin-tin table for a one-element catalog crystal."""

    key: str
    element: str
    z: int
    radius_cm: float
    density_g_cm3: float


def _crystal_key(key: str) -> str | None:
    if key in CATALOG.materials:
        key = CATALOG.material(key).crystal_key
    return key if key in CATALOG.crystals else None


def elemental_solid(key: str | None) -> ElementalSolid | None:
    """Return muffin-tin inputs for an elementary catalog crystal, else ``None``.

    The radius is half the nearest-neighbour distance derived from the
    crystal's own CIF, the touching-sphere muffin-tin construction ELSEPA's
    ``Al.in`` example uses.
    """
    crystal_key = None if key is None else _crystal_key(str(key))
    if crystal_key is None:
        return None
    info = CATALOG.crystal(crystal_key).info
    if len(info.composition) != 1:
        return None
    element, number_density = info.composition[0]
    params = TRANSPORT_ELEMENTS[element]
    d_nn = nearest_neighbour_distance_ang(info.lattice, info.basis)
    density = float(number_density) * float(params["A"]) / (Avogadro * 1.0e-24)
    return ElementalSolid(
        key=crystal_key,
        element=element,
        z=int(params["Z"]),
        radius_cm=0.5 * d_nn * 1.0e-8,
        density_g_cm3=density,
    )


def elemental_solid_for_composition(
    composition: tuple[tuple[str, float], ...] | list[tuple[str, float]],
) -> ElementalSolid | None:
    """Return the elementary catalog crystal a one-element layer is made of.

    Matching is by element *and* number density (relative tolerance 1e-6), so
    a layer is identified by what it contains rather than by what a caller
    named it, and diamond and graphite stay distinct. Compounds, and
    one-element layers matching no catalog crystal, return ``None``.
    """
    if len(composition) != 1:
        return None
    element, number_density = composition[0]
    for key, crystal in CATALOG.crystals.items():
        crystal_composition = crystal.info.composition
        if len(crystal_composition) != 1 or crystal_composition[0][0] != element:
            continue
        if np.isclose(float(crystal_composition[0][1]), float(number_density), rtol=1e-6, atol=0.0):
            return elemental_solid(key)
    return None


def _muffin_tin_energies() -> tuple[float, ...]:
    return tuple(e for e in PRODUCTION_ENERGIES_EV if e <= MUFFIN_TIN_CEILING_EV)


@dataclass(frozen=True)
class ElementElastic:
    """The joined elastic table for one element of one transport layer.

    ``tables`` lists every stored table that contributed, for run identity.
    """

    element: str
    z: int
    tables: tuple[StoredTable, ...]
    arrays: dict[str, np.ndarray]


def joined_arrays(
    free_atom: dict[str, np.ndarray], muffin_tin: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """Join muffin-tin rows below the crossover to free-atom rows above it.

    Without a muffin-tin table this is the free-atom table restricted to the
    sampler's fields.
    """
    if muffin_tin is None:
        return {name: np.asarray(free_atom[name]) for name in _SAMPLER_FIELDS}
    if not np.array_equal(free_atom["mu"], muffin_tin["mu"]):
        raise ValueError("muffin-tin and free-atom tables use different angular grids")
    low = np.asarray(muffin_tin["energy_eV"]) <= MUFFIN_TIN_CEILING_EV
    if not np.any(low):
        raise ValueError("muffin-tin table has no rows at or below the crossover")
    top = float(np.asarray(muffin_tin["energy_eV"])[low].max())
    high = np.asarray(free_atom["energy_eV"]) > top
    joined = {"mu": np.asarray(free_atom["mu"])}
    for name in _SAMPLER_FIELDS[1:]:
        joined[name] = np.concatenate(
            [np.asarray(muffin_tin[name])[low], np.asarray(free_atom[name])[high]]
        )
    return joined


def _require(table: StoredTable | None, what: str, command: str) -> StoredTable:
    if table is None:
        raise TableNotFoundError(f"no ELSEPA table for {what}; generate it with '{command}'")
    return table


def resolve_layer_tables(
    composition: tuple[tuple[str, float], ...] | list[tuple[str, float]],
) -> tuple[ElementElastic, ...]:
    """Resolve the production elastic table of every element in one layer.

    Only a layer matching an elementary catalog crystal
    (:func:`elemental_solid_for_composition`) uses a muffin-tin table.
    Returned entries follow ``composition`` order.
    """
    solid = elemental_solid_for_composition(composition)
    out = []
    for element, _ in composition:
        z = int(TRANSPORT_ELEMENTS[element]["Z"])
        request, _ = element_request(z, PRODUCTION_ENERGIES_EV)
        free = _require(
            resolve(request.key),
            f"free atom {element} (Z={z})",
            "pyrite tables generate --code elsepa --material <catalog material "
            f"containing {element}>",
        )
        tables = [free]
        muffin_arrays = None
        if solid is not None and solid.element == element:
            mt_request, _ = muffin_tin_request(
                solid.key,
                solid.z,
                _muffin_tin_energies(),
                radius_cm=solid.radius_cm,
                density_g_cm3=solid.density_g_cm3,
            )
            muffin = _require(
                resolve(mt_request.key),
                f"elementary solid {solid.key!r}",
                f"pyrite tables generate --code elsepa --material {solid.key}",
            )
            tables.append(muffin)
            muffin_arrays = muffin.arrays()
        out.append(
            ElementElastic(
                element=element,
                z=z,
                tables=tuple(tables),
                arrays=joined_arrays(free.arrays(), muffin_arrays),
            )
        )
    return tuple(out)


def resolve_catalog_tables(key: str) -> tuple[StoredTable, ...]:
    """Every stored table a catalog material's elastic model reads, for identity."""
    return tuple(
        table for entry in resolve_layer_tables(catalog_composition(key)) for table in entry.tables
    )


def catalog_composition(key: str) -> tuple[tuple[str, float], ...]:
    """Return the composition of a catalog material, crystal, or medium."""
    crystal_key = _crystal_key(str(key))
    if crystal_key is not None:
        return tuple(CATALOG.crystal(crystal_key).info.composition)
    if key in CATALOG.media:
        return tuple(CATALOG.media[key].composition)
    choices = sorted(set(CATALOG.materials) | set(CATALOG.crystals) | set(CATALOG.media))
    raise ValueError(f"unknown catalog material {key!r}; choose one of: {', '.join(choices)}")


def generate_catalog(
    key: str, *, overwrite: bool = False, keep_on_failure: bool = False
) -> tuple[GenerationResult, ...]:
    """Generate every production table a catalog material's layer resolves."""
    results = []
    for element, _ in catalog_composition(key):
        z = int(TRANSPORT_ELEMENTS[element]["Z"])
        results.append(
            generate_element(
                z, PRODUCTION_ENERGIES_EV, overwrite=overwrite, keep_on_failure=keep_on_failure
            )
        )
    solid = elemental_solid(key)
    if solid is not None:
        results.append(
            generate_muffin_tin(
                solid.key,
                solid.z,
                _muffin_tin_energies(),
                radius_cm=solid.radius_cm,
                density_g_cm3=solid.density_g_cm3,
                overwrite=overwrite,
                keep_on_failure=keep_on_failure,
            )
        )
    return tuple(results)


__all__ = [
    "ELASTIC_CEILING_EV",
    "ELASTIC_FLOOR_EV",
    "MUFFIN_TIN_CEILING_EV",
    "PRODUCTION_ENERGIES_EV",
    "ElementElastic",
    "ElementalSolid",
    "catalog_composition",
    "elemental_solid",
    "elemental_solid_for_composition",
    "generate_catalog",
    "joined_arrays",
    "nearest_neighbour_distance_ang",
    "resolve_catalog_tables",
    "resolve_layer_tables",
]
