"""Derive one display identity — formula, phase, cut — for a catalog material.

Display labels used to be hand-authored strings in ``data/materials.toml`` that
re-stated data the catalog already held, and drifted from it: labels naming a
phase the crystal did not declare, labels quoting a pinned reflection order
where they meant the surface cut, and parentheses doing double duty for mineral
names. Everything here is derived from typed catalog fields instead, so a label
cannot contradict the record it describes.

The cut is the slab normal. A crystal declares it either as a direct-space
``beam_uvw`` direction or as a reciprocal-space ``surface_hkl`` plane normal --
:func:`pyrite.montecarlo.geometry._orientation_R` treats them as two spellings
of the same axis and rejects both at once. Notation therefore follows the
declaration: ``(hkl)`` for a plane, ``[uvw]`` for a direction. Nothing is
converted between the two, because that conversion is only exact for orthogonal
and hexagonal settings and the catalog holds neither symmetry claim.

Every packaged crystal declares ``surface_hkl``, because a slab normal is the
normal of the cut face and ``g_hkl`` is that normal by construction in any
lattice, whereas the direct axis coincides with it only under extra symmetry
the catalog does not assert. ``beam_uvw`` survives as a Sweep/Layer override
spelling; a crystal that used it would re-split the notation, which
``tests/materials/test_material_catalog.py`` guards against.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

MillerIndices = tuple[int, int, int]

#: Which space a cut's indices live in. ``"plane"`` renders ``(hkl)`` and comes
#: from ``surface_hkl``; ``"direction"`` renders ``[uvw]`` and comes from
#: ``beam_uvw``.
CutFrame = Literal["plane", "direction"]


def reduce_indices(indices: MillerIndices) -> MillerIndices:
    """Return ``indices`` divided by the GCD of their magnitudes.

    A slab normal is defined only up to a positive scale, so the catalog's
    unreduced spellings (``mos2`` declares ``[0, 0, 2]``, ``silicon`` declares
    ``[4, 4, 0]``) name the same axis as their primitive representatives. Those
    magnitudes are what leaked reflection orders such as ``(002)`` and ``(0004)``
    into the old hand-written labels, so reduction happens before rendering.
    """
    first, second, third = (int(component) for component in indices)
    divisor = math.gcd(math.gcd(abs(first), abs(second)), abs(third))
    if divisor <= 1:
        return first, second, third
    return first // divisor, second // divisor, third // divisor


def hexagonal_setting(lattice: Mapping[str, object]) -> bool:
    """Whether ``lattice`` is given in a hexagonal setting: ``a == b``, ``gamma
    == 120``, ``alpha == beta == 90``.

    Decides notation only, never geometry. The catalog stores every cell as a
    general triclinic description (``lattice["system"]`` is always
    ``"general"``), so the setting is recovered from the parameters. Rhombohedral
    crystals given on hexagonal axes -- sapphire, the 1T dichalcogenides --
    answer ``True``, which is what their literature notation follows.

    Tolerances are loose against float noise (CIF-derived ``gamma`` arrives as
    ``119.99999999999999``) and far tighter than the gap to the nearest
    non-hexagonal cell in the catalog, where every ``gamma`` is exactly 90 or
    exactly 120.
    """
    values = []
    for name in ("a", "b", "alpha", "beta", "gamma"):
        value = lattice.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        values.append(float(value))
    a, b, alpha, beta, gamma = values
    return (
        abs(a - b) <= 1.0e-6 * max(abs(a), abs(b))
        and abs(alpha - 90.0) <= 1.0e-6
        and abs(beta - 90.0) <= 1.0e-6
        and abs(gamma - 120.0) <= 1.0e-6
    )


def bravais_indices(indices: MillerIndices) -> tuple[int, int, int, int]:
    """Return the Miller--Bravais ``(hkil)`` spelling of plane ``indices``.

    ``i = -(h + k)`` is exact and integral, so the four-index form carries no
    information the three-index form lacked -- it makes the three equivalent
    a-axes of a hexagonal cell symmetric in the label, which is how this
    literature writes a basal cut (``(0001)``, not ``(001)``).

    Planes only. The direction transform is different and rational
    (``U = (2u - v) / 3``), so it needs denominators cleared and can renormalize
    the triple; :func:`format_indices` therefore leaves a ``[uvw]`` cut in three
    indices rather than guess a scale.
    """
    h, k, ell = indices
    return h, k, -(h + k), ell


def format_indices(indices: MillerIndices, frame: CutFrame, *, hexagonal: bool = False) -> str:
    """Render reduced ``indices`` in the bracket convention for ``frame``.

    Single-digit non-negative triples concatenate the way crystallographers
    write them (``(001)``); anything with a negative or multi-digit component
    would be ambiguous concatenated, so it is space-separated (``(1 0 -1)``).

    ``hexagonal`` renders a plane in four Miller--Bravais indices -- ``(0001)``
    rather than ``(001)``. It is ignored for a direction, per
    :func:`bravais_indices`.
    """
    reduced = reduce_indices(indices)
    components: tuple[int, ...] = reduced
    if hexagonal and frame == "plane":
        components = bravais_indices(reduced)
    if all(0 <= component <= 9 for component in components):
        body = "".join(str(component) for component in components)
    else:
        body = " ".join(str(component) for component in components)
    return f"({body})" if frame == "plane" else f"[{body}]"


@dataclass(frozen=True)
class MaterialIdentity:
    """Structured display identity for one runnable material.

    Parameters
    ----------
    formula
        ASCII chemical formula of the entrance-film crystal, e.g. ``"MoS2"``.
    phase
        Polytype or structural phase, e.g. ``"2H"``, ``"1T'"``, or ``None`` when
        the crystal declares none.
    full_name
        English name of the crystal, e.g. ``"Molybdenum Disulfide"``.
    cut
        Reduced slab-normal indices, or ``None`` when the material declares no
        orientation.
    cut_frame
        Which space :attr:`cut` lives in; ``None`` exactly when :attr:`cut` is.
    display_name
        Catalog override for the name segment, used where formula and phase do
        not produce the name the field uses (``"HOPG"``, ``"h-BN"``) or where the
        material is a stack rather than a single crystal.
    hexagonal
        Whether the crystal's cell is given on hexagonal axes, from
        :func:`hexagonal_setting`. Selects four-index Miller--Bravais rendering
        for a plane cut; affects notation only.
    """

    formula: str
    phase: str | None
    full_name: str | None
    cut: MillerIndices | None
    cut_frame: CutFrame | None
    display_name: str | None = None
    hexagonal: bool = False

    @property
    def name(self) -> str:
        """Name segment: the override when present, else ``<phase>-<formula>``."""
        if self.display_name is not None:
            return self.display_name
        if self.phase:
            return f"{self.phase}-{self.formula}"
        return self.formula

    @property
    def cut_text(self) -> str | None:
        """Rendered cut, or ``None`` when the material declares no orientation."""
        if self.cut is None or self.cut_frame is None:
            return None
        return format_indices(self.cut, self.cut_frame, hexagonal=self.hexagonal)

    @property
    def label(self) -> str:
        """Full display label: the name, then the cut when one is declared."""
        cut_text = self.cut_text
        return f"{self.name} {cut_text}" if cut_text else self.name
