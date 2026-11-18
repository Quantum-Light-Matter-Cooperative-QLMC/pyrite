"""Emission-side BremsLib tables with the lattice-coherent CBS share removed (#1250).

A crystalline layer's CXR route scores the CBS lines of its reflections; the
amorphous BremsLib continuum already holds their orientation-averaged
Debye--Waller share. Each crystalline layer therefore radiates its continuum
from tables whose DDCS has that share subtracted for exactly the reflections
the line route scores (:mod:`pyrite.montecarlo.spectrum.brem_lattice_share`).

Only emission changes. Transport keeps the full tables: the electron does lose
that energy, to the CBS lines instead of the continuum. Corrected tables keep
their parent's key and digest, which name the BremsLib source the coupled
partition check compares; the correction has its own identity marker and
per-case record.

Not corrected (recorded as such): layers without scored reflections, cases
with a channeling zone (their zone rows are not the Born line sum the share
describes), elements without a BremsLib table (their isotropic EEDL fallback),
and incident energies outside the kernel's range.

Validation: brem-lattice-coherent-fraction
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Private case key: ``False`` scores the uncorrected (double-counting)
#: continuum, for validation harnesses that compare the two.
LATTICE_SHARE_CASE_KEY = "_brem_lattice_share"

#: Private case key set by the case driver after a measured reflection
#: selection: ``{layer: (cutoff_invang, reflections)}`` of each selected radiator (``layer``
#: ``None`` for a single-crystal case). It lives only on the emission batch's
#: case copy, never on a stored or hashed case.
SCORED_CUTOFFS_CASE_KEY = "_scored_reflection_cutoffs"


class LatticeShareUncorrectedWarning(UserWarning):
    """A crystalline continuum still carries the lattice share its CBS lines score."""


_KERNEL = None
_CACHE: dict[tuple, tuple[Any, Any]] = {}
_CACHE_LIMIT = 64


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        from ...datasets import require_verified
        from ..spectrum.brem_lattice_share import LatticeShareKernel

        _KERNEL = LatticeShareKernel.load(require_verified("brem-lattice-share"))
    return _KERNEL


def _radiator(case: Mapping[str, Any], layer: int | None):
    if case.get("layer_radiators") is None:
        return case if case.get("crystal") else None
    radiators = case["layer_radiators"]
    if layer is None:
        # A one-layer stack is scored as a single view (layer ``None``).
        return radiators[0] if len(radiators) == 1 else None
    return radiators[layer]


def scored_reflections(case: Mapping[str, Any], layer: int | None) -> list:
    """Reflections the line route scores for ``layer`` (``None``: single-crystal case).

    Validation: brem-lattice-coherent-fraction
    """
    radiator = _radiator(case, layer)
    if radiator is None:
        return []
    cutoffs = case.get(SCORED_CUTOFFS_CASE_KEY)
    if cutoffs is None:
        if case.get("reflection_selection") is not None:
            raise RuntimeError(
                "a measured reflection selection reached the continuum without its "
                "selection record; pass the case through continuum_case(case, record)"
            )
        return list(radiator.get("hkl_list") or [])
    from .reflection_cutoff import iter_reflection_shells

    selection = case["reflection_selection"]
    if layer not in cutoffs and len(cutoffs) == 1:
        layer = next(iter(cutoffs))  # one-layer stack scored as a single view
    cutoff, expected = cutoffs[layer]
    hkls = [
        hkl
        for _, members in iter_reflection_shells(
            radiator["crystal"],
            float(radiator["B_ang2"]),
            float(cutoff),
            float(selection["shell_width_invang"]),
        )
        for hkl in members
    ]
    if len(hkls) != int(expected):
        raise RuntimeError(
            f"layer {layer}: rebuilt {len(hkls)} reflections below {cutoff} 1/A, but the "
            f"selection scored {expected}; the lattice share would not match the lines"
        )
    return hkls


def continuum_case(case: Mapping[str, Any], reflection_selection: Mapping | None):
    """The case the continuum is scored with: ``case`` plus the selected cutoffs.

    A measured selection leaves ``hkl_list`` unset; its record names each
    radiator's cutoff and reflection count, from which
    :func:`scored_reflections` rebuilds the scored set.
    """
    if reflection_selection is None:
        return case
    return {
        **case,
        SCORED_CUTOFFS_CASE_KEY: {
            r["layer"]: (r["cutoff_invang"], r["reflections"])
            for r in reflection_selection["radiators"]
        },
    }


def skip_reason(case: Mapping[str, Any], layer: int | None) -> str | None:
    """Why ``layer``'s continuum stays uncorrected, or ``None`` when it is corrected."""
    if case.get(LATTICE_SHARE_CASE_KEY, True) is False:
        return "disabled"
    if case.get("bremsstrahlung_model") != "bremslib":
        return "no BremsLib tables (EEDL continuum)"
    if case.get("channeling") not in (None, "off"):
        return "channeling zone"
    if _radiator(case, layer) is None:
        return "amorphous layer"
    if not scored_reflections(case, layer):
        return "no scored reflections"
    return None


def lattice_share_applies(case: Mapping[str, Any]) -> bool:
    """Whether any of ``case``'s layers radiates a corrected continuum.

    Decided from the case alone (no kernel load), for identity markers: a
    crystalline BremsLib radiator that scores reflections or selects them.

    Validation: brem-lattice-coherent-fraction
    """
    if case.get(LATTICE_SHARE_CASE_KEY, True) is False:
        return False
    if case.get("bremsstrahlung_model") != "bremslib":
        return False
    if case.get("channeling") not in (None, "off"):
        return False
    layers = [None] if case.get("layer_radiators") is None else range(len(case["layer_radiators"]))
    selecting = case.get("reflection_selection") is not None
    return any(
        (radiator := _radiator(case, layer)) is not None
        and (selecting or bool(radiator.get("hkl_list")))
        for layer in layers
    )


def warn_if_eedl_double_counts(case: Mapping[str, Any]) -> None:
    """Warn when a crystalline case scores its continuum from EEDL.

    The EEDL continuum is isotropic and is not corrected (#1250 follow-up), so
    its lattice-coherent share is counted again by the CBS lines: about 3--8 %
    of the total yield on the #642 scenes.

    Validation: brem-lattice-coherent-fraction
    """
    if case.get(LATTICE_SHARE_CASE_KEY, True) is False:
        return
    eedl = dict(case, bremsstrahlung_model="bremslib")
    if case.get("bremsstrahlung_model") != "bremslib" and lattice_share_applies(eedl):
        import warnings

        warnings.warn(
            "the EEDL continuum is not corrected for its lattice-coherent share, which "
            "the CBS lines also score; select bremsstrahlung_model='bremslib' to remove it",
            LatticeShareUncorrectedWarning,
            stacklevel=3,
        )


def layer_emission_tables(case, layer, composition, tables):
    """``tables`` with the lattice-coherent share removed for ``layer``'s radiator.

    Returns ``(tables, record)``; ``record`` lists each element's removed
    fraction and anything left uncorrected. Unchanged tables pass through.

    Validation: brem-lattice-coherent-fraction
    """
    reason = skip_reason(case, layer)
    if reason is not None or tables is None:
        return tables, dict(layer=layer, corrected=False, reason=reason or "no tables")
    from ..spectrum.brem_lattice_share import lattice_share_table, reflection_shell_weights

    radiator = _radiator(case, layer)
    hkls = scored_reflections(case, layer)
    composition = [(str(el), float(n)) for el, n in composition]
    key = (
        radiator["crystal"],
        float(radiator["B_ang2"]),
        tuple(sorted(tuple(int(i) for i in h) for h in hkls)),
        tuple(composition),
        tuple(sorted((el, t.key, t.digest) for el, t in tables.items())),
    )
    cached = _CACHE.get(key)
    if cached is None:
        kernel = _kernel()
        g, weights = reflection_shell_weights(
            radiator["crystal"], hkls, float(radiator["B_ang2"]), composition
        )
        out = dict(tables)
        elements = []
        for element, n in composition:
            if element not in tables:
                elements.append(dict(element=element, corrected=False, reason="no BremsLib table"))
                continue
            out[element], info = lattice_share_table(
                tables[element], kernel, element, n, g, weights[element]
            )
            elements.append(
                dict(
                    element=element,
                    corrected=True,
                    removed_fraction=info.removed_fraction,
                    clamped_nodes=info.clamped_nodes,
                )
            )
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()
        cached = _CACHE[key] = (
            out,
            dict(
                layer=layer,
                corrected=True,
                crystal=radiator["crystal"],
                reflections=len(hkls),
                shells=int(g.size),
                elements=elements,
            ),
        )
    return cached
