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
        # Without a selection record the line route scores ``hkl_list``: the
        # build-time families of an automatic selection on routes that never
        # run the selector (directional; streamed before #1281).
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



def selected_case(case: Mapping[str, Any], reflection_selection: Mapping | None):
    """``case`` scoring exactly the selected reflections, in lines and continuum.

    Re-evaluations after a measured selection (adaptive batch means) must
    score the set the production spectra scored, not the build-time families.
    """
    out = dict(continuum_case(case, reflection_selection))
    if reflection_selection is None:
        return out
    if case.get("layer_radiators") is None:
        out["hkl_list"] = scored_reflections(out, None)
        return out
    layers = [
        None if radiator is None else dict(radiator, hkl_list=scored_reflections(out, layer))
        for layer, radiator in enumerate(case["layer_radiators"])
    ]
    out["layer_radiators"] = layers
    out["hkl_list"] = next(r for r in layers if r is not None)["hkl_list"]
    return out

def _uncorrected_reason(case: Mapping[str, Any], layer: int | None) -> str | None:
    """:func:`skip_reason` before the scored set is known (a selection in progress)."""
    if case.get(LATTICE_SHARE_CASE_KEY, True) is False:
        return "disabled"
    if case.get("bremsstrahlung_model") != "bremslib":
        return "no BremsLib tables (EEDL continuum)"
    if case.get("channeling") not in (None, "off"):
        return "channeling zone"
    if _radiator(case, layer) is None:
        return "amorphous layer"
    return None


def skip_reason(case: Mapping[str, Any], layer: int | None) -> str | None:
    """Why ``layer``'s continuum stays uncorrected, or ``None`` when it is corrected."""
    reason = _uncorrected_reason(case, layer)
    if reason is None and not scored_reflections(case, layer):
        return "no scored reflections"
    return reason


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


def _line_cutoff_keV(case: Mapping[str, Any]) -> float:
    """Electron energy below which the line route scores no CBS [keV]."""
    return float(case.get("E_cut_lines_keV", 5.0))


def _corrected_tables(radiator, hkls, composition, tables, floor_keV):
    """``tables`` with the share of ``hkls`` removed, plus the per-element record."""
    from ..spectrum.brem_lattice_share import lattice_share_table, reflection_shell_weights

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
            tables[element], kernel, element, n, g, weights[element], T_floor_keV=floor_keV
        )
        elements.append(
            dict(
                element=element,
                corrected=True,
                removed_fraction=info.removed_fraction,
                clamped_nodes=info.clamped_nodes,
            )
        )
    return out, elements, int(g.size)


def layer_emission_tables(case, layer, composition, tables):
    """``tables`` with the lattice-coherent share removed for ``layer``'s radiator.

    Returns ``(tables, record)``; ``record`` lists each element's removed
    fraction and anything left uncorrected. Unchanged tables pass through.
    Electrons below the line cutoff ``E_cut_lines_keV`` keep the full
    continuum: the line route scores no CBS from them.

    Validation: brem-lattice-coherent-fraction
    """
    reason = skip_reason(case, layer)
    if reason is not None or tables is None:
        return tables, dict(layer=layer, corrected=False, reason=reason or "no tables")
    radiator = _radiator(case, layer)
    hkls = scored_reflections(case, layer)
    composition = [(str(el), float(n)) for el, n in composition]
    floor = _line_cutoff_keV(case)
    key = (
        radiator["crystal"],
        float(radiator["B_ang2"]),
        tuple(sorted(tuple(int(i) for i in h) for h in hkls)),
        tuple(composition),
        tuple(sorted((el, t.key, t.digest) for el, t in tables.items())),
        floor,
    )
    cached = _CACHE.get(key)
    if cached is None:
        out, elements, shells = _corrected_tables(radiator, hkls, composition, tables, floor)
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()
        cached = _CACHE[key] = (
            out,
            dict(
                layer=layer,
                corrected=True,
                crystal=radiator["crystal"],
                reflections=len(hkls),
                shells=shells,
                line_cutoff_keV=floor,
                elements=elements,
            ),
        )
    return cached


def share_only_tables(case, layer, composition, tables, hkls):
    """Tables whose DDCS is the lattice share of ``hkls`` alone, for ``layer``'s radiator.

    Each element of ``composition`` with a BremsLib table gets
    ``base - corrected`` (the share :func:`layer_emission_tables` would remove
    for exactly ``hkls``); other entries pass through unchanged. The track-length
    scorer is linear in the table, so scoring these gives the continuum yield
    the correction removes. Returns ``None`` when ``layer`` is not corrected.

    Validation: brem-lattice-coherent-fraction

    Validation: reflection-yield-tolerance-selector
    """
    from dataclasses import replace

    import numpy as np

    if tables is None or _uncorrected_reason(case, layer) is not None or not hkls:
        return None
    composition = [(str(el), float(n)) for el, n in composition]
    corrected, _, _ = _corrected_tables(
        _radiator(case, layer), hkls, composition, tables, _line_cutoff_keV(case)
    )
    out = dict(tables)
    for element, _ in composition:
        if element not in tables:
            continue
        base, corr = tables[element], corrected[element]
        out[element] = replace(
            base,
            scaled_ddcs_mb_sr=np.asarray(base.scaled_ddcs_mb_sr)
            - np.asarray(corr.scaled_ddcs_mb_sr),
            scaled_sdcs_mb=np.asarray(base.scaled_sdcs_mb) - np.asarray(corr.scaled_sdcs_mb),
        )
    return out


#: Log-spaced continuum nodes per decade of the selection band. The continuum
#: and the share are smooth in photon energy apart from absorption-edge steps.
#: Nodes are ~2.4 % apart; an edge step costs a trapezoid error of a few 1e-4
#: of the in-band continuum per edge, far below the selector tolerance.
SELECTION_NODES_PER_DECADE = 96


class SelectionContinuum:
    """In-band continuum and lattice-share yields of each radiator layer (#1250 P5c).

    The reflection selector's net metric needs, for radiator ``layer``, the
    in-band yield of that layer's full BremsLib continuum (:meth:`background`)
    and of the share a shell's reflections remove from it (:meth:`share`). Both
    are scored by the production continuum estimator on the case's own
    continuum electrons and segments, with the same tables (and line-cutoff
    floor) the corrected continuum uses, on log-spaced nodes across
    ``band_eV`` integrated by the trapezoid rule.

    Validation: brem-lattice-coherent-fraction

    Validation: reflection-yield-tolerance-selector
    """

    def __init__(self, case, segments, n_hat, abs_layers, band_eV, *, groove=None, Ne=None):
        import numpy as np

        from .case_tables import _case_bremslib_tables

        lo, hi = (float(x) for x in band_eV)
        if not 0.0 < lo < hi:
            raise ValueError("selection band must satisfy 0 < start < stop")
        count = max(16, int(np.ceil(SELECTION_NODES_PER_DECADE * np.log10(hi / lo))) + 1)
        self.grid = np.geomspace(lo, hi, count)
        step = np.diff(self.grid)
        self.weights = np.zeros(count)
        self.weights[:-1] += 0.5 * step
        self.weights[1:] += 0.5 * step
        self.case = case
        self._scene = (segments, n_hat, abs_layers, groove, Ne)
        self._tables = _case_bremslib_tables(case)
        self._background: dict = {}

    def applies(self, layer: int | None) -> bool:
        """Whether ``layer``'s continuum is corrected, so its shells net out a share."""
        return self._tables is not None and _uncorrected_reason(self.case, layer) is None

    def _composition(self, layer):
        segments, _, abs_layers, _, _ = self._scene
        if int(segments.get("n_layers", 1)) == 1:
            return [(str(el), float(n)) for el, n in self.case["composition"]]
        return [(str(el), float(n)) for el, n in abs_layers[layer][2]]

    def _yield(self, layer, tables, tabled_only):
        import numpy as np

        from ..._backend import _to_cpu
        from .emission import _layer_brem_from_segments

        segments, n_hat, abs_layers, groove, Ne = self._scene
        density = _layer_brem_from_segments(
            segments,
            self.grid,
            self.case,
            n_hat,
            abs_layers,
            layer=layer,
            tables=tables,
            tabled_only=tabled_only,
            groove=groove,
            Ne=Ne,
        )
        return float(np.asarray(_to_cpu(density), dtype=float) @ self.weights)

    def background(self, layer: int | None) -> float:
        """In-band yield of ``layer``'s uncorrected continuum."""
        if layer not in self._background:
            self._background[layer] = self._yield(layer, self._tables, False)
        return self._background[layer]

    def share(self, layer: int | None, hkls: list) -> float:
        """In-band continuum yield the correction removes for ``hkls`` in ``layer``."""
        tables = share_only_tables(
            self.case, layer, self._composition(layer), self._tables, list(hkls)
        )
        return 0.0 if tables is None else self._yield(layer, tables, True)


def lattice_share_record(case: Mapping[str, Any], n_layers: int) -> dict | None:
    """Run provenance of the continuum correction: per layer, corrected or why not.

    ``None`` when no layer could be corrected (EEDL continuum, amorphous or
    channeling cases), so uncorrected runs gain no key. Each layer entry is
    :func:`layer_emission_tables`'s record (cached; no second table build):
    the removed fraction per element, or the skip reason.

    Validation: brem-lattice-coherent-fraction
    """
    from ..spectrum.brem_lattice_share import BREM_LATTICE_SHARE_MODEL
    from .case_tables import _case_bremslib_tables

    if not lattice_share_applies(case):
        return None
    tables = _case_bremslib_tables(case)
    if tables is None:
        return None
    if int(n_layers) == 1:
        views = [(None, case["composition"])]
    else:
        views = [(layer, case["abs_layers"][layer][2]) for layer in range(int(n_layers))]
    return dict(
        model=BREM_LATTICE_SHARE_MODEL,
        layers=[
            layer_emission_tables(case, layer, composition, tables)[1]
            for layer, composition in views
        ],
    )


def lattice_share_outputs(case: Mapping[str, Any], n_layers: int) -> dict:
    """``{"brem_lattice_share": record}`` for a spectrum output, or ``{}``."""
    record = lattice_share_record(case, n_layers)
    return {} if record is None else {"brem_lattice_share": record}


def selection_continuum(case: Mapping[str, Any], batch, Ne_brem):
    """Factory ``band_eV -> SelectionContinuum`` on an emission batch's segments."""
    return lambda band: SelectionContinuum(
        case,
        batch.segments,
        batch.n_hat,
        batch.abs_layers,
        band,
        groove=batch.groove,
        Ne=Ne_brem,
    )
