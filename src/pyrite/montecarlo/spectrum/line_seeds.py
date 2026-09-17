"""Deterministic feature seeds for piecewise line-axis windows (issue #101).

Each provider turns data the case already owns into
:class:`pyrite._line_windows.FeatureSeed` windows. Nothing here evaluates a pilot
spectrum, which can step over a narrow resonance: kinematic seeds read the
case's own transport segments, and edge and characteristic seeds read the same
atomic tables the emission models use.

Providers are looked up by name, and the names a policy enables are part of case
identity. An emission component adds windows by registering a provider with
:func:`register_seed_provider`; the planner has no component-specific cases.

Window extent and spacing are grid policy, not radiation physics. A windowed
spectrum is accepted through the refinement ladder, not through these rules.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._line_windows import FeatureSeed
from ...materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
from ..geometry import _mosaic_quadrature, _orientation_R
from ..transport import beta_from_keV

__all__ = [
    "CHARACTERISTIC_SOURCE",
    "DEFAULT_SAMPLES_PER_FEATURE",
    "DEFAULT_SEED_PROVIDERS",
    "DEFAULT_TAIL_WIDTHS",
    "EDGE_SOURCE",
    "KINEMATIC_SOURCE",
    "SeedContext",
    "absorption_edge_seeds",
    "characteristic_line_seeds",
    "collect_feature_seeds",
    "kinematic_line_seeds",
    "register_seed_provider",
    "seed_provider_names",
]

KINEMATIC_SOURCE = "pxr-kinematic"
EDGE_SOURCE = "absorption-edge"
CHARACTERISTIC_SOURCE = "characteristic"

#: Nodes across the narrowest shape-bearing feature. The issue's 8-10 sample
#: starting heuristic; the refinement ladder, not this number, certifies shape.
DEFAULT_SAMPLES_PER_FEATURE = 8
#: Sinc feature widths added beyond the weighted resonance band on each side,
#: so the lineshape of a segment at the band edge is inside its window.
DEFAULT_TAIL_WIDTHS = 2.0
#: Characteristic windows extend this many natural FWHM each side of the centre.
CHARACTERISTIC_EXTENT_FWHM = 10.0
#: Search half-width around a tabulated xraydb edge energy for the jump in the
#: Chantler ``f2`` table the attenuation model uses. The two tables disagree by
#: up to ~2.1% below 1 keV (N, O, F K; P L3), so the nominal energy is only a
#: locator.
EDGE_SEARCH_FRACTION = 0.05
#: Steepest adjacent Chantler ``f2`` ratio below which a shell carries no step
#: the tabulation resolves. Minor N/M/L1 shells sit at 1.00-1.04; every catalog
#: edge with an xraydb jump ratio of 2 or more is at 1.43 or above.
EDGE_MIN_F2_RATIO = 1.05
#: Native Chantler nodes kept on each side of the steepest bracket; the
#: measured jumps are complete within three.
EDGE_NATIVE_NODES = 3

#: Kinematic resonances below this are dropped by the line kernels too.
_MIN_RESONANCE_EV = 10.0


def _host(array):
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array)


def _weighted_quantiles(values, weights, probabilities):
    order = np.argsort(values, kind="stable")
    ordered, cumulative = values[order], np.cumsum(weights[order])
    targets = np.asarray(probabilities, dtype=float) * cumulative[-1]
    index = np.searchsorted(cumulative, targets, side="left")
    return ordered[np.minimum(index, ordered.size - 1)]


def kinematic_line_seeds(
    segments,
    n_hat,
    *,
    crystal: str,
    hkl_list: Iterable[Sequence[int]],
    feature_width_eV: float,
    samples_per_feature: int = DEFAULT_SAMPLES_PER_FEATURE,
    aliased_weight_limit: float = 1.0e-3,
    tail_widths: float = DEFAULT_TAIL_WIDTHS,
    beam_uvw=None,
    surface_hkl=None,
    azimuth_rad: float = 0.0,
    recip_miscut_rad=None,
    mosaic_fwhm_rad=None,
    mosaic_nodes: int = 1,
    electron_limit: int | None = None,
    label_prefix: str = "",
) -> tuple[list[FeatureSeed], dict[str, Any]]:
    """One window per reflection over its weighted resonance population.

    Source equation: the vacuum resonance the line kernels start from,
    ``E_res = hbar c (v . g) / (1 - v . n_hat)`` with ``v = beta v_hat``
    (Zhai SI Eq. 10; ``lines/_per_hkl.py`` step 1; ledger row
    ``line-energy-dispersion``). ``g`` is built exactly as the kernels build
    their rows: ``R_m R_orient g_hkl`` over the case orientation and each
    mosaic quadrature orientation, weighted by the quadrature weight.

    Assumptions: ``t_L**2`` is the intensity proxy, as in
    :func:`~pyrite.montecarlo.spectrum.diagnostics.sinc_feature_spacing`;
    ``|A|**2``, absorption, and the line-electron energy cutoff are ignored, so
    the band is conservative. The kernels solve the in-medium root
    (``lines/_kernels.py::_in_medium_kinematics``), which sits
    ``dE ~ -E delta beta cos(Theta) / (1 - beta cos(Theta))`` from the vacuum
    root. In a segment's own feature widths that is ``delta cos(Theta) L /
    lambda``, which ``delta`` does not bound: the ``tail_widths`` margin does
    not cover it per segment (up to 5.4 widths above 300 eV at hopg 300 keV;
    18-35 below 300 eV, where ``Re n > 1``). Coverage rests on the weighted
    quantile band being much wider than that shift, not on the margin --
    measured in-medium weight inside the window matched vacuum to 6e-6
    (``docs/validation/beam-transport/line-window-seeding.md``).

    Coverage: each window spans the ``[eps/2, 1 - eps/2]`` weighted quantiles,
    ``eps = aliased_weight_limit``, widened by ``tail_widths`` feature widths.
    Reflections are dropped smallest-weight first while their cumulative share
    of all radiating weight stays within ``eps``, so at most about ``2 eps`` of
    the proxy weight is left to the backbone. Spacing is
    ``feature_width_eV / samples_per_feature``.

    ``feature_width_eV`` resolves the ``eps``-quantile feature rather than the
    narrowest one, so the summary also reports ``narrowest_feature_width_eV``
    and ``samples_at_narrowest``. Measured on hopg and wse2 at 30-100 keV the
    two widths coincide to within 11% -- the longest single flight floors
    ``t_L``, leaving the width distribution no narrow tail -- so the nominal
    ``samples_per_feature`` nodes do land across the narrowest shape-bearing
    feature. A case where that stops being true shows it in
    ``samples_at_narrowest`` instead of under-resolving silently.

    Limiting case: a single straight flight gives a window centred on its
    closed-form resonance with ``tail_widths`` feature widths on each side.

    Returns ``(seeds, summary)``.

    Validation: line-window-seeding
    """
    width = float(feature_width_eV)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("feature_width_eV must be finite and positive")
    if int(samples_per_feature) < 1:
        raise ValueError("samples_per_feature must be at least 1")
    epsilon = float(aliased_weight_limit)
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("aliased_weight_limit must be in [0, 1)")
    spacing = width / int(samples_per_feature)
    margin = float(tail_widths) * width

    energy_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    energy = _host(segments[energy_field]).astype(float, copy=False)
    length = _host(segments["L_ang"]).astype(float, copy=False)
    direction = _host(segments["v_hat"]).astype(float, copy=False).reshape(-1, 3)
    if electron_limit is not None:
        line = _host(segments["elec_id"]) < int(electron_limit)
        energy, length, direction = energy[line], length[line], direction[line]
    beta = beta_from_keV(energy)
    velocity = beta[:, None] * direction
    denominator = 1.0 - velocity @ np.asarray(n_hat, dtype=float)
    weight = (length / beta) ** 2
    usable = np.isfinite(weight) & (weight > 0.0) & np.isfinite(denominator) & (denominator > 0.0)
    velocity, denominator, weight = velocity[usable], denominator[usable], weight[usable]

    # The spacing above resolves the eps-quantile feature, not the narrowest one.
    # Measured over hopg/wse2 at 30-100 keV the two coincide to within 11% -- the
    # longest single flight floors t_L, so the width distribution has no narrow
    # tail -- but that is a property of these cases, not a theorem. Report the
    # narrowest feature so a case where the quantile drifts above it is visible
    # instead of silently under-resolved.
    flight_time = np.sqrt(weight)
    narrowest = (
        float((2.0 * np.pi * HBARC_EV_ANG / (denominator * flight_time)).min())
        if flight_time.size
        else float("nan")
    )

    lattice = CRYSTALS[crystal]["lattice"]
    rotation = _orientation_R(
        lattice, beam_uvw, azimuth_rad, recip_miscut_rad, surface_hkl=surface_hkl
    )
    orientations = _mosaic_quadrature(mosaic_fwhm_rad, mosaic_nodes) or [(None, 1.0)]

    populations = []
    for hkl in hkl_list:
        g_vector, _magnitude = reciprocal_g_vector(hkl, lattice)
        if rotation is not None:
            g_vector = rotation @ g_vector
        energies, weights = [], []
        for mosaic_rotation, mosaic_weight in orientations:
            g_row = g_vector if mosaic_rotation is None else mosaic_rotation @ g_vector
            with np.errstate(divide="ignore", invalid="ignore"):
                resonance = HBARC_EV_ANG * (velocity @ g_row) / denominator
            radiating = np.isfinite(resonance) & (resonance > _MIN_RESONANCE_EV)
            energies.append(resonance[radiating])
            weights.append(weight[radiating] * float(mosaic_weight))
        label = label_prefix + "(" + " ".join(str(int(index)) for index in hkl) + ")"
        population_energy = np.concatenate(energies)
        population_weight = np.concatenate(weights)
        populations.append(
            (label, population_energy, population_weight, float(population_weight.sum()))
        )

    total = sum(entry[3] for entry in populations)
    summary: dict[str, Any] = {
        "reflections": len(populations),
        "coverage": 1.0 - epsilon,
        "feature_width_eV": width,
        "narrowest_feature_width_eV": narrowest,
        "samples_at_narrowest": narrowest / spacing,
        "spacing_eV": spacing,
        "dropped_reflections": [],
        "dropped_weight_fraction": 0.0,
    }
    if total <= 0.0:
        summary["dropped_reflections"] = [entry[0] for entry in populations]
        return [], summary

    dropped_weight = 0.0
    dropped = set()
    for label, _energy, _weight, share in sorted(populations, key=lambda entry: entry[3]):
        if share == 0.0 or dropped_weight + share <= epsilon * total:
            dropped_weight += share
            dropped.add(label)
            continue
        break

    seeds = []
    for label, population_energy, population_weight, _share in populations:
        if label in dropped:
            continue
        low, centre, high = _weighted_quantiles(
            population_energy,
            population_weight,
            (0.5 * epsilon, 0.5, 1.0 - 0.5 * epsilon),
        )
        seeds.append(
            FeatureSeed(
                source=KINEMATIC_SOURCE,
                label=label,
                centre_eV=float(centre),
                below_eV=float(centre - low) + margin,
                above_eV=float(high - centre) + margin,
                spacing_eV=spacing,
            )
        )
    summary["dropped_reflections"] = sorted(dropped)
    summary["dropped_weight_fraction"] = dropped_weight / total
    return seeds, summary


def absorption_edge_seeds(
    elements: Iterable[str],
    start_eV: float,
    stop_eV: float,
) -> tuple[list[FeatureSeed], dict[str, Any]]:
    """Anchored windows on the absorption jumps the attenuation model contains.

    The line kernels attenuate with ``mu`` from the Chantler ``f2`` table
    (``materials.crystal.absorption_length_ang``), so the jump is located in
    that table: the steepest adjacent ``f2`` ratio within
    :data:`EDGE_SEARCH_FRACTION` of each xraydb edge energy. Both nodes of that
    bracket become anchors, so the discontinuity is bracketed by exact line
    coordinates rather than falling inside one interval, and the window keeps
    :data:`EDGE_NATIVE_NODES` native nodes on each side at the median native
    spacing there -- the resolution at which the model itself represents the
    edge. Edges are mandatory seeds: at 30 keV, 95% of hopg coherent-line
    intensity was measured in the one 3 eV bin at the carbon K edge, where
    ``mu`` jumps 18-fold.

    Shells whose steepest ratio is below :data:`EDGE_MIN_F2_RATIO` are reported
    as ``skipped``. Chantler smears a jump over several brackets, so shells are
    visited strongest xraydb jump first and each takes the steepest bracket
    outside the native-node windows already seeded. A secondary jump within
    :data:`EDGE_SEARCH_FRACTION` of a stronger one (Se L2 beside L3) then gets
    its own anchors; a shell with no such bracket at or above
    :data:`EDGE_MIN_F2_RATIO` lies inside a window already seeded.

    Validation: line-window-seeding
    """
    import xraydb

    from ...materials.atomic import load_henke

    seeds: list[FeatureSeed] = []
    skipped: list[str] = []
    start, stop = float(start_eV), float(stop_eV)
    for element in sorted(set(elements)):
        native, _f1, f2 = load_henke(element)
        claimed: set[int] = set()  # bracket indices inside an already-seeded window
        edges = sorted(
            (
                (shell, edge)
                for shell, edge in xraydb.xray_edges(element).items()
                if edge.jump_ratio is not None and float(edge.jump_ratio) > 1.0
            ),
            key=lambda item: (-float(item[1].jump_ratio), item[1].energy, item[0]),
        )
        for shell, edge in edges:
            nominal = float(edge.energy)
            lower, upper = (
                nominal * (1.0 - EDGE_SEARCH_FRACTION),
                nominal * (1.0 + EDGE_SEARCH_FRACTION),
            )
            if upper < start or lower > stop:
                continue
            label = f"{element} {shell}"
            near = np.flatnonzero((native[:-1] >= lower) & (native[1:] <= upper))
            if near.size == 0:
                skipped.append(label)
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = f2[near + 1] / f2[near]
            ratio = np.where(np.isfinite(ratio), ratio, 0.0)
            if ratio.max() < EDGE_MIN_F2_RATIO:
                skipped.append(label)
                continue
            unclaimed = ~np.isin(near, list(claimed))
            if not (unclaimed & (ratio >= EDGE_MIN_F2_RATIO)).any():
                continue  # its jump lies inside a stronger shell's window
            index = int(near[np.argmax(np.where(unclaimed, ratio, 0.0))])
            first = max(index - EDGE_NATIVE_NODES, 0)
            last = min(index + 1 + EDGE_NATIVE_NODES, native.size - 1)
            claimed.update(range(first, last))
            spacing = float(np.median(np.diff(native[first : last + 1])))
            below_node, above_node = float(native[index]), float(native[index + 1])
            seeds.append(
                FeatureSeed(
                    source=EDGE_SOURCE,
                    label=label,
                    centre_eV=below_node,
                    below_eV=below_node - float(native[first]),
                    above_eV=float(native[last]) - below_node,
                    spacing_eV=spacing,
                    anchor=True,
                )
            )
            seeds.append(
                FeatureSeed(
                    source=EDGE_SOURCE,
                    label=f"{label} above",
                    centre_eV=above_node,
                    below_eV=0.0,
                    above_eV=0.0,
                    spacing_eV=spacing,
                    anchor=True,
                )
            )
    return seeds, {"skipped": skipped}


def characteristic_line_seeds(
    compositions: Iterable[Iterable[tuple[str, float]]],
    *,
    samples_per_feature: int = DEFAULT_SAMPLES_PER_FEATURE,
    relaxation_cutoff_eV: float | None = None,
) -> tuple[list[FeatureSeed], dict[str, Any]]:
    """Windows on every characteristic line the emission model deposits.

    Lines, energies, and natural FWHM come from
    :func:`~pyrite.montecarlo.spectrum.characteristic.load_characteristic_cross_sections`,
    with the same relaxation cutoff ``mc_characteristic_spectrum`` applies per
    composition (the largest recommended cutoff of its elements unless one is
    given) and only lines with a nonzero yield per vacancy. Each window is
    :data:`CHARACTERISTIC_EXTENT_FWHM` FWHM on each side at
    ``FWHM / samples_per_feature``.

    Validation: line-window-seeding
    """
    from .characteristic import load_characteristic_cross_sections

    if int(samples_per_feature) < 1:
        raise ValueError("samples_per_feature must be at least 1")
    seeds: list[FeatureSeed] = []
    for composition in compositions:
        elements = sorted({str(element) for element, _density in composition})
        tables = {element: load_characteristic_cross_sections(element) for element in elements}
        cutoff = (
            max(table.recommended_cutoff_eV for table in tables.values())
            if relaxation_cutoff_eV is None
            else float(relaxation_cutoff_eV)
        )
        for element, table in tables.items():
            emitted = np.asarray(table.line_yield_per_vacancy, dtype=float).any(axis=0)
            for label, centre, fwhm, active in zip(
                table.line_labels, table.line_energy_eV, table.line_fwhm_eV, emitted, strict=True
            ):
                if not active or float(centre) <= cutoff or not float(fwhm) > 0.0:
                    continue
                extent = CHARACTERISTIC_EXTENT_FWHM * float(fwhm)
                seeds.append(
                    FeatureSeed(
                        source=CHARACTERISTIC_SOURCE,
                        label=f"{element} {label}",
                        centre_eV=float(centre),
                        below_eV=extent,
                        above_eV=extent,
                        spacing_eV=float(fwhm) / int(samples_per_feature),
                    )
                )
    return seeds, {}


@dataclass(frozen=True, slots=True)
class SeedContext:
    """Everything a provider may read for one case, after transport."""

    case: Mapping[str, Any]
    segments: Mapping[str, Any]
    n_hat: np.ndarray
    electron_limit: int | None
    start_eV: float
    stop_eV: float
    feature_width_eV: float
    samples_per_feature: int = DEFAULT_SAMPLES_PER_FEATURE
    aliased_weight_limit: float = 1.0e-3
    tail_widths: float = DEFAULT_TAIL_WIDTHS


SeedProvider = Callable[[SeedContext], tuple[Sequence[FeatureSeed], Mapping[str, Any]]]


def _kinematic_provider(context: SeedContext):
    from .lines import _segments_in_layer

    case = context.case

    def _seeds(segments, radiator, label_prefix=""):
        # Orientation keys fall back to the case, as _lines_for_segments does.
        return kinematic_line_seeds(
            segments,
            context.n_hat,
            crystal=radiator["crystal"],
            hkl_list=radiator["hkl_list"],
            feature_width_eV=context.feature_width_eV,
            samples_per_feature=context.samples_per_feature,
            aliased_weight_limit=context.aliased_weight_limit,
            tail_widths=context.tail_widths,
            beam_uvw=radiator.get("beam_uvw"),
            surface_hkl=radiator.get("surface_hkl"),
            azimuth_rad=radiator.get("azimuth_rad", case.get("azimuth_rad", 0.0)),
            recip_miscut_rad=radiator.get("recip_miscut_rad", case.get("recip_miscut_rad")),
            mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),
            mosaic_nodes=case.get("mosaic_mc_nodes", 1),
            electron_limit=context.electron_limit,
            label_prefix=label_prefix,
        )

    radiators = case.get("layer_radiators")
    if radiators is None:
        return _seeds(context.segments, case)
    seeds: list[FeatureSeed] = []
    summary: dict[str, Any] = {}
    for layer, radiator in enumerate(radiators):
        if radiator is None:
            continue
        layer_segments = _segments_in_layer(context.segments, layer)
        if layer_segments["L_ang"].size == 0:
            continue
        layer_seeds, layer_summary = _seeds(layer_segments, radiator, f"layer {layer} ")
        seeds.extend(layer_seeds)
        summary[f"layer {layer}"] = layer_summary
    return seeds, summary


def _case_compositions(case: Mapping[str, Any]) -> list[list[tuple[str, float]]]:
    layers = case.get("abs_layers")
    if layers:
        return [list(layer[2]) for layer in layers]
    return [list(case["composition"])]


def _edge_provider(context: SeedContext):
    case = context.case
    elements = {element for composition in _case_compositions(case) for element, _ in composition}
    crystal = CRYSTALS.get(case.get("crystal"))
    if crystal is not None:
        elements.update(element for element, _ in crystal["basis"])
    return absorption_edge_seeds(elements, context.start_eV, context.stop_eV)


def _characteristic_provider(context: SeedContext):
    return characteristic_line_seeds(
        _case_compositions(context.case), samples_per_feature=context.samples_per_feature
    )


_PROVIDERS: dict[str, SeedProvider] = {
    KINEMATIC_SOURCE: _kinematic_provider,
    EDGE_SOURCE: _edge_provider,
    CHARACTERISTIC_SOURCE: _characteristic_provider,
}

#: Providers an automatic window policy enables when none are named.
DEFAULT_SEED_PROVIDERS = (KINEMATIC_SOURCE, EDGE_SOURCE, CHARACTERISTIC_SOURCE)


def register_seed_provider(name: str, provider: SeedProvider, *, replace: bool = False) -> None:
    """Make ``provider`` selectable by ``name`` in a window policy.

    A provider returns ``(seeds, summary)``; every seed's ``source`` must equal
    ``name`` so provenance can attribute each window. Registering does not
    enable a provider -- the policy's provider list does, and that list is part
    of case identity.
    """
    if not name or not isinstance(name, str):
        raise ValueError("seed provider name must be a non-empty string")
    if name in _PROVIDERS and not replace:
        raise ValueError(f"seed provider {name!r} is already registered")
    _PROVIDERS[name] = provider


def seed_provider_names() -> tuple[str, ...]:
    """Registered provider names, sorted."""
    return tuple(sorted(_PROVIDERS))


def collect_feature_seeds(
    context: SeedContext, names: Iterable[str]
) -> tuple[list[FeatureSeed], dict[str, Any]]:
    """Run the named providers in order; refuse unknown names and misattributed seeds."""
    seeds: list[FeatureSeed] = []
    summaries: dict[str, Any] = {}
    for name in names:
        provider = _PROVIDERS.get(name)
        if provider is None:
            raise ValueError(
                f"line window policy names unknown seed provider {name!r}; "
                f"registered providers are {list(seed_provider_names())}"
            )
        provided, summary = provider(context)
        provided = list(provided)
        for seed in provided:
            if not isinstance(seed, FeatureSeed) or seed.source != name:
                raise ValueError(
                    f"seed provider {name!r} returned {seed!r}; every seed must be a "
                    f"FeatureSeed whose source is {name!r}"
                )
        seeds.extend(provided)
        summaries[name] = {**dict(summary), "seeds": len(provided)}
    return seeds, summaries
