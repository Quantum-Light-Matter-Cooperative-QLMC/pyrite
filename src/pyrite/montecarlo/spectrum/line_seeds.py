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
    "ResonancePopulation",
    "SEEDING_REVISION",
    "EdgeBracket",
    "SeedContext",
    "absorption_edge_brackets",
    "absorption_edge_seeds",
    "LOCAL_SPACING_SOURCE",
    "case_line_stop_eV",
    "case_resonance_populations",
    "characteristic_line_seeds",
    "characteristic_stop_eV",
    "collect_feature_seeds",
    "kinematic_line_seeds",
    "local_spacing_seeds",
    "register_seed_provider",
    "resonance_population_stop_eV",
    "resonance_populations",
    "seed_provider_names",
    "sincsq_upper_tail_bound",
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

#: Bumped whenever the same inputs would seed different windows; it keys the
#: resolved-grid speed cache so a plan from older seeding is not reused.
#: 2: in-medium kinematic root; secondary absorption edges.
#: 3: measured bandwidth and local spacing use production line weights.
SEEDING_REVISION = 3

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


@dataclass(frozen=True, slots=True)
class ResonancePopulation:
    """Radiating lines: resonance, line coefficient, and first-zero width.

    ``width_eV`` is the sinc first-zero width ``pi / a_w`` of each line,
    ``2 pi hbar c / (denominator t_L)``, with the same in-medium denominator
    as the resonance.
    """

    label: str
    energy_eV: np.ndarray
    weight: np.ndarray
    width_eV: np.ndarray


_POPULATION_BLOCK_SIZE = 1 << 18


def _population_blocks(populations: Sequence[ResonancePopulation]):
    """Yield aligned line-array views in bounded blocks."""
    for population in populations:
        size = population.energy_eV.size
        for start in range(0, size, _POPULATION_BLOCK_SIZE):
            stop = min(start + _POPULATION_BLOCK_SIZE, size)
            yield (
                population.energy_eV[start:stop],
                population.weight[start:stop],
                population.width_eV[start:stop],
            )


def resonance_populations(
    segments,
    n_hat,
    *,
    crystal: str,
    hkl_list: Iterable[Sequence[int]],
    beam_uvw=None,
    surface_hkl=None,
    azimuth_rad: float = 0.0,
    recip_miscut_rad=None,
    mosaic_fwhm_rad=None,
    mosaic_nodes: int = 1,
    electron_limit: int | None = None,
    label_prefix: str = "",
    composition: Iterable[tuple[str, float]] | None = None,
    band_eV: tuple[float, float] | None = None,
) -> list[ResonancePopulation]:
    """Per-reflection resonance populations of the line segments.

    The resonance, root solver, ``g`` construction, and ``t_L**2`` weight are
    those documented on :func:`kinematic_line_seeds`, which consumes this;
    mosaic orientations are pooled into their reflection with the quadrature
    weight folded into ``weight``.

    Validation: line-window-seeding
    """
    energy_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    energy = _host(segments[energy_field]).astype(float, copy=False)
    length = _host(segments["L_ang"]).astype(float, copy=False)
    direction = _host(segments["v_hat"]).astype(float, copy=False).reshape(-1, 3)
    if electron_limit is not None:
        line = _host(segments["elec_id"]) < int(electron_limit)
        energy, length, direction = energy[line], length[line], direction[line]
    beta = beta_from_keV(energy)
    velocity = beta[:, None] * direction
    v_dot_n = velocity @ np.asarray(n_hat, dtype=float)
    weight = (length / beta) ** 2
    usable = np.isfinite(weight) & (weight > 0.0) & np.isfinite(v_dot_n) & (v_dot_n < 1.0)
    velocity, v_dot_n, weight = velocity[usable], v_dot_n[usable], weight[usable]
    flight_time = np.sqrt(weight)
    refractive = None
    if band_eV is not None:
        from ...materials.crystal import refractive_index
        from .lines._kernels import _line_tabulation_grid

        # The kernels' table for an axis spanning band_eV (lines/_setup.py).
        start, stop = float(band_eV[0]), float(band_eV[1])
        pad = 0.2 * (stop - start)
        table_energy = _line_tabulation_grid(
            CRYSTALS[crystal], list(composition or ()), max(start - pad, 1.0), stop + pad
        )
        refractive = (
            np.asarray(refractive_index(crystal, table_energy).real, dtype=float),
            table_energy,
        )
    if refractive is not None:
        from .lines._kernels import _in_medium_kinematics

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
        energies, weights, widths = [], [], []
        for mosaic_rotation, mosaic_weight in orientations:
            g_row = g_vector if mosaic_rotation is None else mosaic_rotation @ g_vector
            v_dot_g = velocity @ g_row
            with np.errstate(divide="ignore", invalid="ignore"):
                if refractive is None:
                    denominator = 1.0 - v_dot_n
                else:
                    denominator, _ = _in_medium_kinematics(v_dot_n, v_dot_g, *refractive)
                resonance = HBARC_EV_ANG * v_dot_g / denominator
            radiating = np.isfinite(resonance) & (resonance > _MIN_RESONANCE_EV)
            energies.append(resonance[radiating])
            weights.append(weight[radiating] * float(mosaic_weight))
            widths.append(
                2.0 * np.pi * HBARC_EV_ANG / (denominator[radiating] * flight_time[radiating])
            )
        label = label_prefix + "(" + " ".join(str(int(index)) for index in hkl) + ")"
        populations.append(
            ResonancePopulation(
                label, np.concatenate(energies), np.concatenate(weights), np.concatenate(widths)
            )
        )
    return populations


def sincsq_upper_tail_bound(width_eV, distance_eV):
    """Upper bound on a ``sinc**2`` line's mass fraction beyond ``distance_eV``.

    With ``a_w = pi / width`` the line is ``sinc**2(a_w (E - E_res) / pi)`` and
    its whole integral is ``pi / a_w = width``. On one side,
    ``int_D^inf sin**2(a_w x) / (a_w x)**2 dx <= int_D^inf dx / (a_w x)**2
    = 1 / (a_w**2 D)``, so the fraction is at most
    ``width / (pi**2 D)``; it is capped at one, and a line whose resonance is on
    the wrong side (``D <= 0``) counts wholly.

    Validation: line-grid-resonance-bandwidth
    """
    width = np.asarray(width_eV, dtype=float)
    distance = np.asarray(distance_eV, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = np.where(distance > 0.0, width / (np.pi**2 * distance), 1.0)
    return np.minimum(bound, 1.0)


def resonance_population_stop_eV(
    populations: Sequence[ResonancePopulation],
    *,
    ceiling_eV: float,
    truncation_limit: float,
    round_to_eV: float = 100.0,
) -> tuple[float | None, dict[str, Any]]:
    """Line-axis ``stop`` from a case's measured resonance population.

    Source equation: each segment ``i`` of each reflection radiates the
    finite-time line ``W_i sinc**2(a_i (E - E_i) / pi)`` (ledger rows
    ``finite-time-lineshape``, ``line-energy-dispersion``), whose weight
    ``W_i`` does not depend on ``E``. Its mass above ``stop`` is at most
    ``W_i width_i * width_i / (pi**2 (stop - E_i))``
    (:func:`sincsq_upper_tail_bound`). ``stop`` is the smallest value, rounded up
    to ``round_to_eV``, whose summed tail bound is at most ``truncation_limit``
    of the summed line mass. The caller supplies ``W_i`` in ``weight``.

    Assumptions: each line's coefficient is constant in energy, as in the
    production kernel. The spectrum phase measures the truncated fraction again
    and the runner refuses a case above its share. ``stop`` never exceeds
    ``ceiling_eV``, the closed-form
    bound of ``line-grid-kinematic-bandwidth``; a population that cannot meet
    the limit below it returns the ceiling.

    Lines resonating above ``stop`` count wholly, so a population may leave its
    lightest, farthest lines outside the axis within ``truncation_limit``.

    Limiting case: a single straight flight gives
    ``stop = E_res + width / (pi**2 truncation_limit)``, before rounding.

    Returns ``(stop, summary)``; ``stop`` is ``None`` when nothing radiates.

    Validation: line-grid-resonance-bandwidth
    """
    limit = float(truncation_limit)
    if not 0.0 < limit < 1.0:
        raise ValueError("truncation_limit must lie in (0, 1)")
    ceiling = float(ceiling_eV)
    total = 0.0
    n_lines = 0
    min_energy = np.inf
    max_energy = -np.inf
    for energy_block, weight_block, width_block in _population_blocks(populations):
        mass = weight_block * width_block
        keep = np.isfinite(mass) & (mass > 0.0)
        if not keep.any():
            continue
        energies = energy_block[keep]
        total += float(mass[keep].sum())
        n_lines += int(keep.sum())
        min_energy = min(min_energy, float(energies.min()))
        max_energy = max(max_energy, float(energies.max()))
    summary: dict[str, Any] = {
        "truncation_limit": limit,
        "ceiling_eV": ceiling,
        "n_lines": n_lines,
    }
    if n_lines == 0:
        summary.update(max_resonance_eV=None, proxy_truncated_fraction=0.0, capped=False)
        return None, {**summary, "stop_eV": None}

    def lost(stop):
        numerator = 0.0
        for energy_block, weight_block, width_block in _population_blocks(populations):
            mass = weight_block * width_block
            keep = np.isfinite(mass) & (mass > 0.0)
            if keep.any():
                numerator += float(
                    (
                        mass[keep]
                        * sincsq_upper_tail_bound(width_block[keep], stop - energy_block[keep])
                    ).sum()
                )
        return numerator / total

    # Lines above a trial edge count wholly, so the search may start below the
    # whole population: a rare hard-scattered line spends the share instead of
    # forcing the axis out to its resonance.
    low = min_energy
    stop = ceiling
    if low < ceiling and lost(ceiling) <= limit:
        high = ceiling
        # The tail sum falls monotonically in stop; 60 halvings reach float64
        # resolution of any band this axis can carry.
        for _ in range(60):
            middle = 0.5 * (low + high)
            if lost(middle) > limit:
                low = middle
            else:
                high = middle
        stop = min(ceiling, float(np.ceil(high / round_to_eV) * round_to_eV))
    summary.update(
        max_resonance_eV=max_energy,
        proxy_truncated_fraction=lost(stop),
        capped=stop >= ceiling,
        stop_eV=stop,
    )
    return stop, summary


def characteristic_stop_eV(
    compositions: Iterable[Iterable[tuple[str, float]]],
    *,
    truncation_limit: float,
    relaxation_cutoff_eV: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Lowest line-axis ``stop`` that keeps every characteristic line's upper tail.

    Source equation: a characteristic line is a Lorentzian of natural FWHM
    ``Gamma`` (ledger row ``characteristic-radiation``). Its mass fraction more
    than ``D`` above the centre is ``1/2 - arctan(2 D / Gamma) / pi``, which is
    at most ``Gamma / (2 pi D)``. Every emitted line, selected exactly as
    :func:`characteristic_line_seeds` selects them, therefore loses at most
    ``truncation_limit`` of its own mass above
    ``centre + Gamma / (2 pi truncation_limit)``.

    Assumptions: the bound is per line, so it holds for any mixture of line
    yields; the lower edge is not addressed.

    Limiting case: ``Gamma -> 0`` puts ``stop`` on the highest centre.

    Returns ``(stop, summary)``; ``stop`` is ``None`` when no line is emitted.

    Validation: line-grid-resonance-bandwidth
    """
    from .characteristic import load_characteristic_cross_sections

    limit = float(truncation_limit)
    if not 0.0 < limit < 1.0:
        raise ValueError("truncation_limit must lie in (0, 1)")
    stop, label_at_stop = None, None
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
                edge = float(centre) + float(fwhm) / (2.0 * np.pi * limit)
                if stop is None or edge > stop:
                    stop, label_at_stop = edge, f"{element} {label}"
    return stop, {"stop_eV": stop, "line": label_at_stop, "truncation_limit": limit}


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
    composition: Iterable[tuple[str, float]] | None = None,
    band_eV: tuple[float, float] | None = None,
) -> tuple[list[FeatureSeed], dict[str, Any]]:
    """One window per reflection over its weighted resonance population.

    Source equation: the resonance the line kernels place each segment's line
    at, ``E_res = hbar c (v . g) / (1 - Re n(E_res) v . n_hat)`` with
    ``v = beta v_hat`` (Zhai SI Eq. 10 with the bulk dispersion
    ``k = Re n omega``; ledger rows ``line-energy-dispersion`` and
    ``xray-in-medium-resonance``). Given ``band_eV``, the root is solved by the
    kernels' own ``lines/_kernels.py::_in_medium_kinematics`` on the table they
    build for an axis spanning ``band_eV`` (``_line_tabulation_grid`` over the
    crystal basis and ``composition``, ``refractive_index(crystal, ...)``), so
    pairs the kernels reject as unsettled are left out here too. Without
    ``band_eV`` the vacuum root ``Re n = 1`` is used. ``g`` is built exactly as
    the kernels build their rows: ``R_m R_orient g_hkl`` over the case
    orientation and each mosaic quadrature orientation, weighted by the
    quadrature weight.

    Assumptions: ``t_L**2`` is the intensity proxy, as in
    :func:`~pyrite.montecarlo.spectrum.diagnostics.sinc_feature_spacing`;
    ``|A|**2``, absorption, and the line-electron energy cutoff are ignored, so
    the band is conservative. The vacuum root sits
    ``dE ~ -E delta beta cos(Theta) / (1 - beta cos(Theta))`` from the
    in-medium one, which is ``delta cos(Theta) L / lambda`` of a segment's own
    feature widths and not bounded by ``delta`` -- measured up to 5.4 widths
    above 300 eV (``docs/validation/beam-transport/line-window-seeding.md``) --
    which is why production seeding passes ``band_eV``. Feature widths use the
    same in-medium denominator the kernels' ``a_width`` does.

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

    populations = resonance_populations(
        segments,
        n_hat,
        crystal=crystal,
        hkl_list=hkl_list,
        beam_uvw=beam_uvw,
        surface_hkl=surface_hkl,
        azimuth_rad=azimuth_rad,
        recip_miscut_rad=recip_miscut_rad,
        mosaic_fwhm_rad=mosaic_fwhm_rad,
        mosaic_nodes=mosaic_nodes,
        electron_limit=electron_limit,
        label_prefix=label_prefix,
        composition=composition,
        band_eV=band_eV,
    )
    # The spacing above resolves the eps-quantile feature, not the narrowest one.
    # Measured over hopg/wse2 at 30-100 keV the two coincide to within 11% -- the
    # longest single flight floors t_L, so the width distribution has no narrow
    # tail -- but that is a property of these cases, not a theorem. Report the
    # narrowest radiating feature so a case where the quantile drifts above it is
    # visible instead of silently under-resolved.
    narrowest = min(
        (float(entry.width_eV.min()) for entry in populations if entry.width_eV.size),
        default=float("inf"),
    )
    populations = [
        (entry.label, entry.energy_eV, entry.weight, float(entry.weight.sum()))
        for entry in populations
    ]

    if not np.isfinite(narrowest):
        narrowest = float("nan")
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


@dataclass(frozen=True, slots=True)
class EdgeBracket:
    """One absorption jump, located in the table the attenuation model reads.

    ``below_eV``/``above_eV`` are the two adjacent *native* Chantler nodes the
    steepest ``f2`` step lies between, so the discontinuity is bounded by exact
    tabulated coordinates. ``first_eV``/``last_eV`` bound the kept native
    neighbourhood and ``spacing_eV`` is its median native spacing -- the
    resolution at which the model itself represents the edge, and therefore the
    finest placement any consumer can justify here.
    """

    label: str
    below_eV: float
    above_eV: float
    first_eV: float
    last_eV: float
    spacing_eV: float


def absorption_edge_brackets(
    elements: Iterable[str],
    start_eV: float,
    stop_eV: float,
) -> tuple[list[EdgeBracket], dict[str, Any]]:
    """Locate the absorption jumps the attenuation model actually contains.

    The emission and escape models attenuate with ``mu`` from the Chantler
    ``f2`` table (``materials.crystal.absorption_length_ang``), so a jump is
    located in *that* table rather than taken from a nominal edge energy: the
    steepest adjacent ``f2`` ratio within :data:`EDGE_SEARCH_FRACTION` of each
    xraydb edge energy. The two tables disagree by up to ~2.1% below 1 keV, so
    the xraydb energy is only a locator.

    Shells whose steepest ratio is below :data:`EDGE_MIN_F2_RATIO` are reported
    as ``skipped``. Chantler smears a jump over several brackets, so shells are
    visited strongest xraydb jump first and each takes the steepest bracket
    outside the native-node neighbourhoods already claimed. A secondary jump
    within :data:`EDGE_SEARCH_FRACTION` of a stronger one (Se L2 beside L3) then
    gets its own bracket; a shell with no such bracket at or above
    :data:`EDGE_MIN_F2_RATIO` lies inside one already claimed.

    An edge whose search interval does not overlap ``[start_eV, stop_eV]`` is
    dropped silently: it is outside the modelled band and places no requirement
    on a grid over that band.

    Shared by the line-axis window seeds (issue #101) and the photon-continuum
    node refinement (:mod:`pyrite.energy_grid.refine`, issue #100) so both read
    one locator rather than two copies of an edge list.
    """
    import xraydb

    from ...materials.atomic import load_henke

    brackets: list[EdgeBracket] = []
    skipped: list[str] = []
    start, stop = float(start_eV), float(stop_eV)
    for element in sorted(set(elements)):
        native, _f1, f2 = load_henke(element)
        claimed: set[int] = set()  # bracket indices inside an already-claimed window
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
            brackets.append(
                EdgeBracket(
                    label=label,
                    below_eV=float(native[index]),
                    above_eV=float(native[index + 1]),
                    first_eV=float(native[first]),
                    last_eV=float(native[last]),
                    spacing_eV=float(np.median(np.diff(native[first : last + 1]))),
                )
            )
    return brackets, {"skipped": skipped}


def absorption_edge_seeds(
    elements: Iterable[str],
    start_eV: float,
    stop_eV: float,
) -> tuple[list[FeatureSeed], dict[str, Any]]:
    """Anchored windows on the absorption jumps the attenuation model contains.

    Edge positions come from :func:`absorption_edge_brackets`. Both nodes of the
    located bracket become anchors, so the discontinuity is bracketed by exact
    line coordinates rather than falling inside one interval, and the window
    keeps :data:`EDGE_NATIVE_NODES` native nodes on each side at the median
    native spacing there -- the resolution at which the model itself represents
    the edge. Edges are mandatory seeds: at 30 keV, 95% of hopg coherent-line
    intensity was measured in the one 3 eV bin at the carbon K edge, where
    ``mu`` jumps 18-fold.

    Validation: line-window-seeding
    """
    brackets, summary = absorption_edge_brackets(elements, start_eV, stop_eV)
    seeds: list[FeatureSeed] = []
    for bracket in brackets:
        seeds.append(
            FeatureSeed(
                source=EDGE_SOURCE,
                label=bracket.label,
                centre_eV=bracket.below_eV,
                below_eV=bracket.below_eV - bracket.first_eV,
                above_eV=bracket.last_eV - bracket.below_eV,
                spacing_eV=bracket.spacing_eV,
                anchor=True,
            )
        )
        seeds.append(
            FeatureSeed(
                source=EDGE_SOURCE,
                label=f"{bracket.label} above",
                centre_eV=bracket.above_eV,
                below_eV=0.0,
                above_eV=0.0,
                spacing_eV=bracket.spacing_eV,
                anchor=True,
            )
        )
    return seeds, summary


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

    def _seeds(segments, radiator, composition, label_prefix=""):
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
            composition=composition,
            band_eV=(context.start_eV, context.stop_eV),
        )

    compositions = _case_compositions(case)
    radiators = case.get("layer_radiators")
    if radiators is None:
        return _seeds(context.segments, case, compositions[0])
    seeds: list[FeatureSeed] = []
    summary: dict[str, Any] = {}
    for layer, radiator in enumerate(radiators):
        if radiator is None:
            continue
        layer_segments = _segments_in_layer(context.segments, layer)
        if layer_segments["L_ang"].size == 0:
            continue
        layer_seeds, layer_summary = _seeds(
            layer_segments, radiator, compositions[layer], f"layer {layer} "
        )
        seeds.extend(layer_seeds)
        summary[f"layer {layer}"] = layer_summary
    return seeds, summary


def case_resonance_populations(
    case: Mapping[str, Any],
    segments: Mapping[str, Any],
    n_hat,
    *,
    electron_limit: int | None,
    band_eV: tuple[float, float],
) -> list[ResonancePopulation]:
    """Every radiating layer's resonance populations, read as the kernels read them.

    Per layer exactly as :func:`_kinematic_provider` reads it, with the in-medium
    root tabulated over ``band_eV``.

    Validation: line-grid-resonance-bandwidth
    """
    from .lines import _segments_in_layer

    compositions = _case_compositions(case)
    radiators = case.get("layer_radiators")
    if radiators is None:
        layers = [(segments, case, compositions[0], "")]
    else:
        layers = []
        for layer, radiator in enumerate(radiators):
            if radiator is None:
                continue
            layer_segments = _segments_in_layer(segments, layer)
            if layer_segments["L_ang"].size:
                layers.append((layer_segments, radiator, compositions[layer], f"layer {layer} "))
    populations: list[ResonancePopulation] = []
    for layer_segments, radiator, composition, prefix in layers:
        # Orientation keys fall back to the case, as _lines_for_segments does.
        populations.extend(
            resonance_populations(
                layer_segments,
                n_hat,
                crystal=radiator["crystal"],
                hkl_list=radiator["hkl_list"],
                beam_uvw=radiator.get("beam_uvw"),
                surface_hkl=radiator.get("surface_hkl"),
                azimuth_rad=radiator.get("azimuth_rad", case.get("azimuth_rad", 0.0)),
                recip_miscut_rad=radiator.get("recip_miscut_rad", case.get("recip_miscut_rad")),
                mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),
                mosaic_nodes=case.get("mosaic_mc_nodes", 1),
                electron_limit=electron_limit,
                label_prefix=prefix,
                composition=composition,
                band_eV=(float(band_eV[0]), float(band_eV[1])),
            )
        )
    return populations


def case_line_stop_eV(
    case: Mapping[str, Any],
    populations: Sequence[ResonancePopulation],
    *,
    start_eV: float,
    ceiling_eV: float,
    truncation_limit: float,
    proxy_safety: float,
    round_to_eV: float = 100.0,
) -> tuple[float, dict[str, Any]]:
    """Measured line-axis ``stop`` for one case: PXR/CBS and characteristic lines.

    ``populations`` contain every production line over
    ``(start_eV, ceiling_eV)``; their edge is chosen at
    ``truncation_limit / proxy_safety`` (:func:`resonance_population_stop_eV`).
    Characteristic lines use their exact per-line bound at ``truncation_limit``
    (:func:`characteristic_stop_eV`). The larger edge wins, rounded up to
    ``round_to_eV``, never above ``ceiling_eV`` and at least one rounding step
    above ``start_eV``.

    Validation: line-grid-resonance-bandwidth
    """
    kinematic_stop, kinematic = resonance_population_stop_eV(
        populations,
        ceiling_eV=ceiling_eV,
        truncation_limit=float(truncation_limit) / float(proxy_safety),
        round_to_eV=round_to_eV,
    )
    characteristic_edge, characteristic = characteristic_stop_eV(
        _case_compositions(case), truncation_limit=truncation_limit
    )
    edges = [value for value in (kinematic_stop, characteristic_edge) if value is not None]
    stop = max([float(start_eV) + round_to_eV, *edges])
    stop = min(float(ceiling_eV), float(np.ceil(stop / round_to_eV) * round_to_eV))
    return stop, {
        "stop_eV": stop,
        "truncation_limit": float(truncation_limit),
        "proxy_safety": float(proxy_safety),
        "kinematic": kinematic,
        "characteristic": characteristic,
    }


LOCAL_SPACING_SOURCE = "pxr-local"


def local_spacing_seeds(
    populations: Sequence[ResonancePopulation],
    *,
    start_eV: float,
    stop_eV: float,
    floor_spacing_eV: float,
    max_spacing_eV: float,
    halo_limit: float,
    bin_eV: float = 100.0,
) -> tuple[list[FeatureSeed], dict[str, Any]]:
    """Energy-dependent spacing: fine only where narrow lines resonate.

    Each line narrower than ``max_spacing_eV`` needs nodes no farther apart
    than its first-zero width ``w`` (``h <= pi / a_w`` integrates its
    ``sinc**2`` exactly on a uniform grid, ledger row ``line-grid-sinc-convergence``),
    but only within a halo ``D = w / (pi**2 halo_limit)``: beyond it the line
    keeps at most ``halo_limit`` of its mass (:func:`sincsq_upper_tail_bound`),
    which is all a coarser sampling there can misplace. Required spacings are
    quantised to ``floor_spacing_eV * 2**k``, with ``floor_spacing_eV`` the
    case's global sinc-Nyquist step, so lines the global rule already lets alias
    (their weight is inside its aliased-weight budget) are held to that floor
    rather than refined further. Bins of ``bin_eV`` take the finest level any
    overlapping halo asks for; runs of equal level become one seed each.

    Assumptions: no weighting beyond the population itself -- every
    sub-backbone line is resolved within its halo, so this is at least as fine
    as the uniform step around every line that step resolves. Nonuniform
    trapezoid error inside a halo is not bounded here; the window ladder
    measures it.

    Limiting case: one line of width ``w`` gives one window of spacing
    ``<= w`` spanning ``E_res +- D``, rounded out to bins, on the backbone.

    Returns ``(seeds, summary)``.

    Validation: line-grid-resonance-local-spacing
    """
    floor = float(floor_spacing_eV)
    backbone = float(max_spacing_eV)
    if not 0.0 < floor:
        raise ValueError("floor_spacing_eV must be positive")
    if not 0.0 < float(halo_limit) < 1.0:
        raise ValueError("halo_limit must lie in (0, 1)")
    start, stop = float(start_eV), float(stop_eV)
    levels = [floor * 2.0**k for k in range(64) if floor * 2.0**k < backbone]
    summary: dict[str, Any] = {
        "floor_spacing_eV": floor,
        "max_spacing_eV": backbone,
        "halo_limit": float(halo_limit),
        "bin_eV": float(bin_eV),
        "narrow_lines": 0,
    }
    if not levels:
        return [], {**summary, "windows": 0}
    n_bins = int(np.ceil((stop - start) / float(bin_eV)))
    required = np.full(n_bins, len(levels), dtype=np.int64)
    coverage = np.zeros((len(levels), n_bins + 1), dtype=np.int64)
    for energy_block, _, width_block in _population_blocks(populations):
        narrow = np.isfinite(energy_block) & np.isfinite(width_block) & (width_block < backbone)
        if not narrow.any():
            continue
        energy = energy_block[narrow]
        width = width_block[narrow]
        summary["narrow_lines"] += int(width.size)
        level = np.clip(np.floor(np.log2(np.maximum(width, floor) / floor)), 0, len(levels) - 1)
        halo = width / (np.pi**2 * float(halo_limit))
        first = np.clip(np.floor((energy - halo - start) / bin_eV), 0, n_bins).astype(np.int64)
        last = np.clip(np.ceil((energy + halo - start) / bin_eV), 0, n_bins).astype(np.int64)
        inside = last > first
        first, last, level = first[inside], last[inside], level[inside].astype(np.int64)
        for k in range(len(levels)):
            chosen = level == k
            if not chosen.any():
                continue
            np.add.at(coverage[k], first[chosen], 1)
            np.add.at(coverage[k], last[chosen], -1)
    if summary["narrow_lines"] == 0:
        return [], {**summary, "windows": 0}
    for k in range(len(levels)):
        covered = np.cumsum(coverage[k, :-1]) > 0
        required[covered & (required > k)] = k
    seeds = []
    index = 0
    while index < n_bins:
        k = int(required[index])
        run = index
        while run < n_bins and required[run] == k:
            run += 1
        if k < len(levels):
            lo = start + index * bin_eV
            hi = min(stop, start + run * bin_eV)
            seeds.append(
                FeatureSeed(
                    source=LOCAL_SPACING_SOURCE,
                    label=f"{lo:.0f}-{hi:.0f} eV",
                    centre_eV=lo,
                    below_eV=0.0,
                    above_eV=hi - lo,
                    spacing_eV=levels[k],
                )
            )
        index = run
    return seeds, {**summary, "windows": len(seeds), "levels_eV": levels}


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
