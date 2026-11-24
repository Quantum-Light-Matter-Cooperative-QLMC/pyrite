"""Private batch producers in existing spectral slots (ADR-0017).

Runtime contexts contain shared arrays and mutable audit/cache accumulators.
They are not serialized manifests. Catalog construction is explicit and immutable;
there is no plugin discovery or mutation of a production registration table.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..._physics_contract import BUILTIN_DESCRIPTORS, ModelDescriptor


@dataclass(frozen=True, slots=True)
class EmissionBatch:
    """One native grid/population over the caller's already staged segments.

    Grids are eV; segment coordinates are angstrom and electron energies keV.
    Scorers retain their existing output units and normalization. Runtime arrays,
    layer data and accumulators are borrowed by reference, without copying.
    """

    segments: Mapping[str, Any]
    energy_eV: Any
    case: Mapping[str, Any]
    n_hat: Any
    abs_layers: Any = None
    groove: Any = None
    electron_limit: int | None = None
    coherent: bool = False
    table_cache: dict | None = None
    truncation_audit: dict | None = None
    temporal: Any = None
    coefficient_capture: Any = None
    chunk: int | None = None
    electron_band_weights: Any = None
    line_screen: Any = None


@dataclass(frozen=True, slots=True)
class EmissionOutput:
    """One existing component, never a mixture of coherent and incoherent CXR."""

    slot: str
    values: Any


Score = Callable[[EmissionBatch], EmissionOutput]


@dataclass(frozen=True, slots=True)
class EmissionAdapter:
    """Descriptor plus batch score; optional native continuum wrapper target.

    ``score`` preserves the existing runner façade seams. ``from_segments`` is
    the continuum wrapper's implementation on the same context/output contract,
    so existing replay/recompute callers need no signature or output migration.
    """

    descriptor: ModelDescriptor
    score: Score
    route: str = "default"
    from_segments: Score | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ModelDescriptor):
            raise TypeError("emission descriptor must be ModelDescriptor")
        if self.descriptor.family not in {
            "emission.cxr",
            "emission.characteristic",
            "emission.continuum",
        }:
            raise ValueError("adapter requires an existing emission family")
        allowed = {
            "emission.cxr": {"spec", "spec_coherent"},
            "emission.characteristic": {"spec_characteristic"},
            "emission.continuum": {"brem_wide", "brem"},
        }[self.descriptor.family]
        if not self.descriptor.output_slots or not set(self.descriptor.output_slots) <= allowed:
            raise ValueError("adapter requires existing output slots")
        if not callable(self.score) or (
            self.from_segments is not None and not callable(self.from_segments)
        ):
            raise TypeError("batch producers must be callable")
        if self.route not in {"default", "uncoupled", "coupled"}:
            raise ValueError("unknown emission route")

    def evaluate(self, batch: EmissionBatch, slot: str, *, native: bool = False) -> EmissionOutput:
        producer = self.from_segments if native else self.score
        if producer is None:
            raise ValueError("adapter has no native continuum producer")
        if self.descriptor.family == "emission.continuum":
            coupled = batch.case.get("radiative_model") is not None
            if coupled != (self.route == "coupled"):
                raise ValueError("coupled tracks require the coupled continuum adapter")
        output = producer(batch)
        if (
            not isinstance(output, EmissionOutput)
            or output.slot != slot
            or slot not in self.descriptor.output_slots
        ):
            raise ValueError("producer returned an incompatible spectral component")
        expected = batch.energy_eV.shape
        if batch.electron_band_weights is not None:
            expected = (
                int(batch.segments["Ne"] if batch.electron_limit is None else batch.electron_limit),
            )
        if output.values.shape != expected:
            raise ValueError("producer output must match its native grid or electron population")
        # A continuum adapter's batch score re-enters its native producer
        # through ``_brem_wide_from_segments``, which folds copies once,
        # outside its electron blocks (#379); other families fold here.
        if self.from_segments is None and batch.electron_band_weights is None:
            record = batch.segments.get("cone_split")
            if record is not None and record.get("copies") is not None:
                values = output.values + _copy_strata(self, batch, slot, native)
                return EmissionOutput(slot, values)
        return output


@dataclass(frozen=True, slots=True)
class EmissionCatalog:
    """Local immutable registration; duplicate descriptor/route keys fail."""

    adapters: tuple[EmissionAdapter, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.adapters)
        if any(not isinstance(entry, EmissionAdapter) for entry in entries):
            raise TypeError("catalog entries must be EmissionAdapter")
        keys = [(entry.descriptor.id, entry.route) for entry in entries]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate emission adapter ID/route")
        object.__setattr__(
            self,
            "adapters",
            tuple(sorted(entries, key=lambda entry: (entry.descriptor.id, entry.route))),
        )

    def with_adapter(self, adapter: EmissionAdapter) -> "EmissionCatalog":
        return EmissionCatalog((*self.adapters, adapter))

    def get(self, descriptor_id: str, route: str = "default") -> EmissionAdapter:
        for adapter in self.adapters:
            if (adapter.descriptor.id, adapter.route) == (descriptor_id, route):
                return adapter
        raise ValueError(f"unknown emission adapter: {descriptor_id}/{route}")


@dataclass(frozen=True, slots=True)
class EmissionSelection:
    cxr: EmissionAdapter
    characteristic: EmissionAdapter
    continuum: EmissionAdapter

    def preflight(
        self,
        case: Mapping[str, Any],
        *,
        streaming: bool = False,
        adaptive_observables: tuple[str, ...] = (),
    ) -> None:
        """Refuse unsupported reductions before transport or segment uploads.

        An empty streaming declaration means unsupported. Unknown requirements
        fail closed; the supported reduction is additive whole-electron blocks.
        Adaptive CXR producers must populate ``truncation_audit['electron_mass']``;
        continuum producers must accept ``electron_band_weights``.
        """
        if streaming:
            known = {"whole-electron blocks", "incoherent", "no temporal profile"}
            for adapter in (self.cxr, self.characteristic, self.continuum):
                requirements = set(adapter.descriptor.streaming_requirements)
                if "whole-electron blocks" not in requirements or requirements - known:
                    raise ValueError(f"{adapter.descriptor.id} does not support streaming")
                if "incoherent" in requirements and case.get("coherent_emission"):
                    raise ValueError("coherent emission does not support streaming")
                if "no temporal profile" in requirements and case.get("temporal_profile"):
                    raise ValueError("temporal profile does not support streaming")
        for observable in adaptive_observables:
            adapter = {"cxr": self.cxr, "brem": self.continuum}.get(observable)
            if adapter is None or observable not in adapter.descriptor.adaptive_observables:
                raise ValueError(
                    f"selected emission does not support adaptive observable {observable}"
                )


#: Caller audit settings a copy stratum's own audit shares (#379).
_COPY_AUDIT_SETTINGS = ("start_eV", "stop_eV", "band_eV", "ceiling_eV", "extension_stops_eV")
#: Un-normalized audit sums a copy stratum adds at its weight.
_COPY_AUDIT_SUMS = ("line_mass", "mass_above", "mass_below", "band_mass", "mass_above_extension")


def _copy_strata(adapter, batch: EmissionBatch, slot: str, native: bool) -> Any:
    """Detector-cone copies' spectrum, per power-of-two weight stratum (#379).

    A caller's truncation audit gets each stratum's line-mass sums at weight
    ``W``, and its per-electron masses fold onto the parents
    (``m_i + sum_c W m_c``), so the line-yield statistics read weighted
    per-history samples. A line-collecting audit (grid resolution) is not
    extended here: grid resolution collects copy lines itself
    (:func:`.cone_split_strata.collect_copy_lines`). Per-electron (monitor) batches
    never reach here: they fold copies per history instead.

    Validation: detector-cone-variance-reduction
    """
    from dataclasses import replace

    from .. import runner
    from .cone_split_strata import fold_copies

    record = batch.segments["cone_split"]
    n_norm = int(batch.segments["Ne"] if batch.electron_limit is None else batch.electron_limit)
    caller = batch.truncation_audit
    if caller is not None and "collect" in caller:
        caller = None

    def score(segments, n_copies):
        audit = None
        if caller is not None:
            audit = {k: caller[k] for k in _COPY_AUDIT_SETTINGS if k in caller}
            if "n_electrons" in caller:
                audit["n_electrons"] = n_copies
        part = replace(
            batch,
            segments=runner._segments_on_device(segments),
            electron_limit=n_copies,
            truncation_audit=audit,
            line_screen=None,
        )
        values = adapter.evaluate(part, slot, native=native).values
        if caller is not None and audit is not None:
            _fold_copy_audit(caller, audit, record, segments)
        return values

    return fold_copies(batch.segments, n_norm, score)


def _fold_copy_audit(caller: dict, audit: dict, record, segments) -> None:
    """Add one stratum's audit to the caller's at its weight, masses onto parents."""
    import numpy as np

    from ..._backend import _to_cpu
    from .electron_blocks import _electron_ids

    ids = np.unique(np.asarray(_electron_ids(segments), dtype=np.int64))
    if ids.size == 0:
        return
    weight = float(np.asarray(record["weight"])[ids[0]])
    for key in _COPY_AUDIT_SUMS:
        if key in audit:
            caller[key] = caller.get(key, 0.0) + weight * audit[key]
    mass = audit.get("electron_mass")
    if mass is None or "n_electrons" not in caller:
        return
    mass = np.asarray(_to_cpu(mass), dtype=np.float64)
    parent = int(record["electron_start"]) + np.asarray(record["parent"], dtype=np.int64)[ids]
    added = np.bincount(parent, weights=weight * mass[ids], minlength=int(caller["n_electrons"]))
    existing = caller.get("electron_mass")
    if existing is None or np.ndim(existing) == 0:
        caller["electron_mass"] = (0.0 if existing is None else existing) + added
    else:
        xp = np if isinstance(existing, np.ndarray) else _backend_xp()
        caller["electron_mass"] = existing + xp.asarray(added)


def _backend_xp():
    """The configured array backend module."""
    from ..._backend import BACKEND

    return BACKEND.xp


def _cxr(batch: EmissionBatch) -> EmissionOutput:
    from .. import runner

    if batch.line_screen is not None and not batch.coherent:
        return EmissionOutput("spec", batch.line_screen.score(batch))
    kwargs = dict(
        coherent=batch.coherent,
        Ne=batch.electron_limit,
        table_cache=batch.table_cache,
        temporal=batch.temporal,
    )
    # Keep the exact façade call shape, including audit kwargs used only by the
    # appropriate evaluation; the same table cache serves both CXR evaluations.
    if batch.coherent:
        kwargs["coefficient_capture"] = batch.coefficient_capture
    else:
        kwargs["truncation_audit"] = batch.truncation_audit
    values = runner._cxr_for_segments(
        batch.segments,
        batch.energy_eV,
        batch.case,
        batch.n_hat,
        batch.abs_layers,
        batch.groove,
        **kwargs,
    )
    return EmissionOutput("spec_coherent" if batch.coherent else "spec", values)


def _no_cxr(batch: EmissionBatch) -> EmissionOutput:
    """Zero line spectrum of a ``cxr_source="none"`` case (#1327); scores nothing."""
    import numpy as np

    shape = batch.energy_eV.shape
    if batch.electron_band_weights is not None:
        shape = (
            int(batch.segments["Ne"] if batch.electron_limit is None else batch.electron_limit),
        )
    return EmissionOutput("spec_coherent" if batch.coherent else "spec", np.zeros(shape))


def _characteristic(batch: EmissionBatch) -> EmissionOutput:
    from .. import runner

    values = runner._characteristic_from_segments(
        batch.segments,
        batch.energy_eV,
        batch.case,
        batch.n_hat,
        batch.abs_layers,
        groove=batch.groove,
        Ne=batch.electron_limit,
        table_cache=batch.table_cache,
    )
    return EmissionOutput("spec_characteristic", values)


def _continuum(batch: EmissionBatch) -> EmissionOutput:
    from .. import runner

    extra = (
        {}
        if batch.electron_band_weights is None
        else {"electron_band_weights": batch.electron_band_weights}
    )
    values = runner._brem_wide_from_segments(
        batch.segments,
        batch.energy_eV,
        batch.case,
        batch.n_hat,
        batch.abs_layers,
        groove=batch.groove,
        Ne=batch.electron_limit,
        **extra,
    )
    return EmissionOutput("brem_wide", values)


def _uncoupled(batch: EmissionBatch, model_kwargs: dict) -> EmissionOutput:
    from .emission import _uncoupled_brem_from_segments

    values = _uncoupled_brem_from_segments(
        batch.segments,
        batch.energy_eV,
        batch.case,
        batch.n_hat,
        batch.abs_layers,
        groove=batch.groove,
        Ne=batch.electron_limit,
        brem_chunk=batch.chunk,
        model_kwargs=model_kwargs,
        electron_band_weights=batch.electron_band_weights,
    )
    return EmissionOutput("brem_wide", values)


def _eedl(batch: EmissionBatch) -> EmissionOutput:
    return _uncoupled(batch, dict(cross_section_model="eedl"))


def _bremslib(batch: EmissionBatch) -> EmissionOutput:
    from .. import runner

    tables = runner._case_bremslib_tables(batch.case)
    if tables is None:
        raise ValueError("BremsLib adapter requires bremsstrahlung_model='bremslib'")
    return _uncoupled(batch, dict(cross_section_model="bremslib", bremslib_tables=tables))


def _coupled(batch: EmissionBatch) -> EmissionOutput:
    from .emission import _coupled_brem_from_segments

    values = _coupled_brem_from_segments(
        batch.segments,
        batch.energy_eV,
        batch.case,
        batch.n_hat,
        batch.abs_layers,
        Ne=batch.electron_limit,
        brem_chunk=batch.chunk,
        electron_band_weights=batch.electron_band_weights,
    )
    return EmissionOutput("brem_wide", values)


BUILTIN_EMISSION = EmissionCatalog(
    (
        EmissionAdapter(BUILTIN_DESCRIPTORS.get("emission.cxr", "cxr"), _cxr),
        EmissionAdapter(
            BUILTIN_DESCRIPTORS.get("emission.characteristic", "characteristic"), _characteristic
        ),
        EmissionAdapter(
            BUILTIN_DESCRIPTORS.get("emission.characteristic", "bote-salvat"), _characteristic
        ),
        EmissionAdapter(
            BUILTIN_DESCRIPTORS.get("emission.continuum", "eedl"), _continuum, "uncoupled", _eedl
        ),
        EmissionAdapter(
            BUILTIN_DESCRIPTORS.get("emission.continuum", "bremslib"),
            _continuum,
            "uncoupled",
            _bremslib,
        ),
        EmissionAdapter(
            BUILTIN_DESCRIPTORS.get("emission.continuum", "bremslib"),
            _continuum,
            "coupled",
            _coupled,
        ),
    )
)


def select_continuum(
    case: Mapping[str, Any], catalog: EmissionCatalog = BUILTIN_EMISSION
) -> EmissionAdapter:
    descriptor = BUILTIN_DESCRIPTORS.get(
        "emission.continuum", case.get("bremsstrahlung_model", "eedl")
    )
    route = "coupled" if case.get("radiative_model") is not None else "uncoupled"
    if route == "coupled" and descriptor.model != "bremslib":
        raise ValueError("coupled radiative scoring requires bremsstrahlung_model='bremslib'")
    return catalog.get(descriptor.id, route)


def select_emission(
    case: Mapping[str, Any], catalog: EmissionCatalog = BUILTIN_EMISSION
) -> EmissionSelection:
    """Select case emission descriptors, including the shell-ionization model.

    Validation: bote-salvat-ionization-backend
    """
    cxr = catalog.get("emission.cxr/cxr")
    if case.get("cxr_source") == "none":
        cxr = EmissionAdapter(cxr.descriptor, _no_cxr)
    return EmissionSelection(
        cxr,
        catalog.get(
            "emission.characteristic/"
            + ("bote-salvat" if case.get("shell_ionization_model") else "characteristic")
        ),
        select_continuum(case, catalog),
    )
