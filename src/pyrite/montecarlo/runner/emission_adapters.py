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


def _cxr(batch: EmissionBatch) -> EmissionOutput:
    from .. import runner

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
    return EmissionSelection(
        catalog.get("emission.cxr/cxr"),
        catalog.get("emission.characteristic/characteristic"),
        select_continuum(case, catalog),
    )
