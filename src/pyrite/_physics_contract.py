"""Private resource-free metadata for in-tree extensions (ADR-0017).

Descriptors document implementation requirements, not validated physical domains.
There is no selection resolution, table loading or backend probing here.
"""

import json
from dataclasses import asdict, dataclass
from typing import Literal


def _strings(values: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("metadata sequences require non-empty strings")
    values = tuple(values)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("metadata sequences require non-empty strings")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Stable family/model identity and implementation requirements."""

    family: str
    model: str
    owner: str
    requirements: tuple[str, ...] = ()
    implementations: tuple[str, ...] = ()
    output_slots: tuple[str, ...] = ()
    required_datasets: tuple[str, ...] = ()
    streaming_requirements: tuple[str, ...] = ()
    adaptive_observables: tuple[str, ...] = ()
    validation_domain: Literal["not-assessed"] = "not-assessed"

    def __post_init__(self) -> None:
        for name in ("family", "model", "owner"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or "/" in value:
                raise ValueError(f"{name} must be a non-empty string without '/'")
        for name in (
            "requirements",
            "implementations",
            "output_slots",
            "required_datasets",
            "streaming_requirements",
            "adaptive_observables",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        if self.validation_domain != "not-assessed":
            raise ValueError("metadata cannot certify a physical validation domain")

    @property
    def id(self) -> str:
        return f"{self.family}/{self.model}"


@dataclass(frozen=True, slots=True)
class DescriptorCatalog:
    """Immutable deterministic catalog rejecting duplicate family/model IDs."""

    descriptors: tuple[ModelDescriptor, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.descriptors)
        if any(not isinstance(entry, ModelDescriptor) for entry in entries):
            raise TypeError("catalog entries must be ModelDescriptor")
        if len({entry.id for entry in entries}) != len(entries):
            raise ValueError("duplicate physics descriptor ID")
        object.__setattr__(self, "descriptors", tuple(sorted(entries, key=lambda entry: entry.id)))

    def get(self, family: str, model: str) -> ModelDescriptor:
        for descriptor in self.descriptors:
            if (descriptor.family, descriptor.model) == (family, model):
                return descriptor
        raise ValueError(f"unknown physics descriptor: {family}/{model}")


_NATIVE = ("cpu.lockstep", "cpu.per-electron", "cuda.exact", "cuda.lut")
_EXACT = ("cpu.lockstep", "cpu.per-electron", "cuda.exact")
BUILTIN_DESCRIPTORS = DescriptorCatalog(
    (
        ModelDescriptor(
            "transport.stopping",
            "sbethe",
            "montecarlo.transport.stopping",
            ("composition-matched stopping tables",),
            _NATIVE,
            required_datasets=("sbethe",),
        ),
        ModelDescriptor(
            "transport.elastic",
            "mott",
            "montecarlo.runner.case_tables",
            ("SRD64 tables for each layer element",),
            _NATIVE,
            required_datasets=("mott-srd64",),
        ),
        ModelDescriptor(
            "transport.elastic",
            "elsepa",
            "montecarlo.runner.case_tables",
            (
                "ELSEPA tables covering each layer element and energy",
                "exact cores even when LUT configuration is enabled",
            ),
            _EXACT,
            required_datasets=("elsepa",),
        ),
        ModelDescriptor(
            "transport.inelastic", "continuous", "montecarlo.transport.api", implementations=_NATIVE
        ),
        ModelDescriptor(
            "transport.inelastic",
            "shell-soft-hard",
            "montecarlo.runner.case_tables",
            (
                "midpoint",
                "shell data for every catalog layer",
                "no grooves",
                "no CUDA LUT",
                "secondary cascade requires a secondary threshold",
            ),
            _EXACT,
            required_datasets=("shell-configuration", "sbethe"),
        ),
        ModelDescriptor("transport.radiative", "uncoupled", "montecarlo.runner.case_tables"),
        ModelDescriptor(
            "transport.radiative",
            "bremslib-soft-hard",
            "montecarlo.runner.case_tables",
            (
                "midpoint",
                "BremsLib",
                "radiative cutoff",
                "no grooves",
                "no CUDA LUT",
                "photon rows completed on host",
            ),
            _EXACT,
            required_datasets=("bremslib",),
        ),
        ModelDescriptor(
            "transport.pair",
            "penelope-2024",
            "montecarlo.transport.pair_production",
            ("coupled hard photons", "shell-soft-hard", "secondary threshold", "EPDL domain"),
            required_datasets=("epdl",),
        ),
        ModelDescriptor(
            "transport.positron",
            "positron",
            "montecarlo.transport.secondaries",
            (
                "pair conversion",
                "secondary cascade",
                "positron stopping and elastic tables",
                "no Mott positron model",
                "host annihilation completion",
            ),
            required_datasets=("sbethe-positron", "elsepa-positron"),
        ),
        ModelDescriptor(
            "transport.atomic-electron",
            "kawrakow",
            "montecarlo.transport.atomic_electrons",
            ("ELSEPA when shell transport is selected",),
        ),
        ModelDescriptor(
            "transport.atomic-electron", "none", "montecarlo.transport.atomic_electrons"
        ),
        ModelDescriptor(
            "emission.continuum",
            "eedl",
            "montecarlo.runner.emission",
            ("uncoupled tracks", "layer compositions and absorption stack"),
            output_slots=("brem_wide", "brem"),
            required_datasets=("eedl",),
            streaming_requirements=("whole-electron blocks",),
            adaptive_observables=("brem",),
        ),
        ModelDescriptor(
            "emission.continuum",
            "bremslib",
            "montecarlo.runner.emission",
            ("BremsLib tables for every layer element", "coupled estimator for coupled tracks"),
            output_slots=("brem_wide", "brem"),
            required_datasets=("bremslib",),
            streaming_requirements=("whole-electron blocks",),
            adaptive_observables=("brem",),
        ),
        ModelDescriptor(
            "emission.cxr",
            "cxr",
            "montecarlo.runner",
            (
                "radiator reflections and orientation",
                "whole population for coherent/temporal scoring",
            ),
            output_slots=("spec", "spec_coherent"),
            streaming_requirements=("whole-electron blocks", "incoherent", "no temporal profile"),
            adaptive_observables=("cxr",),
        ),
        ModelDescriptor(
            "transport.channeling",
            "planar",
            "montecarlo.channeling",
            (
                "channeling thermal data for the crystal",
                "at least one bound planar state at the beam energy",
                "single-layer ungrooved slab without mosaic",
                "midpoint integration, uncoupled radiative scoring, no secondaries",
            ),
            implementations=("cpu.per-electron",),
            required_datasets=("channeling_thermal",),
        ),
        ModelDescriptor(
            "emission.channeling",
            "planar",
            "montecarlo.runner.channeling_emission",
            (
                "planar channeling entrance zone",
                "constant-energy expected-value source",
                "no replay, directions, streaming or adaptive precision",
            ),
            implementations=("cpu",),
            output_slots=("spec_channeling",),
            required_datasets=("channeling_thermal", "shell-configuration"),
        ),
        ModelDescriptor(
            "emission.transition",
            "planar-stack",
            "montecarlo.runner.transition_emission",
            (
                "planar slab or stack, midpoint row events",
                "locally straight crossings, laterally infinite faces",
                "no grooves, channeling, adaptive precision or temporal profiles",
            ),
            implementations=("cpu",),
            output_slots=("spec_transition",),
        ),
        ModelDescriptor(
            "emission.transition",
            "two-beam",
            "montecarlo.runner.dtr_emission",
            (
                "experimental transported-current Maxwell/CBS solve",
                "one perfect crystal layer and one reflection",
                "unverified; CPU scoring; no recoil",
            ),
            implementations=("cpu",),
            output_slots=("spec", "spec_coherent", "spec_transition"),
        ),
        ModelDescriptor(
            "emission.characteristic",
            "characteristic",
            "montecarlo.runner.emission",
            ("layer compositions and absorption stack",),
            output_slots=("spec_characteristic",),
            streaming_requirements=("whole-electron blocks",),
        ),
        ModelDescriptor(
            "emission.characteristic",
            "bote-salvat",
            "montecarlo.runner.emission",
            (
                "layer compositions and absorption stack",
                "Bote-Salvat K/L/M primary vacancies; EEDL for higher shells",
            ),
            output_slots=("spec_characteristic",),
            required_datasets=("bote-salvat",),
            streaming_requirements=("whole-electron blocks",),
        ),
    )
)


@dataclass(frozen=True, slots=True)
class ApplicabilityEvidence:
    """Construction checks, separate from pending table and physical validation."""

    checks: tuple[str, ...]
    pending: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", _strings(self.checks))
        object.__setattr__(self, "pending", _strings(self.pending))


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    """Caller-supplied hardware evidence, outside physical identity."""

    backend: str
    available: bool
    evidence: str

    def __post_init__(self) -> None:
        _strings((self.backend, self.evidence))
        if not isinstance(self.available, bool):
            raise TypeError("available must be a bool")


def _pairs(values: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    if isinstance(values, str):
        raise ValueError("markers require key/value pairs")
    values = tuple(values)
    if any(isinstance(pair, str) for pair in values):
        raise ValueError("markers require key/value pairs")
    entries = tuple(tuple(pair) for pair in values)
    if any(len(pair) != 2 for pair in entries):
        raise ValueError("markers require key/value pairs")
    for pair in entries:
        _strings(pair)
    if len({pair[0] for pair in entries}) != len(entries):
        raise ValueError("duplicate marker key")
    return tuple(sorted(entries))


@dataclass(frozen=True, slots=True)
class ResolvedPhysicsView:
    """Data-only descriptive snapshot; not a new cache identity payload.

    Canonical JSON keeps nested parameters immutable. ``parameters`` decodes a
    detached copy. Table markers describe the caller's per-case inventory.
    """

    descriptor_ids: tuple[str, ...]
    model_markers: tuple[tuple[str, str], ...]
    table_markers: tuple[tuple[str, str], ...]
    parameters_json: str
    applicability: ApplicabilityEvidence
    backend_availability: tuple[BackendAvailability, ...] = ()
    schema: str = "pyrite.resolved-physics.v1"

    def __post_init__(self) -> None:
        if self.schema != "pyrite.resolved-physics.v1":
            raise ValueError("unknown resolved physics schema")
        ids = _strings(self.descriptor_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate resolved descriptor ID")
        object.__setattr__(self, "descriptor_ids", tuple(sorted(ids)))
        for name in ("model_markers", "table_markers"):
            object.__setattr__(self, name, _pairs(getattr(self, name)))
        parameters = json.loads(self.parameters_json)
        if not isinstance(parameters, dict):
            raise ValueError("resolved parameters must be a JSON object")
        object.__setattr__(
            self,
            "parameters_json",
            json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )
        if not isinstance(self.applicability, ApplicabilityEvidence):
            raise TypeError("applicability must be ApplicabilityEvidence")
        availability = tuple(self.backend_availability)
        if any(not isinstance(item, BackendAvailability) for item in availability):
            raise TypeError("backend evidence must be BackendAvailability")
        if len({item.backend for item in availability}) != len(availability):
            raise ValueError("duplicate backend evidence")
        object.__setattr__(
            self, "backend_availability", tuple(sorted(availability, key=lambda item: item.backend))
        )

    @property
    def parameters(self) -> dict:
        return json.loads(self.parameters_json)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, encoded: str) -> "ResolvedPhysicsView":
        payload = json.loads(encoded)
        payload["applicability"] = ApplicabilityEvidence(**payload["applicability"])
        payload["backend_availability"] = tuple(
            BackendAvailability(**item) for item in payload["backend_availability"]
        )
        return cls(**payload)
