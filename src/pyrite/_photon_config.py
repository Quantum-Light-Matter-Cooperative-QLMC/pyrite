"""Immutable opt-in selection for CPU/CUDA ordinary-hit photon transport."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

CPU_MODEL = "ordinary-hit-photon-cpu.v1"
CUDA_MODEL = "ordinary-hit-photon-cuda.v1"


@dataclass(frozen=True, slots=True)
class PhotonTransport:
    """Select one experimental aperture observable through :class:`Physics`.

    ``edges_eV`` are histogram edges, not directional-spectrum nodes. ``source``
    selects weighted continuous bremsstrahlung or one physical event component.
    Bremsstrahlung requires a positive ``n_samples``; event sources require
    ``n_samples=None``. Counts use the original incident-electron population.
    ``seed=None`` uses the resolved Case seed. Cutoffs/ceiling are in eV.

    CPU/CUDA, ideal unpixelated physical detectors and fixed populations only.
    Charged feedback follows target and named catalog filter materials across
    vacuum gaps and re-entry; source primary transport stays target-only.
    The bounded reference models do not certify full-track MeV validation or
    pixel response.
    """

    edges_eV: tuple[float, ...]
    source: Literal["brem", "characteristic", "annihilation"] = "brem"
    n_samples: int | None = None
    photon_cutoff_eV: float = 1000.0
    electron_cutoff_eV: float = 1000.0
    source_ceiling_eV: float = 1_000_000.0
    seed: int | None = None
    charged_feedback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges_eV", _histogram_edges(self.edges_eV))
        if self.source not in ("brem", "characteristic", "annihilation"):
            raise ValueError("source must be 'brem', 'characteristic', or 'annihilation'")
        if self.source == "brem":
            if type(self.n_samples) is not int or self.n_samples <= 0:
                raise ValueError("brem photon transport requires positive integer n_samples")
        elif self.n_samples is not None:
            raise ValueError("physical event sources require n_samples=None")
        for name in ("photon_cutoff_eV", "electron_cutoff_eV", "source_ceiling_eV"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.source == "brem" and self.source_ceiling_eV <= self.photon_cutoff_eV:
            raise ValueError("source_ceiling_eV must exceed photon_cutoff_eV")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**64):
            raise ValueError("seed must be an unsigned 64-bit integer or None")
        if type(self.charged_feedback) is not bool:
            raise TypeError("charged_feedback must be a bool")


def _histogram_edges(values) -> tuple[float, ...]:
    """Normalize a finite increasing histogram, independently of source sampling."""
    edges = tuple(float(value) for value in values)
    if len(edges) < 2 or any(not isfinite(x) or x < 0 for x in edges):
        raise ValueError("edges_eV must contain at least two finite nonnegative edges")
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError("edges_eV must be strictly increasing")
    return edges


def validate_photon_record(record: object) -> None:
    """Validate the data-only Case extension without loading scientific tables."""
    keys = {
        "model",
        "selection",
        "source_identity_digest",
        "source_case_digest",
        "source_table_markers",
        "observation_identity_digest",
        "identity",
        "width_ang",
        "height_ang",
    }
    if not isinstance(record, dict) or record.keys() != keys:
        raise ValueError("photon_transport must be a complete resolved photon record")
    if record["model"] not in (CPU_MODEL, CUDA_MODEL):
        raise ValueError("unsupported photon transport model")
    if not isinstance(record["selection"], dict):
        raise ValueError("photon transport selection must be a record")
    PhotonTransport(**record["selection"])
    for name in ("source_identity_digest", "source_case_digest", "observation_identity_digest"):
        value = record[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"photon transport {name} must be a SHA-256 digest")
    if not isinstance(record["identity"], dict) or not isinstance(
        record["source_table_markers"], dict
    ):
        raise ValueError("photon transport requires identity and source table records")
    width, height = record["width_ang"], record["height_ang"]
    if (width is None) != (height is None):
        raise ValueError("photon target footprint dimensions must both be set or absent")
    if width is not None and any(
        isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (width, height)
    ):
        raise ValueError("photon target footprint dimensions must be finite and positive")
