"""Explicit scalar optical response for bounded source-only TR/CTR diagnostics.

No catalog inference or atomic-table fallback. The passive convention is
exp(-i omega t), so Im(epsilon) >= 0. Validation: optical-material-response
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .crystal import HC_EV_ANG


@dataclass(frozen=True)
class OpticalResponse:
    """Local isotropic relative permittivity, linearly interpolated in eV.

    ``segments`` explicitly separate supported intervals: interpolation across a
    gap is forbidden. Arrays are copied into immutable storage. Temperature is
    kelvin or None when the source does not specify it; sample state must say
    whether the ordinates are measured or fitted. ``medium`` is an explicit
    sample label, never a composition-based inference.

    Validation: optical-material-response
    """

    medium: str
    segments: tuple[tuple[np.ndarray, np.ndarray], ...]
    source: str
    source_sha256: str
    license: str
    sample_state: str
    temperature_K: float | None = None
    convention: str = "exp(-i omega t)"
    symmetry: str = "scalar-isotropic"

    def __post_init__(self) -> None:
        if self.symmetry != "scalar-isotropic":
            raise ValueError("optical response: unsupported tensor/anisotropic response")
        if self.convention != "exp(-i omega t)":
            raise ValueError("optical response requires passive exp(-i omega t) convention")
        if not all((self.medium, self.source, self.license, self.sample_state)):
            raise ValueError("optical response requires medium, source, license and sample state")
        if len(self.source_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_sha256
        ):
            raise ValueError("optical response requires a lowercase source SHA-256")
        if self.temperature_K is not None and (
            not np.isfinite(self.temperature_K) or self.temperature_K <= 0
        ):
            raise ValueError("temperature_K must be positive and finite or unspecified")
        if not self.segments:
            raise ValueError("optical response requires at least one supported segment")
        frozen = []
        previous = 0.0
        for nodes, values in self.segments:
            energy, eps = np.asarray(nodes, dtype=float), np.asarray(values, dtype=complex)
            _validate_segment(energy, eps, previous)
            # bytes-backed arrays cannot be made writable by a consumer.
            frozen.append(
                (np.frombuffer(energy.tobytes(), float), np.frombuffer(eps.tobytes(), complex))
            )
            previous = float(energy[-1])
        object.__setattr__(self, "segments", tuple(frozen))

    @property
    def bands_eV(self) -> tuple[tuple[float, float], ...]:
        """Closed supported intervals in photon energy [eV]."""
        return tuple((float(e[0]), float(e[-1])) for e, _ in self.segments)

    @property
    def digest(self) -> str:
        """Hash source identity, response policy, sample state and numeric content."""
        identity = repr(
            (
                self.medium,
                self.source,
                self.source_sha256,
                self.license,
                self.sample_state,
                self.temperature_K,
                self.convention,
                self.symmetry,
                "linear-epsilon-in-eV-v1",
            )
        )
        digest = sha256(identity.encode())
        for energy, eps in self.segments:
            digest.update(len(energy).to_bytes(8, "little"))
            digest.update(np.asarray(energy, dtype="<f8").tobytes())
            digest.update(np.asarray(eps, dtype="<c16").tobytes())
        return digest.hexdigest()

    def epsilon(self, energy_eV: Any, *, xp: Any = np) -> Any:
        """Interpolate on the host, validate coverage, then upload complex128.

        Rejects nonpositive/nonfinite energies and gaps before device upload.
        Linear interpolation of complex epsilon is a convex combination, hence
        preserves passivity and the vacuum limit epsilon=1. ``xp`` may be CuPy;
        inputs are host coordinates and output retains their shape.

        Source: Rakić et al., Appl. Opt. 37, 5271 (1998), n+ik ordinates;
        epsilon=(n+ik)^2 for local nonmagnetic media. No extrapolation.
        Validation: optical-material-response
        """
        energy = np.asarray(energy_eV, dtype=float)
        if energy.size == 0 or not np.all(np.isfinite(energy)) or np.any(energy <= 0):
            raise ValueError(f"{self.medium}: optical energies must be finite and positive")
        result = np.full(energy.shape, np.nan + 1j * np.nan, dtype=complex)
        for nodes, eps in self.segments:
            mask = (energy >= nodes[0]) & (energy <= nodes[-1])
            selected = energy[mask]
            result[mask] = np.interp(selected, nodes, eps.real) + 1j * np.interp(
                selected, nodes, eps.imag
            )
        if not np.all(np.isfinite(result)):
            raise ValueError(
                f"{self.medium}: optical response unsupported energy or gap; bands {self.bands_eV} eV"
            )
        return xp.asarray(result, dtype=xp.complex128)


def _validate_segment(energy: np.ndarray, eps: np.ndarray, previous: float) -> None:
    if energy.ndim != 1 or energy.size < 2 or eps.shape != energy.shape:
        raise ValueError("optical response needs scalar epsilon on matching 1-D nodes")
    if not np.all(np.isfinite(energy)) or energy[0] <= previous or np.any(np.diff(energy) <= 0):
        raise ValueError("optical response segments must be positive, ordered and disjoint")
    if not np.all(np.isfinite(eps)) or np.any(eps.imag < 0):
        raise ValueError("optical response must be finite and passive: Im(epsilon) >= 0")


_RAKIC_REVISION = "c5c2f188e848453def5970e347399d653df2ffc2"
_RAKIC_SHA256 = "a91fca57ee99efc77354908fa11b9c4822e8852394042069caaac7610c52e602"


def gold_rakic_response() -> OpticalResponse:
    """Pinned CC0 Au Lorentz–Drude fitted response; actual tabulated band only.

    The record spans about 0.2–5 eV, narrower than the paper's overall model
    ranges. The source does not specify a sample temperature. This is a fit
    to measurements, not a measured sample-specific spectrum.

    Source: Rakić et al., Appl. Opt. 37, 5271 (1998); pinned CC0 record.
    Assumes local scalar nonmagnetic response. Vacuum n=1, k=0 gives epsilon=1.
    Validation: optical-material-response
    """
    path = Path(__file__).parents[1] / "data" / "optical" / "Au-Rakic-LD.yml"
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != _RAKIC_SHA256:
        raise ValueError("Au Rakić optical data failed pinned SHA-256 verification")
    record = yaml.safe_load(raw)["DATA"][0]
    if record["type"] != "tabulated nk":
        raise ValueError("Au Rakić optical record must contain scalar tabulated nk")
    rows = np.array(
        [line.split() for line in record["data"].splitlines() if line.strip()], dtype=float
    )
    energy = HC_EV_ANG / (rows[:, 0] * 1e4)
    order = np.argsort(energy)
    eps = (rows[:, 1] + 1j * rows[:, 2]) ** 2
    return OpticalResponse(
        medium="Au",
        segments=((energy[order], eps[order]),),
        source=f"https://github.com/polyanskiy/refractiveindex.info-database/blob/{_RAKIC_REVISION}/database/data/main/Au/nk/Rakic-LD.yml",
        source_sha256=_RAKIC_SHA256,
        license="CC0-1.0",
        sample_state="Rakić 1998 Lorentz–Drude fit to multiple experimental sources; temperature unspecified",
    )
