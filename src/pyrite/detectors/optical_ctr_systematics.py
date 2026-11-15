"""Transverse-source and detector-response systematics for the Gaussian CTR fit (#617).

Each declared mode ``k`` is one Gaussian nuisance ``a_k ~ N(0, 1)`` that
changes the expected two-port voltages linearly at fixed source scale ``s``,

    V(sigma, a) = s * [K(sigma) + sum_k a_k * D_k(sigma)] + offsets,

with ``K`` the nominal calibrated model and ``D_k`` the 1-sigma voltage change
of mode ``k`` at the trial duration and unit source scale:

``kind="transverse"``
    a fractional 1-sigma change ``u_k(E)`` of the coherent pair envelope
    (transverse form factor, collected acceptance of the pair term), so
    ``D_k = R @ (u_k * Phi_pair,0 * exp[-(E sigma/hbar)^2])`` on both ports;
``kind="response"``
    a fractional 1-sigma change ``u_k(E)`` of the spectral efficiency of one
    port (``port=0`` or ``1``) or of both together (``port=None``), so
    ``D_k[:, p] = R_p @ (u_k * Phi(E; sigma))`` for each affected port.

``R`` is the calibrated response kernel shared with the nominal model. Each
``u_k(E)`` is fully correlated across energy; independent energy structure
needs separate modes. Profiling the linear ``a_k`` with the unit Gaussian
prior gives exactly the chi-square of the covariance ``C + s^2 sum_k D_k D_k^T``
(``C`` the readout covariance) at fixed ``s`` and offsets, so the profile
interval of ``sigma`` includes these systematics. With a fitted scale the
model is bilinear; the fit profiles it exactly (pulls and offsets linear at
each ``s``, ``s`` searched), so ``a_k`` stays the declared fractional pull.
This is a constrained-nuisance profile, not a marginal likelihood (no
``ln det`` term). With *fixed* nonzero true pulls the prior shrinks the
fitted pulls toward zero, so frequentist coverage of ``sigma`` falls below
95 % when the data constrain a pull more tightly than its prior.

Pulls are confined to their physical box ``1 + a_k u_k(E) >= 0`` on the
band (``-1/max u_k <= a_k <= -1/min u_k``): a pull may not make an
efficiency or the pair envelope negative. The fit is then a bounded linear
least squares at each ``s``. Declare shapes with ``|u_k| << 1`` where the
template matters; a shape near ``-1`` somewhere is outside the Gaussian
nuisance regime and its box binds. A two-port response mode partly duplicates a free source scale: at its box
edge ``a_k = -1/max u_k`` the efficiency keeps only the remnant
``1 - u_k/max u_k`` for a fixed penalty, and a large ``s`` rescales that
remnant while the other pulls become cheap, so ``chi2(s)`` can have a deep
branch far out in ``s``. With a free scale and no scale prior such modes are
refused; give the scale a prior (``scale_prior_rms_fraction``) or fix it.

The modes are linear in ``u_k``: when the data resolve
the second-order change of a 1-sigma mode, the interval is too narrow and
the shifted templates must be refitted instead.

A Gaussian transverse form-factor change, ``u(E) = -A (E/E0)^2``, is
degenerate with the Gaussian duration to first order: it moves the fitted
``sigma`` by ``A hbar^2 / (2 sigma E0^2)`` per unit pull and widens the
interval accordingly instead of being separately identified.

Validation: optical-instrument-observable
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CTRSystematicMode"]

_KINDS = ("transverse", "response")
_HBAR_EV_FS = 0.6582119569509067


@dataclass(frozen=True, eq=False)
class CTRSystematicMode:
    """One declared fractional 1-sigma systematic shape on the template grid.

    ``relative_rms`` is dimensionless, on ``GaussianCTRTemplates.energy_eV``,
    and may change sign with energy; it is the signed 1-sigma *fractional*
    change of the coherent pair envelope (``"transverse"``) or of the port
    spectral efficiency (``"response"``). Not a variance: two modes with
    opposite sign are not interchangeable once other modes are present.

    Validation: optical-instrument-observable
    """

    name: str
    kind: str
    relative_rms: np.ndarray
    provenance: str
    port: int | None = None

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("systematic mode name is required")
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}")
        if self.kind == "transverse" and self.port is not None:
            raise ValueError("transverse modes change the source and act on both ports")
        if self.port is not None and (isinstance(self.port, bool) or self.port not in (0, 1)):
            raise ValueError("port must be 0, 1 or None")
        u = np.array(self.relative_rms, dtype=float, copy=True)
        if u.ndim != 1 or u.size < 2 or not np.all(np.isfinite(u)):
            raise ValueError("relative_rms must be a finite 1-D array on the template grid")
        if not np.any(u != 0):
            raise ValueError("relative_rms is identically zero; omit the mode")
        if not self.provenance or not self.provenance.strip():
            raise ValueError("systematic mode provenance is required")
        u.setflags(write=False)
        object.__setattr__(self, "relative_rms", u)

    @classmethod
    def from_shifted_density(
        cls, name, kind, nominal, shifted, *, provenance, port=None
    ) -> CTRSystematicMode:
        """Mode from a nominal and a 1-sigma-shifted density on the same grid.

        For ``"transverse"``, pass the coherent pair envelopes regenerated with
        the transverse size (or acceptance) moved by its 1-sigma uncertainty;
        for ``"response"``, the nominal and shifted port efficiencies. Where
        both are zero the change is zero; new support (nominal zero, shifted
        nonzero) cannot be a fractional change and is refused.
        """
        a = np.asarray(nominal, dtype=float)
        b = np.asarray(shifted, dtype=float)
        if a.shape != b.shape or a.ndim != 1:
            raise ValueError("nominal and shifted must be 1-D arrays of one shape")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(a < 0):
            raise ValueError("densities must be finite and the nominal nonnegative")
        if np.any((a == 0) & (b != 0)):
            raise ValueError("shifted density has support where the nominal is zero")
        u = np.divide(b - a, a, out=np.zeros_like(a), where=a > 0)
        return cls(name, kind, u, provenance, port)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "port": self.port,
            "provenance": self.provenance,
            "max_abs_relative_rms": float(np.max(np.abs(self.relative_rms))),
        }


def validate_systematics(
    modes, energy_eV, *, free_scale=False, charge_jitter=False
) -> tuple[CTRSystematicMode, ...]:
    """Check types, grid shape, unique names and identifiability; return a tuple.

    ``free_scale``: the source scale is fitted without a prior, so a two-port
    (``port=None``) response mode, which partly duplicates it, is refused. Modes are not
    combined with the ``charge_jitter`` likelihood.
    """
    modes = tuple(modes)
    names = [m.name for m in modes if isinstance(m, CTRSystematicMode)]
    if len(names) != len(modes):
        raise TypeError("systematics must be CTRSystematicMode instances")
    if len(set(names)) != len(names):
        raise ValueError("systematic mode names must be unique")
    if modes and charge_jitter:
        raise ValueError("systematic modes are not combined with the charge-jitter likelihood")
    for mode in modes:
        u = mode.relative_rms
        if u.shape != np.shape(energy_eV):
            raise ValueError(f"systematic mode {mode.name!r} is not on the template energy grid")
        if free_scale and mode.kind == "response" and mode.port is None:
            raise ValueError(
                f"systematic mode {mode.name!r} is a two-port response, partly "
                "degenerate with a free source scale; give the scale a prior "
                "(scale_prior_rms_fraction) or fix it"
            )
    return modes


def systematic_voltage_modes(modes, templates, kernel, sigma_fs) -> np.ndarray:
    """1-sigma voltage changes ``D_k``, shape (n_modes, n_delay, 2), at unit scale.

    ``kernel`` is the calibrated response (n_delay, 2, n_energy) of
    ``optical_ctr_inference._response_kernel``.

    Validation: optical-instrument-observable
    """
    temporal = np.exp(-np.square(templates.energy_eV * float(sigma_fs) / _HBAR_EV_FS))
    pair = templates.coherent_envelope_photons_per_eV_per_pulse * temporal
    total = templates.incoherent_photons_per_eV_per_pulse + pair
    out = np.zeros((len(modes), kernel.shape[0], 2))
    for k, mode in enumerate(modes):
        if mode.kind == "transverse":
            out[k] = kernel @ (mode.relative_rms * pair)
        else:
            ports = (0, 1) if mode.port is None else (mode.port,)
            for p in ports:
                out[k, :, p] = kernel[:, p, :] @ (mode.relative_rms * total)
    return out
