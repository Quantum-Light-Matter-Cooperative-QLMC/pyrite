"""Finite-band enclosures of the production coherent propagation phase (#350).

The phase law is ``w(E) = E (1 - Re sqrt(1 + chi_0(E))) / hbar c``.
Chantler's f1 is a cubic spline selected using the full request's extrema;
f2 is log-linear. Reconstruct that same selection, split at its knots, and
bound the secant residual by ``max |w''| (hi-lo)**2 / 8``. No local call to
``henke_dispersion`` or endpoint-only residual estimate supplies this bound.

Validation: coherent-line-grid-windowed-resolution
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from math import comb

import numpy as np
import xraydb
from scipy.interpolate import PPoly, make_interp_spline

from ...materials.atomic import Z_TABLE
from ...materials.crystal import CRYSTALS, HBARC_EV_ANG, HC_EV_ANG, R_E_ANG


def _outward(lo: float, hi: float, scale: float = 0.0) -> tuple[float, float]:
    """Conservative float64 allowance around a real-arithmetic enclosure."""
    margin = 64.0 * np.finfo(float).eps * max(abs(lo), abs(hi), scale)
    return float(np.nextafter(lo - margin, -np.inf)), float(np.nextafter(hi + margin, np.inf))


def _polynomial_range(poly: PPoly, lo: float, hi: float) -> tuple[float, float]:
    """Bernstein convex-hull enclosure on one polynomial piece, including extrema."""
    index = int(np.searchsorted(poly.x, 0.5 * (lo + hi), side="right") - 1)
    index = min(max(index, 0), poly.c.shape[1] - 1)
    powers = poly.c[:, index][::-1]
    degree = powers.size - 1
    offset, width = lo - poly.x[index], hi - lo
    terms = [
        np.array(
            [powers[j] * comb(j, i) * offset ** (j - i) * width**i for j in range(i, degree + 1)]
        )
        for i in range(degree + 1)
    ]
    translated = np.array([term.sum() for term in terms])
    bernstein = np.array(
        [
            sum(translated[i] * comb(k, i) / comb(degree, i) for i in range(k + 1))
            for k in range(degree + 1)
        ]
    )
    return _outward(
        float(bernstein.min()),
        float(bernstein.max()),
        float(sum(np.abs(term).sum() for term in terms)),
    )


@dataclass(frozen=True, slots=True)
class DispersionCertificate:
    """Secant phase law and a uniform residual on one smooth energy interval.

    ``slope``: 1/(Ang eV); ``intercept``/``residual_max``: 1/Ang;
    ``first_derivative_max``: 1/(Ang eV). Bounds describe the reconstructed
    material law, with an explicit float64 evaluation allowance.

    Validation: coherent-line-grid-windowed-resolution
    """

    start_eV: float
    stop_eV: float
    slope: float
    intercept: float
    residual_max: float
    first_derivative_max: float


@dataclass(frozen=True, slots=True)
class _AtomLaw:
    count: int
    f1: tuple[PPoly, PPoly, PPoly]
    energies: np.ndarray
    f2: np.ndarray

    def imaginary(self, energies: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.log(energies), np.log(self.energies), np.log(self.f2)))


class CoherentDispersionLaw:
    """The full-axis material phase and certified smooth subintervals.

    This adapter follows xraydb's current Chantler selection exactly. Its
    raw table access is intentionally isolated here; changes to xraydb's
    interpolation must update the production-agreement regression too.

    Source: ``chi_0 = -C sum(f1 + i f2)/E**2`` and
    ``delta_omega = E (1 - Re sqrt(1 + chi_0))/hbar c`` from the production
    material/propagation law. Positive finite energy, single crystal and
    float64 interpolation are assumed. Vacuum gives zero propagation phase.

    Validation: coherent-line-grid-windowed-resolution
    """

    def __init__(self, crystal: str, start_eV: float, stop_eV: float, *, use_henke: bool = True):
        self.start, self.stop = float(start_eV), float(stop_eV)
        if (
            not np.isfinite(self.start)
            or not np.isfinite(self.stop)
            or not 0 < self.start < self.stop
        ):
            raise ValueError(
                "a coherent dispersion certificate needs a finite positive energy band"
            )
        info = CRYSTALS[crystal]
        self.prefactor = R_E_ANG * HC_EV_ANG**2 / (np.pi * info["V_cell"])
        self.atoms: list[_AtomLaw] = []
        self.forward_constant = 0.0
        digest = hashlib.sha256()
        digest.update(f"chantler-cubic-log-v1:{xraydb.__version__}".encode())
        digest.update(
            np.asarray([self.start, self.stop, self.prefactor, use_henke], dtype=float).tobytes()
        )
        breaks = [self.start, self.stop]
        populations = Counter(element for element, _position in info["basis"])
        for element, count in populations.items():
            digest.update(f"{element}:{count}:{Z_TABLE[element]}".encode())
            if not use_henke:
                self.forward_constant += count * Z_TABLE[element]
                continue
            # _from_chantler trims three native nodes beyond the full query
            # extrema, then interpolates f1 with a not-a-knot cubic. f2 uses
            # the same trimmed table with log-linear interpolation.
            row = xraydb.get_xraydb().get_cache("Chantler", column="element", value=element)[0]
            native = np.asarray(json.loads(row.energy), dtype=float)
            if self.start <= native.min() or self.stop >= native.max() or self.stop > 1e6:
                raise ValueError(
                    f"coherent dispersion band is outside the supported {element} table"
                )
            first = max(0, int(np.searchsorted(native, self.start, side="right")) - 4)
            last = min(native.size, int(np.searchsorted(native, self.stop, side="right")) + 2)
            energies = native[first : last + 1]
            anomalous = np.asarray(json.loads(row.f1), dtype=float)[first : last + 1]
            imaginary = np.asarray(json.loads(row.f2), dtype=float)[first : last + 1]
            anomalous = np.where(np.abs(anomalous) < 1e-99, 1e-99, anomalous)
            imaginary = np.where(np.abs(imaginary) < 1e-99, 1e-99, imaginary)
            if (
                energies.size < 4
                or np.any(imaginary <= 0)
                or np.any(~np.isfinite(anomalous))
                or np.any(~np.isfinite(imaginary))
                or np.any(~np.isfinite(energies))
                or np.any(np.diff(energies) <= 0)
            ):
                raise ValueError(f"invalid Chantler interpolation data for {element}")
            for array in (energies, anomalous, imaginary):
                digest.update(array.tobytes())
            spline = PPoly.from_spline(make_interp_spline(energies, anomalous, k=3))
            # Add Z to the cubic's constant coefficients; it is not part of
            # either derivative. This is chi_0's forward-factor convention.
            spline.c[-1] += Z_TABLE[element]
            self.atoms.append(
                _AtomLaw(
                    count, (spline, spline.derivative(), spline.derivative(2)), energies, imaginary
                )
            )
            breaks.extend(energies[(energies > self.start) & (energies < self.stop)])
            breaks.extend(spline.x[(spline.x > self.start) & (spline.x < self.stop)])
        self.breaks = np.unique(breaks)
        self.fingerprint = digest.hexdigest()

    def __call__(self, energies):
        """Evaluate exactly the full-axis interpolation, even for a local query."""
        energy = np.asarray(energies, dtype=float)
        if np.any(~np.isfinite(energy)) or np.any((energy < self.start) | (energy > self.stop)):
            raise ValueError("dispersion evaluation must remain inside its certified full axis")
        forward = np.full(energy.shape, complex(self.forward_constant))
        for atom in self.atoms:
            forward += atom.count * (atom.f1[0](energy) + 1j * atom.imaginary(energy))
        chi = -self.prefactor * forward / energy**2
        return (1.0 - np.sqrt(1.0 + chi).real) * energy / HBARC_EV_ANG

    def certificate(self, start_eV: float, stop_eV: float) -> DispersionCertificate:
        """Bound the secant residual on an interval without interior interpolation knots.

        Source: ``n'=chi'/(2n)``, ``n''=chi''/(2n)-chi'^2/(4n^3)``
        and ``w''=(-2 Re n' - E Re n'')/hbar c``. Cubic Bernstein bounds
        and power-law endpoint bounds enclose chi and both derivatives.
        A possible zero of ``1+chi`` is refused rather than differentiated.
        Vacuum gives zero real-arithmetic residual; the explicit numerical
        evaluation allowance remains.

        Validation: coherent-line-grid-windowed-resolution
        """
        lo, hi = float(start_eV), float(stop_eV)
        if not self.start <= lo < hi <= self.stop:
            raise ValueError("a dispersion certificate must be inside its full axis")
        if np.any((self.breaks > lo) & (self.breaks < hi)):
            raise ValueError("split the dispersion certificate at every interpolation knot")
        real_lo = real_hi = self.forward_constant
        real_abs = abs(self.forward_constant)
        real_first = real_second = imag_lo = imag_hi = imag_first = imag_second = 0.0
        for atom in self.atoms:
            ranges = [_polynomial_range(poly, lo, hi) for poly in atom.f1]
            real_lo += atom.count * ranges[0][0]
            real_hi += atom.count * ranges[0][1]
            real_abs += atom.count * max(abs(ranges[0][0]), abs(ranges[0][1]))
            real_first += atom.count * max(abs(ranges[1][0]), abs(ranges[1][1]))
            real_second += atom.count * max(abs(ranges[2][0]), abs(ranges[2][1]))
            index = int(np.searchsorted(atom.energies, 0.5 * (lo + hi), side="right") - 1)
            exponent = np.log(atom.f2[index + 1] / atom.f2[index]) / np.log(
                atom.energies[index + 1] / atom.energies[index]
            )
            endpoints = atom.imaginary(np.array([lo, hi]))
            bottom, top = _outward(float(endpoints.min()), float(endpoints.max()))
            imag_lo += atom.count * max(bottom, 0.0)
            imag_hi += atom.count * top
            # Differentiate f2/E^2 directly, retaining its power exponent.
            imag_first += atom.count * abs(exponent - 2.0) * top / lo**3
            imag_second += atom.count * abs((exponent - 2.0) * (exponent - 3.0)) * top / lo**4
        ratios = np.array([real_lo / lo**2, real_lo / hi**2, real_hi / lo**2, real_hi / hi**2])
        x_lo, x_hi = 1.0 - self.prefactor * ratios.max(), 1.0 - self.prefactor * ratios.min()
        minimum_real = max(x_lo, -x_hi, 0.0)
        minimum_imag = self.prefactor * imag_lo / hi**2
        minimum_n = np.sqrt(np.hypot(minimum_real, minimum_imag))
        if not np.isfinite(minimum_n) or minimum_n <= 0.0:
            raise ValueError("the dispersion interval cannot exclude a refractive square-root zero")
        chi_abs = self.prefactor * np.hypot(real_abs, imag_hi) / lo**2
        chi_first = self.prefactor * np.hypot(real_first / lo**2 + 2 * real_abs / lo**3, imag_first)
        chi_second = self.prefactor * np.hypot(
            real_second / lo**2 + 4 * real_first / lo**3 + 6 * real_abs / lo**4, imag_second
        )
        first_bound = (chi_abs + hi * chi_first / (2 * minimum_n)) / HBARC_EV_ANG
        second_bound = (
            chi_first / minimum_n
            + hi * chi_second / (2 * minimum_n)
            + hi * chi_first**2 / (4 * minimum_n**3)
        ) / HBARC_EV_ANG
        values = self(np.array([lo, hi]))
        slope = float((values[1] - values[0]) / (hi - lo))
        intercept = float(values[0] - slope * lo)
        # The analytic secant bound is exact in real arithmetic. Retain a
        # separate allowance for constructing and evaluating the float64
        # interpolant, including cancellation in 1-Re(n) and sE+b.
        rounding = (
            64
            * np.finfo(float).eps
            * ((hi / HBARC_EV_ANG) * (1 + np.sqrt(1 + chi_abs)) + abs(intercept) + hi * abs(slope))
        )
        residual = float(np.nextafter(second_bound * (hi - lo) ** 2 / 8 + rounding, np.inf))
        if not np.all(np.isfinite([slope, intercept, residual, first_bound])):
            raise ValueError("the material interval has no finite float64 dispersion certificate")
        return DispersionCertificate(
            lo, hi, slope, intercept, residual, float(np.nextafter(first_bound, np.inf))
        )
