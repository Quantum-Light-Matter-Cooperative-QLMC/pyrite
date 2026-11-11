"""
detectors/alexs_response.py

Charge-integrating forward model of the greateyes ALEX-s 2k512 BI UV1 CCD.

The ALEX-s images a dispersed soft X-ray spectrum: every absorbed photon of
energy ``E`` frees ``E / W`` electrons, the pixel integrates that charge over
the exposure together with dark charge, the well saturates at its full-well
capacity, and the output amplifier adds read noise before the ADC converts
electrons to counts (ADU) with the mode-dependent gain::

    ADU = gain * clip( sum_photons QE(E) * E / W  +  I_dark * t , 0, FW )  (+ read noise)

Datasheet constants (greateyes DS-ALEXs, Rev05, May 2025, public) for the
2k512 BI UV1 are recorded below. The quantum efficiency comes from a digitized
catalog curve (``qe_curve``); until one is supplied, :func:`quantum_efficiency`
falls back to the Beer-Lambert absorption model of
:func:`pyrite.detectors.grating.qe_absorption` with the datasheet 15 um
epitaxial thickness. That fallback is an approximation: its entrance-window
(dead-layer) peak factor is a placeholder and it has no UV1 coating structure.

The register well (400 ke-) only limits on-chip binning, which this model does
not perform, and ADC saturation is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from ._tabulated import TabulatedEfficiency

# ---- datasheet constants: greateyes DS-ALEXs Rev05 (May 2025), 2k512 BI UV1 ---
EV_PER_ELECTRON = 3.66  # mean energy per electron-hole pair [eV] (datasheet)
FULL_WELL_E = 100_000.0  # pixel full well [e-]
REGISTER_WELL_E = 400_000.0  # output-register well [e-]; binning only, not modelled
DARK_CURRENT_E_PER_PX_S = 0.00025  # dark current [e-/px/s] at -100 C
DARK_CURRENT_TEMP_C = -100.0  # temperature of the dark-current specification
EPI_THICKNESS_UM = 15.0  # epitaxial Si thickness, standard sensor (40 um: DD only)
#: Read noise [e- RMS] at the datasheet readout rates [Hz].
READ_NOISE_E_BY_RATE_HZ = MappingProxyType({50.0e3: 3.5, 1.0e6: 6.8, 3.0e6: 10.7})
#: ADC gain [ADU per e-] by gain mode.
GAIN_ADU_PER_E = MappingProxyType({"standard": 1.0, "high-capacity": 0.34})


class AlexsSensor(TypedDict):
    """One greateyes ALEX-s sensor format.

    ``n_pix`` and ``pixel_um`` run along the dispersion (long) axis,
    ``n_rows`` across it; ``active_mm`` is ``(long, short)`` active extent.
    """

    n_pix: int
    n_rows: int
    pixel_um: float
    active_mm: tuple[float, float]


#: ALEX-s formats. The 2k512 row count (515 usable rows) is from the datasheet;
#: the 1k256 row count is the nominal format value, not datasheet-verified.
ALEXS_SENSORS: dict[str, AlexsSensor] = {
    "1k256": AlexsSensor(n_pix=1024, n_rows=256, pixel_um=26.0, active_mm=(26.6, 6.7)),
    "2k512": AlexsSensor(n_pix=2048, n_rows=515, pixel_um=13.5, active_mm=(27.6, 6.9)),
}


def read_noise_e(readout_rate_hz: float) -> float:
    """Datasheet read noise [e- RMS] at one of the specified readout rates."""
    for rate, noise in READ_NOISE_E_BY_RATE_HZ.items():
        if np.isclose(float(readout_rate_hz), rate, rtol=1.0e-9, atol=0.0):
            return noise
    raise ValueError(
        f"no ALEX-s read-noise specification at {readout_rate_hz:g} Hz; "
        f"datasheet rates are {sorted(READ_NOISE_E_BY_RATE_HZ)} Hz"
    )


def gain_adu_per_e(gain_mode: str) -> float:
    """ADC gain [ADU/e-] for a named ALEX-s gain mode."""
    try:
        return GAIN_ADU_PER_E[gain_mode]
    except KeyError:
        raise ValueError(
            f"unknown ALEX-s gain mode {gain_mode!r}; have {sorted(GAIN_ADU_PER_E)}"
        ) from None


def quantum_efficiency(E_eV, qe_curve: TabulatedEfficiency | None = None) -> np.ndarray:
    """ALEX-s quantum efficiency at photon energy ``E_eV`` [eV].

    A tabulated ``qe_curve`` (the digitized datasheet curve from the catalog)
    is interpolated and is zero outside its band. Without one, the
    Beer-Lambert fallback ``qe_absorption(E, active_um=EPI_THICKNESS_UM)``
    applies (the ``alexs-qe-absorption`` claim): an approximation whose
    entrance-window peak is a placeholder, not the vendor curve.
    """
    if qe_curve is not None:
        return qe_curve(E_eV)
    from .grating import qe_absorption

    return qe_absorption(E_eV, active_um=EPI_THICKNESS_UM)


def signal_electrons(E_eV, photons, qe) -> np.ndarray:
    """Expected collected charge [e-] from incident photons at energy ``E_eV``.

    ``N_e = N * QE(E) * E / W`` with ``W = EV_PER_ELECTRON``: each absorbed
    photon converts its whole energy into electron-hole pairs at the mean
    pair-creation energy (Fano fluctuations are below the read noise for an
    integrating frame and are neglected). ``photons`` may be a count or a
    density per eV; the result carries the same measure. Limiting cases:
    ``QE = 0`` gives no charge, and one absorbed photon of energy ``W`` gives
    one electron.

    Validation: alexs-adu-conversion
    """
    E = np.asarray(E_eV, dtype=float)
    return np.asarray(photons, dtype=float) * np.asarray(qe, dtype=float) * E / EV_PER_ELECTRON


def alexs_frame(
    signal_e,
    *,
    exposure_s: float,
    readout_rate_hz: float = 1.0e6,
    gain_mode: str = "standard",
    photons=None,
    rng: np.random.Generator | None = None,
    shot_noise: bool = False,
    read_noise: bool = False,
    dark: bool = True,
) -> np.ndarray:
    """Convert expected per-pixel signal charge into one ALEX-s frame in ADU.

    Parameters
    ----------
    signal_e
        Expected signal charge per pixel [e-] over the exposure (any shape).
    exposure_s
        Exposure time [s]; dark charge is ``DARK_CURRENT_E_PER_PX_S * t``.
    readout_rate_hz
        Datasheet readout rate selecting the read noise.
    gain_mode
        ``"standard"`` (1 ADU/e-) or ``"high-capacity"`` (0.34 ADU/e-).
    photons
        Expected absorbed photons per pixel, same shape as ``signal_e``;
        required for ``shot_noise`` so the Poisson draw acts on photons and
        each drawn photon keeps the pixel's mean charge ``signal_e/photons``.
    rng
        Explicit generator; required when ``shot_noise`` or ``read_noise``.
    shot_noise
        Poisson-draw absorbed photons and dark electrons.
    read_noise
        Add zero-mean Gaussian read noise of the datasheet RMS per pixel.
    dark
        Include dark charge.

    Returns
    -------
    numpy.ndarray
        ADU per pixel (float). Charge is clipped to ``[0, FULL_WELL_E]`` before
        read noise; the noise-free result is
        ``gain * min(signal_e + I_dark t, FULL_WELL_E)``.

    Notes
    -----
    Read-noise variance in ADU is ``(gain * sigma_read)**2``; dark charge is
    linear in exposure. The dark rate is the datasheet value at -100 C.

    Validation: alexs-adu-conversion
    """
    gain = gain_adu_per_e(gain_mode)
    sigma = read_noise_e(readout_rate_hz)
    if exposure_s < 0 or not np.isfinite(exposure_s):
        raise ValueError("exposure_s must be finite and nonnegative")
    if (shot_noise or read_noise) and rng is None:
        raise ValueError("shot_noise/read_noise require an explicit numpy Generator rng")
    charge = np.clip(np.asarray(signal_e, dtype=float), 0.0, None)
    if shot_noise:
        assert rng is not None
        if photons is None:
            raise ValueError("shot_noise requires the expected photon image `photons`")
        expected = np.clip(np.asarray(photons, dtype=float), 0.0, None)
        if expected.shape != charge.shape:
            raise ValueError("photons must match signal_e in shape")
        per_photon = np.divide(charge, expected, out=np.zeros_like(charge), where=expected > 0)
        charge = rng.poisson(expected) * per_photon
    dark_e = DARK_CURRENT_E_PER_PX_S * float(exposure_s) if dark else 0.0
    if dark_e > 0.0:
        if shot_noise:
            assert rng is not None
            charge = charge + rng.poisson(dark_e, charge.shape)
        else:
            charge = charge + dark_e
    charge = np.minimum(charge, FULL_WELL_E)
    if read_noise:
        assert rng is not None
        charge = charge + rng.normal(0.0, sigma, charge.shape)
    return gain * charge


@dataclass(frozen=True)
class AlexsFormat:
    """Pixel geometry of one ALEX-s format, for pixel-grid factories."""

    columns: int
    rows: int
    pitch_mm: float

    @classmethod
    def named(cls, variant: str) -> AlexsFormat:
        """Look up a named format (``"1k256"`` or ``"2k512"``)."""
        if variant not in ALEXS_SENSORS:
            raise KeyError(f"unknown ALEX-s variant {variant!r} (have {list(ALEXS_SENSORS)})")
        sensor = ALEXS_SENSORS[variant]
        return cls(sensor["n_pix"], sensor["n_rows"], sensor["pixel_um"] * 1.0e-3)

    @property
    def height_mm(self) -> float:
        """Active height across the dispersion axis [mm]."""
        return self.rows * self.pitch_mm

    @property
    def width_mm(self) -> float:
        """Active width along the dispersion axis [mm]."""
        return self.columns * self.pitch_mm


__all__ = [
    "ALEXS_SENSORS",
    "DARK_CURRENT_E_PER_PX_S",
    "EPI_THICKNESS_UM",
    "EV_PER_ELECTRON",
    "FULL_WELL_E",
    "GAIN_ADU_PER_E",
    "READ_NOISE_E_BY_RATE_HZ",
    "REGISTER_WELL_E",
    "AlexsFormat",
    "AlexsSensor",
    "alexs_frame",
    "gain_adu_per_e",
    "quantum_efficiency",
    "read_noise_e",
    "signal_electrons",
]
