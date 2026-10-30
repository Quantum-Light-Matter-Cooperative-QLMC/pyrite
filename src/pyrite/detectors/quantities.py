"""Explicit Pint adapters for numerical detector and grating helpers (#816).

Spectral densities have no universal photon/count normalization. Requiring a
quantity for them ensures that broadening preserves the caller's density unit,
dispersion multiplies it by energy/length, and pixel integration by energy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._units import _is_pint_quantity, normalize_unit


def _input(value: Any, unit: str, name: str, *, kind: str | None = None):
    if not _is_pint_quantity(value):
        raise TypeError(f"{name} requires a Pint quantity convertible to {unit}")
    if np.asarray(value.magnitude).dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real-valued")
    converted = np.asarray(normalize_unit(value, unit, field=name, expected=unit, kind=kind))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        magnitudes = converted.astype(float)
    if np.any(
        np.isfinite(converted) & (~np.isfinite(magnitudes) | ((converted != 0) & (magnitudes == 0)))
    ):
        raise ValueError(f"{name} cannot be represented in float64 {unit}")
    return magnitudes.item() if magnitudes.ndim == 0 else magnitudes.copy()


def _out(value: Any, unit: str):
    from ..units import Q_

    return Q_(np.array(value, copy=True), unit)


def _density(value: Any):
    """Density is caller-supplied and must already carry a concrete unit."""
    if not _is_pint_quantity(value):
        raise TypeError("spectrum requires a Pint quantity (such as photon/eV)")
    original = np.asarray(value.magnitude)
    if original.dtype.kind not in "iuf":
        raise TypeError("spectrum density must be real")
    from ..units import ureg

    # Normalizing through the target registry avoids registry-mismatch
    # multiplication if the input originated from an external Pint registry.
    try:
        unit = ureg.Unit(str(value.units))
    except Exception as exc:
        raise ValueError("spectrum uses units unknown to the PyRITE registry") from exc
    data = np.array(original, dtype=float, copy=True)
    return data, unit


def detector_efficiency_q(energy, *, polymer_thickness, aluminum_thickness, grid_open=0.78):
    """Dimensionless SDD quantum efficiency, energy and window lengths with units."""
    from .response import detector_efficiency

    result = detector_efficiency(
        _input(energy, "eV", "energy"),
        polymer_nm=_input(polymer_thickness, "nm", "polymer_thickness"),
        al_nm=_input(aluminum_thickness, "nm", "aluminum_thickness"),
        grid_open=grid_open,
    )
    return _out(result, "dimensionless")


def convolve_detector_q(energy, spectrum, fwhm, *, return_outside=False):
    """Gaussian broadening on the energy grid, keeping density and lost mass units."""
    from ..units import ureg
    from .response import convolve_detector

    e = _input(energy, "eV", "energy")
    width = _input(fwhm, "eV", "fwhm")
    values, density_unit = _density(spectrum)
    result = convolve_detector(e, values, width, return_outside=return_outside)
    if not return_outside:
        return _out(result, str(density_unit))
    blurred, (below, above) = result
    return (
        _out(blurred, str(density_unit)),
        (_out(below, str(density_unit * ureg.eV)), _out(above, str(density_unit * ureg.eV))),
    )


def grating_q(*, groove_density, incidence_angle, order=1, coating="Au", groove_efficiency=1.0):
    """Build the unchanged numeric Grating record from explicit density and angle."""
    from .grating import Grating

    return Grating(
        groove_density_per_mm=_input(groove_density, "1/mm", "groove_density"),
        alpha_rad=_input(incidence_angle, "radian", "incidence_angle", kind="angle"),
        order=order,
        coating=coating,
        groove_efficiency=groove_efficiency,
    )


def grating_parameters(grating):
    """Detached physical geometry in a dict; selectors stay numeric."""
    return {
        "groove_density": _out(grating.groove_density_per_mm, "1/mm"),
        "incidence_angle": _out(grating.alpha_rad, "radian"),
        "grazing_angle": _out(grating.grazing_angle_rad, "radian"),
        "groove_spacing": _out(grating.d_angstrom, "angstrom"),
        "order": grating.order,
        "coating": grating.coating,
        "groove_efficiency": grating.groove_efficiency,
    }


def ccd_q(*, n_pix, pixel_pitch):
    """Build unchanged SimpleCCD numeric geometry using an explicit length."""
    from .grating import SimpleCCD

    return SimpleCCD(n_pix=n_pix, pixel_mm=_input(pixel_pitch, "mm", "pixel_pitch"))


def ccd_parameters(ccd):
    return {
        "n_pix": ccd.n_pix,
        "pixel_pitch": _out(ccd.pixel_mm, "mm"),
        "width": _out(ccd.width_mm, "mm"),
    }


def wavelength_q(energy):
    from .grating import wavelength_angstrom

    return _out(wavelength_angstrom(_input(energy, "eV", "energy")), "angstrom")


def groove_spacing_q(groove_density):
    from .grating import groove_spacing_angstrom

    return _out(
        groove_spacing_angstrom(_input(groove_density, "1/mm", "groove_density")), "angstrom"
    )


def diffraction_angle_q(grating, energy):
    from .grating import Grating

    if not isinstance(grating, Grating):
        raise TypeError("grating must be a Grating")
    return _out(grating.diffraction_angle_rad(_input(energy, "eV", "energy")), "radian")


def angular_dispersion_q(grating, energy):
    return _out(
        grating.angular_dispersion_rad_per_angstrom(_input(energy, "eV", "energy")),
        "radian/angstrom",
    )


def grating_throughput_q(grating, energy):
    return _out(grating.throughput(_input(energy, "eV", "energy")), "dimensionless")


def grating_reflectivity_q(grating, energy):
    return _out(grating.reflectivity(_input(energy, "eV", "energy")), "dimensionless")


def detector_position_q(beta, reference_beta, distance):
    from .grating import detector_position_mm

    return _out(
        detector_position_mm(
            _input(beta, "radian", "beta", kind="angle"),
            _input(reference_beta, "radian", "reference_beta", kind="angle"),
            _input(distance, "mm", "distance"),
        ),
        "mm",
    )


def resolving_power_q(energy, grating, distance, pixel_pitch, *, reference_beta=None):
    from .grating import resolving_power

    return _out(
        resolving_power(
            _input(energy, "eV", "energy"),
            grating,
            _input(distance, "mm", "distance"),
            _input(pixel_pitch, "mm", "pixel_pitch"),
            beta_ref_rad=(
                None
                if reference_beta is None
                else _input(reference_beta, "radian", "reference_beta", kind="angle")
            ),
        ),
        "dimensionless",
    )


def disperse_spectrum_q(
    energy,
    spectrum,
    grating,
    distance,
    *,
    reference_beta=None,
    weight_by_throughput=False,
):
    """Position in mm; density converts from per-energy to per-length."""
    from ..units import ureg
    from .grating import disperse_spectrum

    values, unit = _density(spectrum)
    x, intensity = disperse_spectrum(
        _input(energy, "eV", "energy"),
        values,
        grating,
        _input(distance, "mm", "distance"),
        beta_ref_rad=None
        if reference_beta is None
        else _input(reference_beta, "radian", "reference_beta", kind="angle"),
        weight_by_throughput=weight_by_throughput,
    )
    return _out(x, "mm"), _out(intensity, str(unit * ureg.eV / ureg.mm))


def bin_to_pixels_q(position, intensity, ccd, *, center=None):
    """Rebin spatial density to the integrated quantity in each CCD pixel."""
    from ..units import ureg
    from .grating import bin_to_pixels

    values, unit = _density(intensity)
    x, counts = bin_to_pixels(
        _input(position, "mm", "position"),
        values,
        ccd,
        center_mm=None if center is None else _input(center, "mm", "center"),
    )
    return _out(x, "mm"), _out(counts, str(unit * ureg.mm))


def detected_image_q(
    energy,
    spectrum,
    grating,
    ccd,
    distance,
    *,
    weight_by_throughput=True,
    reference_beta=None,
    center=None,
    physical=False,
    active_thickness=None,
    depletion_voltage=None,
    temperature=None,
    peak_qe=None,
):
    """Pixel centres in mm and integrated counts (density × input energy)."""
    from ..units import ureg
    from .grating import detected_image, detected_image_physical

    if not physical and any(
        x is not None for x in (active_thickness, depletion_voltage, temperature, peak_qe)
    ):
        raise ValueError("physical CCD controls require physical=True")
    values, unit = _density(spectrum)
    args = dict(
        weight_by_throughput=weight_by_throughput,
        beta_ref_rad=None
        if reference_beta is None
        else _input(reference_beta, "radian", "reference_beta", kind="angle"),
        center_mm=None if center is None else _input(center, "mm", "center"),
    )
    if physical:
        args.update(
            active_um=None
            if active_thickness is None
            else _input(active_thickness, "um", "active_thickness"),
            v_dep=None
            if depletion_voltage is None
            else _input(depletion_voltage, "V", "depletion_voltage"),
            temp_c=None
            if temperature is None
            else _input(temperature, "degC", "temperature", kind="temperature"),
            peak_qe=peak_qe,
        )
    method = detected_image_physical if physical else detected_image
    x, counts = method(
        _input(energy, "eV", "energy"),
        values,
        grating,
        ccd,
        _input(distance, "mm", "distance"),
        **args,
    )
    return _out(x, "mm"), _out(counts, str(unit * ureg.eV))


def coating_number_density_q(element):
    """Number of coating atoms per cubic angstrom, not electric charge."""
    from .grating import coating_number_density_per_ang3

    return _out(coating_number_density_per_ang3(element), "1/angstrom**3")


def qe_absorption_q(energy, *, active_thickness=None, peak=None):
    """Physical CCD quantum efficiency, with optional active thickness."""
    from .grating import qe_absorption

    return _out(
        qe_absorption(
            _input(energy, "eV", "energy"),
            active_um=None
            if active_thickness is None
            else _input(active_thickness, "um", "active_thickness"),
            peak=peak,
        ),
        "dimensionless",
    )


def charge_cloud_sigma_q(
    energy,
    *,
    active_thickness=None,
    depletion_voltage=None,
    temperature=None,
):
    """CCD charge-cloud RMS width; temperature is *absolute*, not delta K."""
    from .grating import charge_cloud_sigma_um

    return _out(
        charge_cloud_sigma_um(
            _input(energy, "eV", "energy"),
            active_um=None
            if active_thickness is None
            else _input(active_thickness, "um", "active_thickness"),
            v_dep=None
            if depletion_voltage is None
            else _input(depletion_voltage, "V", "depletion_voltage"),
            temp_c=None
            if temperature is None
            else _input(temperature, "degC", "temperature", kind="temperature"),
        ),
        "um",
    )


def energy_fwhm_q(energy, *, n_pix=4, read_noise_e=None):
    """CCD single-photon FWHM energy resolution (read noise in electron counts)."""
    from .grating import energy_fwhm_eV

    return _out(
        energy_fwhm_eV(
            _input(energy, "eV", "energy"),
            n_pix=n_pix,
            read_noise_e=read_noise_e,
        ),
        "eV",
    )


__all__ = [
    "detector_efficiency_q",
    "convolve_detector_q",
    "grating_q",
    "grating_parameters",
    "ccd_q",
    "ccd_parameters",
    "wavelength_q",
    "groove_spacing_q",
    "diffraction_angle_q",
    "angular_dispersion_q",
    "grating_throughput_q",
    "grating_reflectivity_q",
    "detector_position_q",
    "resolving_power_q",
    "disperse_spectrum_q",
    "bin_to_pixels_q",
    "detected_image_q",
    "coating_number_density_q",
    "qe_absorption_q",
    "charge_cloud_sigma_q",
    "energy_fwhm_q",
]
