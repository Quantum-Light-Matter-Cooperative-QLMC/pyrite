"""
detectors/grating_image.py

Analytic slit-image kernel and 2D focal-plane image for the flat-field grating
spectrometer (issue #1081): spectrum -> entrance slit -> grating -> ALEX-s.

The entrance slit is not a scene object here: each wavelength's image on the
focal plane is a rect of the slit's anamorphic image width, optionally
convolved with a Gaussian aberration width, and along the slit the
grazing-incidence astigmatism spreads it uniformly over a height H(lambda).
No ray tracing. The finite-slit/finite-source model is issue #1076; slit
acceptance (solid angle from the source) is issue #1075.
"""

from dataclasses import dataclass

import numpy as np

from .grating import (
    Grating,
    SimpleCCD,
    _focal_plane,
    bin_to_pixels,
    disperse_spectrum,
    wavelength_angstrom,
)

_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
_ZERO_WIDTH_MM = 1.0e-12


@dataclass(frozen=True)
class SlitImage:
    """Entrance-slit parameters for the analytic slit-image kernel.

    Parameters
    ----------
    width_mm
        Slit width ``w`` along the dispersion direction [mm]; zero is the
        point-slit limit.
    entrance_arm_mm
        Slit-to-grating distance ``r`` [mm]; ``None`` uses the grating mount's
        entrance arm.
    resolving_power
        Optional vendor resolving power ``lambda/dlambda`` whose FWHM, mapped
        through the linear dispersion, sets a Gaussian aberration width.
    """

    width_mm: float
    entrance_arm_mm: float | None = None
    resolving_power: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.width_mm) or self.width_mm < 0.0:
            raise ValueError("SlitImage.width_mm must be finite and nonnegative")
        if self.entrance_arm_mm is not None and not self.entrance_arm_mm > 0.0:
            raise ValueError("SlitImage.entrance_arm_mm must be positive")
        if self.resolving_power is not None and not self.resolving_power > 0.0:
            raise ValueError("SlitImage.resolving_power must be positive")

    def arm_mm(self, grating: Grating) -> float:
        """Resolved entrance arm ``r`` [mm]."""
        if self.entrance_arm_mm is not None:
            return float(self.entrance_arm_mm)
        if grating.mount is None:
            raise ValueError("SlitImage.entrance_arm_mm is required without a grating mount")
        return grating.mount.entrance_arm_mm


@dataclass(frozen=True)
class AlongSlitSpread:
    """Uniform along-slit (cross-dispersion) spread of each wavelength's image.

    Give exactly one of ``height_mm`` (astigmatic image height ``H``) or
    ``capture_fraction`` (vendor-tabulated fraction reaching the sensor). With
    an empty ``energy_eV`` the single value applies at every energy; otherwise
    values interpolate linearly in energy and energies outside the table are
    not delivered (capture zero, never extrapolated). Values come from the
    user's private spot-diagram data; nothing is bundled.
    """

    energy_eV: tuple[float, ...] = ()
    height_mm: tuple[float, ...] | None = None
    capture_fraction: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (self.height_mm is None) == (self.capture_fraction is None):
            raise ValueError("AlongSlitSpread needs exactly one of height_mm or capture_fraction")
        values = np.asarray(
            self.height_mm if self.height_mm is not None else self.capture_fraction, float
        )
        expected = max(len(self.energy_eV), 1)
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            raise ValueError("AlongSlitSpread values must be finite and match energy_eV")
        if self.height_mm is not None and np.any(values <= 0.0):
            raise ValueError("AlongSlitSpread.height_mm must be positive")
        if self.capture_fraction is not None and np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("AlongSlitSpread.capture_fraction must lie in [0, 1]")
        if np.any(np.diff(self.energy_eV) <= 0.0):
            raise ValueError("AlongSlitSpread.energy_eV must be strictly increasing")

    def _sample(self, values, E) -> np.ndarray:
        if not self.energy_eV:
            return np.full(E.shape, float(values[0]))
        inside = (E >= self.energy_eV[0]) & (E <= self.energy_eV[-1])
        return np.where(inside, np.interp(E, self.energy_eV, values), np.nan)

    def capture(self, E_eV, sensor_height_mm: float) -> np.ndarray:
        """Fraction of each wavelength's flux landing on the sensor rows.

        A uniform spread of height ``H`` centred on a sensor of height
        ``H_det`` delivers ``min(1, H_det / H)``; a tabulated capture fraction
        is used as given. Zero outside the tabulated band.
        """
        E = np.asarray(E_eV, float)
        if self.capture_fraction is not None:
            out = self._sample(self.capture_fraction, E)
        else:
            out = np.minimum(1.0, sensor_height_mm / self._sample(self.height_mm, E))
        return np.nan_to_num(out, nan=0.0)

    def effective_height_mm(self, E_eV, sensor_height_mm: float) -> np.ndarray:
        """Height over which flux spreads; a capture ``f`` acts as ``H_det / f``."""
        E = np.asarray(E_eV, float)
        if self.height_mm is not None:
            return np.nan_to_num(self._sample(self.height_mm, E), nan=sensor_height_mm)
        f = self.capture(E, sensor_height_mm)
        return np.where(f > 0.0, sensor_height_mm / np.where(f > 0.0, f, 1.0), sensor_height_mm)


def slit_image_width_mm(
    E_eV, grating, slit_width_mm, *, entrance_arm_mm=None, distance_mm=None, beta_ref_rad=None
):
    """Width of the entrance-slit image on the focal plane [mm].

    At fixed wavelength the grating equation ``sin alpha + sin beta = m
    lambda/d`` gives ``cos(alpha) dalpha + cos(beta) dbeta = 0``; a slit of
    width ``w`` at distance ``r`` subtends ``dalpha = w/r``, so its image
    subtends ``dbeta = w cos(alpha) / (r cos(beta))``. Mapping through the
    focal plane, ``ds/dbeta = r'(beta)/cos(beta - beta_H)`` with
    ``r' = r_H'/cos(beta - beta_H)`` the exit arm, gives

        w_plane = w * (cos alpha / cos beta) * (r' / r) / cos(beta - beta_H),

    the anamorphic image width ``w' = w (cos alpha/cos beta)(r'/r)`` projected
    onto the (possibly inclined) plane. At normal incidence on the plane
    (``beta = beta_H``) ``w_plane = w'``; ``w -> 0`` gives zero width (the
    point-slit dispersion). Source: Samson & Ederer, *Vacuum Ultraviolet
    Spectroscopy* (Academic, 2000), grating-equation differentials; flat-field
    geometry as in :meth:`FlatFieldMount.position_mm`.

    ``distance_mm``/``beta_ref_rad`` select the flat-detector map as in
    :func:`disperse_spectrum`; ``None`` uses the grating mount. Non-propagating
    energies return NaN.

    Validation: grating-slit-image-width
    """
    E = np.asarray(E_eV, float)
    if entrance_arm_mm is None:
        entrance_arm_mm = SlitImage(0.0).arm_mm(grating)
    beta = grating.diffraction_angle_rad(E)
    _, slope = _focal_plane(grating, beta, distance_mm, beta_ref_rad)
    dbeta = float(slit_width_mm) * np.cos(grating.alpha_rad) / (entrance_arm_mm * np.cos(beta))
    return np.abs(slope * dbeta)


def _aberration_sigma_mm(E, beta, slope, grating, resolving_power):
    """Gaussian sigma [mm] of a vendor resolving power through the dispersion."""
    if resolving_power is None:
        return np.zeros_like(E)
    dlam_fwhm = wavelength_angstrom(E) / float(resolving_power)
    dbeta_dlam = grating.order / (grating.d_angstrom * np.cos(beta))
    return np.abs(slope * dbeta_dlam) * dlam_fwhm / _FWHM_PER_SIGMA


def _gauss_integrals(v, sigma):
    """First and second antiderivatives of the N(0, sigma) CDF at ``v``.

    ``A(v) = v Phi(v/s) + s phi(v/s)`` and
    ``B(v) = (v^2 + s^2)/2 Phi(v/s) + v s/2 phi(v/s)``; for ``s -> 0`` they
    become ``max(v, 0)`` and ``max(v, 0)^2 / 2`` (the step CDF).
    """
    from scipy.special import ndtr

    gauss = sigma > _ZERO_WIDTH_MM
    s = np.where(gauss, sigma, 1.0)
    z = v / s
    cdf, pdf = ndtr(z), np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    first = np.where(gauss, v * cdf + s * pdf, np.maximum(v, 0.0))
    second = np.where(
        gauss, 0.5 * (v * v + s * s) * cdf + 0.5 * v * s * pdf, 0.5 * np.maximum(v, 0.0) ** 2
    )
    return first, second


def _kernel_cdf(u, width, pitch, sigma):
    """CDF at offset ``u`` of rect(width) * rect(pitch) * N(0, sigma).

    The source pixel's content is uniform over its own ``pitch``; the slit
    image adds a rect of ``width`` and the aberration a Gaussian. With ``B``
    the second antiderivative of the Gaussian CDF,
    ``F(u) = [B(u+(w+p)/2) - B(u+(w-p)/2) - B(u-(w-p)/2) + B(u-(w+p)/2)] / (w p)``;
    ``w -> 0`` reduces to ``[A(u+p/2) - A(u-p/2)] / p``, which with
    ``sigma -> 0`` is the source pixel itself.
    """
    rect = width > _ZERO_WIDTH_MM
    w = np.where(rect, width, 1.0)

    def B(v):
        return _gauss_integrals(v, sigma)[1]

    def A(v):
        return _gauss_integrals(v, sigma)[0]

    # The kernel is symmetric, F(u) = 1 - F(-u): evaluate at -|u| only, where
    # every antiderivative is small, avoiding cancellation far from the line.
    v = -np.abs(u)
    half_sum, half_diff = 0.5 * (w + pitch), 0.5 * (w - pitch)
    wide = (B(v + half_sum) - B(v + half_diff) - B(v - half_diff) + B(v - half_sum)) / (w * pitch)
    narrow = (A(v + 0.5 * pitch) - A(v - 0.5 * pitch)) / pitch
    lower = np.where(rect, wide, narrow)
    return np.where(u > 0.0, 1.0 - lower, lower)


def _column_kernel(E, grating, slit, distance_mm, beta_ref_rad, centers_mm):
    """Slit-image width and aberration sigma [mm] at each pixel column."""
    beta = grating.diffraction_angle_rad(E)
    x, slope = _focal_plane(grating, beta, distance_mm, beta_ref_rad)
    dbeta = slit.width_mm * np.cos(grating.alpha_rad) / (slit.arm_mm(grating) * np.cos(beta))
    width = np.abs(slope * dbeta)
    sigma = _aberration_sigma_mm(E, beta, slope, grating, slit.resolving_power)
    return _at_columns(x, (width, sigma), centers_mm)


def _at_columns(x, values, centers_mm):
    """Interpolate node quantities, given at focal-plane positions ``x``, to columns."""
    ok = np.isfinite(x)
    order = np.argsort(x[ok])
    nodes = x[ok][order]
    return tuple(np.interp(centers_mm, nodes, value[ok][order]) for value in values)


def slit_image_pixels(
    E_grid_eV,
    spec,
    grating,
    ccd,
    slit,
    *,
    distance_mm=None,
    beta_ref_rad=None,
    center_mm=None,
    weight_by_throughput=True,
    chunk=64,
):
    """Dispersed spectrum convolved with the slit-image kernel, per pixel.

    The point-slit image is first formed exactly as :func:`detected_image`
    (:func:`disperse_spectrum` then :func:`bin_to_pixels`). Each pixel's flux,
    taken uniform across that pixel, is then spread by a rect of the slit
    image width (:func:`slit_image_width_mm`, evaluated at the pixel's
    focal-plane position) convolved with the optional :class:`SlitImage`
    aberration Gaussian, and integrated exactly over every pixel. The kernel
    is normalized, so flux is conserved apart from light spread off the
    sensor ends. With zero width and no aberration the result is the
    point-slit image exactly.

    Parameters
    ----------
    E_grid_eV, spec
        Ascending energy grid [eV] and spectral density per eV.
    grating
        :class:`Grating`; its mount supplies the focal plane when
        ``distance_mm`` is ``None``.
    ccd
        :class:`SimpleCCD` pixel grid along the dispersion axis.
    slit
        :class:`SlitImage`.
    distance_mm, beta_ref_rad
        Optional flat-detector map, as in :func:`disperse_spectrum`.
    center_mm
        Pixel-grid centre on the focal plane [mm]; ``None`` centres on the
        dispersed flux, as :func:`bin_to_pixels` does.
    weight_by_throughput
        Multiply by ``grating.throughput(E)`` first.
    chunk
        Lit source pixels per vectorized block (memory bound only).

    Returns
    -------
    pixel_centers_mm, flux_per_pixel
        Arrays of length ``ccd.n_pix``, in the integral units of ``spec``.

    Notes
    -----
    Validation: grating-slit-image-width
    """
    E = np.asarray(E_grid_eV, float)
    position, density = disperse_spectrum(
        E,
        spec,
        grating,
        distance_mm,
        beta_ref_rad=beta_ref_rad,
        weight_by_throughput=weight_by_throughput,
    )
    centers, flux = bin_to_pixels(position, density, ccd, center_mm=center_mm)
    if slit.width_mm == 0.0 and slit.resolving_power is None:
        return centers, flux
    width, sigma = _column_kernel(E, grating, slit, distance_mm, beta_ref_rad, centers)
    edges = ccd.pixel_edges_mm(float(np.mean(centers)))
    reach = 0.5 * (width + ccd.pixel_mm) + 8.0 * sigma  # kernel support, Gaussian tails
    lit = np.flatnonzero(flux)
    out = np.zeros(ccd.n_pix)
    for start in range(0, lit.size, chunk):
        part = lit[start : start + chunk]
        low = np.searchsorted(edges, np.min(centers[part] - reach[part]), side="right") - 1
        high = np.searchsorted(edges, np.max(centers[part] + reach[part]), side="left") + 1
        low, high = max(low, 0), min(high, edges.size)
        offsets = edges[None, low:high] - centers[part, None]
        cdf = _kernel_cdf(offsets, width[part, None], ccd.pixel_mm, sigma[part, None])
        out[low : high - 1] += flux[part] @ np.diff(cdf, axis=1)
    return centers, out


def _row_profiles(row_edges_mm, heights_mm):
    """Normalized (rows, columns) overlap of centred uniform spreads with rows."""
    half = 0.5 * np.asarray(heights_mm, float)[None, :]
    low, high = row_edges_mm[:-1, None], row_edges_mm[1:, None]
    overlap = np.clip(np.minimum(high, half) - np.maximum(low, -half), 0.0, None)
    total = overlap.sum(axis=0, keepdims=True)
    return np.divide(overlap, total, out=np.zeros_like(overlap), where=total > 0)


def detected_image_2d(
    E_grid_eV,
    spec,
    grating,
    sensor,
    slit,
    *,
    along_slit=None,
    distance_mm=None,
    beta_ref_rad=None,
    center_mm=None,
    weight_by_throughput=True,
):
    """Two-dimensional (rows x columns) focal-plane image through a slit.

    Columns follow :func:`slit_image_pixels` for the captured spectrum
    ``spec * throughput(E) * capture(E)``. Each column's flux is spread over
    the rows as a centred uniform band of the effective height
    (:class:`AlongSlitSpread`) at that column's focal-plane position, clipped
    to the sensor and renormalized so that every column's row sum equals its
    captured flux. Hence ``image.sum(axis=0)`` is the 1D slit image of the
    captured spectrum; for a constant capture ``f`` it is ``f`` times the 1D
    image. Without ``along_slit`` all flux is captured and spread uniformly
    over the rows.

    Parameters
    ----------
    sensor
        :class:`~pyrite.detectors.alexs_response.AlexsFormat` (columns along
        dispersion, rows across, square pitch).
    slit
        :class:`SlitImage`.
    along_slit
        Optional :class:`AlongSlitSpread`.

    Other parameters are as in :func:`slit_image_pixels`.

    Returns
    -------
    column_centers_mm, row_centers_mm, image
        ``image`` has shape ``(sensor.rows, sensor.columns)`` in the integral
        units of ``spec``.
    """
    E = np.asarray(E_grid_eV, float)
    spec = np.asarray(spec, float)
    if weight_by_throughput:
        spec = spec * grating.throughput(E)
    height = sensor.height_mm
    ccd = SimpleCCD(n_pix=sensor.columns, pixel_mm=sensor.pitch_mm)
    captured = spec if along_slit is None else spec * along_slit.capture(E, height)
    columns, flux = slit_image_pixels(
        E,
        captured,
        grating,
        ccd,
        slit,
        distance_mm=distance_mm,
        beta_ref_rad=beta_ref_rad,
        center_mm=center_mm,
        weight_by_throughput=False,
    )
    column_height = np.full(columns.shape, height)
    if along_slit is not None:
        beta = grating.diffraction_angle_rad(E)
        x, _ = _focal_plane(grating, beta, distance_mm, beta_ref_rad)
        spread = along_slit.effective_height_mm(E, height)
        (column_height,) = _at_columns(x, (spread,), columns)
    row_edges = SimpleCCD(n_pix=sensor.rows, pixel_mm=sensor.pitch_mm).pixel_edges_mm()
    image = _row_profiles(row_edges, column_height) * flux[None, :]
    return columns, 0.5 * (row_edges[:-1] + row_edges[1:]), image


@dataclass(frozen=True)
class SpectrometerImage:
    """One ALEX-s frame of a grating spectrometer, in ADU.

    ``adu`` has shape ``(rows, columns)``; ``column_mm``/``row_mm`` are pixel
    centres on the focal plane. ``expected_photons`` and ``expected_electrons``
    are the noise-free absorbed-photon and signal-charge images.
    """

    column_mm: np.ndarray
    row_mm: np.ndarray
    adu: np.ndarray
    expected_photons: np.ndarray
    expected_electrons: np.ndarray

    def row_summed(self) -> np.ndarray:
        """The 1D dispersion-axis view: ADU summed over sensor rows."""
        return self.adu.sum(axis=0)


def alexs_spectrometer_image(
    E_grid_eV,
    photons_per_eV,
    grating,
    response,
    slit,
    *,
    exposure_s,
    sensor=None,
    along_slit=None,
    distance_mm=None,
    center_mm=None,
    rng=None,
    shot_noise=False,
    read_noise=False,
    dark=True,
):
    """Forward-model spectrum -> slit -> grating -> ALEX-s frame in ADU.

    ``photons_per_eV`` is the photon spectrum entering the slit during the
    exposure (photons per eV): a scored source spectrum already multiplied by
    the slit acceptance, beam rate and exposure. The grating throughput and
    the ALEX-s QE weight the photons; the signal charge per photon is
    ``E / 3.66 eV``; :func:`~pyrite.detectors.alexs_response.alexs_frame`
    then adds dark charge, clips at full well, optionally draws Poisson and
    read noise with the explicit ``rng``, and applies the mode gain.

    ``response`` is an :class:`~pyrite.detectors.AlexS`; ``sensor`` defaults to
    the 2k512 format and must match ``response.sensor_format``.
    """
    from .alexs_response import AlexsFormat, alexs_frame, quantum_efficiency, signal_electrons

    sensor = AlexsFormat.named("2k512") if sensor is None else sensor
    if (sensor.columns, sensor.rows) != tuple(response.sensor_format):
        raise ValueError("sensor format does not match response.sensor_format")
    E = np.asarray(E_grid_eV, float)
    absorbed = np.asarray(photons_per_eV, float) * quantum_efficiency(E, response.qe)
    kwargs = dict(along_slit=along_slit, distance_mm=distance_mm)
    columns, rows, photons = detected_image_2d(
        E, absorbed, grating, sensor, slit, center_mm=center_mm, **kwargs
    )
    electrons = detected_image_2d(
        E,
        signal_electrons(E, absorbed, 1.0),
        grating,
        sensor,
        slit,
        center_mm=float(np.mean(columns)),
        **kwargs,
    )[2]
    adu = alexs_frame(
        electrons,
        exposure_s=exposure_s,
        readout_rate_hz=response.readout_rate_hz,
        gain_mode=response.gain_mode,
        photons=photons,
        rng=rng,
        shot_noise=shot_noise,
        read_noise=read_noise,
        dark=dark,
    )
    return SpectrometerImage(columns, rows, adu, photons, electrons)


__all__ = [
    "AlongSlitSpread",
    "SlitImage",
    "SpectrometerImage",
    "alexs_spectrometer_image",
    "detected_image_2d",
    "slit_image_pixels",
    "slit_image_width_mm",
]
