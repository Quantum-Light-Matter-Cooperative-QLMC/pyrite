"""
grating.py

EXPLORATORY forward model for a grazing-incidence soft X-ray diffraction grating
spectrometer (TODO P3 #8): disperse the model's PXR+CBS line spectrum across a
position-sensitive detector (Eagle XO CCD or similar) so a *spatial* image can be
compared to experiment, instead of an energy spectrum at a fixed take-off angle.

Physics -- the reflection grating equation, normal-referenced:

    d (sin alpha + sin beta) = m lambda                      (sin beta = m lambda/d - sin alpha)

with alpha the incidence angle and beta the diffraction angle, both from the
grating NORMAL, d = 1/(groove density) the groove spacing, m the order. m = 0 is
specular (beta = -alpha). Soft X-rays need GRAZING incidence (grazing angle
theta_g = 90 deg - alpha of a few degrees) for usable reflectivity; that is a
property of `alpha`, not a different equation.

This module is standalone (nothing in the sweep/plot pipeline imports it yet);
see docs/grazing-grating.md for the modality and the phased plan. Grazing-
incidence Fresnel reflectivity of the coating is modelled (`Grating.reflectivity`);
groove-profile diffraction efficiency is NOT -- `Grating.groove_efficiency` is a
placeholder scalar (a rigorous scalar/RCWA treatment is future work, see
docs/grazing-grating.md).

`SimpleCCD` + `bin_to_pixels` (phased-plan step 3) rebin a dispersed profile
onto a fixed pixel grid sized to the greateyes ALEX-s 1k256/2k512 formats --
geometry only (no QE, charge-sharing, or energy resolution yet).

`detected_image` (phased-plan step 4) is the single combined forward-model
entry: `mc_spectrum` output + a `Grating` + a `SimpleCCD` -> the dispersed,
detected image (counts vs pixel), chaining `disperse_spectrum` and
`bin_to_pixels` so it slots in beside the Timepix / Eagle XO detector models.

`qe_absorption`, `charge_cloud_sigma_um`, `energy_fwhm_eV`, and
`detected_image_physical` (phased-plan step 5) upgrade the geometry-only
`SimpleCCD`/`bin_to_pixels` with closed-form detector physics -- Beer-Lambert
absorption QE, Einstein-relation drift-diffusion charge sharing, Fano +
read-noise energy resolution -- the same ladder `eaglexo_response.py` climbed
for the Eagle XO. No public greateyes ALEX-s datasheet QE/noise curve exists
(unlike the Eagle XO's digitized `eaglexo_qe.csv`), so these are physics
formulas with device-specific constants (active thickness, depletion voltage,
temperature, read noise) flagged '### FILL IN' pending a real datasheet.
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from ._si_sensor import FANO_SI, SI_N_PER_ANG3, W_EHP_EV
from .crystallography import (
    HC_EV_ANG,
    absorption_length_ang,
    optical_constants,
)  # h*c [eV*Angstrom]

# ---- coating optical constants (grazing-incidence reflectivity) --------------
# Atomic number density n = rho/A * N_A, converted cm^-3 -> Ang^-3 (the unit
# crystallography.optical_constants wants) -- same pattern as
# _si_sensor.SI_N_PER_ANG3 (0.602214076 = N_A * 1e-24). Densities and molar
# masses are standard elemental values (CRC Handbook of Chemistry and Physics /
# IUPAC standard atomic weights) for the coatings common on SXR grazing-
# incidence gratings and mirrors.
_COATING_DENSITY_G_CM3 = {
    "Au": 19.30,  # gold: most common SXR grazing-incidence grating/mirror coating
    "Pt": 21.45,  # platinum: common alternative, higher-Z
    "Ni": 8.908,  # nickel: common alternative, cheaper
}
_COATING_MOLAR_MASS_G_MOL = {
    "Au": 196.96657,
    "Pt": 195.084,
    "Ni": 58.6934,
}


def coating_number_density_per_ang3(element):
    """Atomic number density [1/Angstrom^3] of a coating element (rho/A * N_A,
    cm^-3 -> Ang^-3), for the elements in `_COATING_DENSITY_G_CM3`."""
    if element not in _COATING_DENSITY_G_CM3:
        raise KeyError(
            f"No coating density/molar-mass data for '{element}'. "
            f"Known coatings: {sorted(_COATING_DENSITY_G_CM3)}."
        )
    return _COATING_DENSITY_G_CM3[element] / _COATING_MOLAR_MASS_G_MOL[element] * 0.602214076


def wavelength_angstrom(E_eV):
    """Photon wavelength [Angstrom] from energy [eV] (lambda = hc / E)."""
    return HC_EV_ANG / np.asarray(E_eV, float)


def groove_spacing_angstrom(groove_density_per_mm):
    """Groove spacing d [Angstrom] from line density [lines/mm] (1 mm = 1e7 A)."""
    return 1.0e7 / float(groove_density_per_mm)


@dataclass(frozen=True)
class Grating:
    """A reflection grating in a fixed mount.

    groove_density_per_mm : ruling density (e.g. 1200 lines/mm).
    alpha_rad : incidence angle from the grating NORMAL (grazing => near pi/2).
    order : diffraction order m (1 typical; 0 is specular).
    coating : reflective coating element symbol, one of `_COATING_DENSITY_G_CM3`
        (default "Au"). Confirmed for the McPherson 251MX's four gratings
        (120/300/1200/2400 g/mm all gold-coated per McPherson's own product
        materials, see docs/grazing-grating.md "Hardware targets"); the exact
        groove *profile* (laminar vs blazed) feeding a future groove-efficiency
        model is still McPherson-family-typical, not 251MX-confirmed.
    groove_efficiency : placeholder scalar diffraction efficiency in [0, 1],
        NOT a real groove-profile efficiency model (that needs a scalar or
        rigorous-coupled-wave (RCWA) treatment, out of scope here -- see
        docs/grazing-grating.md "What is NOT modelled yet"). Default 1.0
        (i.e. no groove-efficiency penalty applied) is itself the placeholder.
        ### FILL IN once a groove-profile model or measured efficiency curve
        lands.
    """

    groove_density_per_mm: float
    alpha_rad: float
    order: int = 1
    coating: str = "Au"
    groove_efficiency: float = 1.0

    @property
    def d_angstrom(self) -> float:
        return groove_spacing_angstrom(self.groove_density_per_mm)

    @property
    def grazing_angle_rad(self) -> float:
        """Incidence grazing angle theta_g = 90 deg - alpha."""
        return 0.5 * np.pi - self.alpha_rad

    def diffraction_angle_rad(self, E_eV):
        """Diffraction angle beta [rad] from the grating normal for energy E_eV
        (array-safe). sin beta = m lambda/d - sin alpha; energies that would need
        |sin beta| > 1 (no propagating order) return NaN."""
        lam = wavelength_angstrom(E_eV)
        s = self.order * lam / self.d_angstrom - np.sin(self.alpha_rad)
        s = np.where(np.abs(s) <= 1.0, s, np.nan)
        return np.arcsin(s)

    def angular_dispersion_rad_per_angstrom(self, E_eV):
        """d(beta)/d(lambda) = m / (d cos beta) [rad/Angstrom]; the grating's
        intrinsic dispersion before any detector geometry."""
        beta = self.diffraction_angle_rad(E_eV)
        return self.order / (self.d_angstrom * np.cos(beta))

    def reflectivity(self, E_eV):
        """Grazing-incidence Fresnel reflectivity R(E) of the coating (bare-film;
        groove-profile diffraction efficiency is NOT included, see `throughput`).

        Complex refractive index n = 1 - delta - i*beta of `coating`
        (`crystallography.optical_constants`, Henke/Chantler f1 = Z+f', f2). At
        the incidence grazing angle theta = `grazing_angle_rad` (measured from
        the surface; theta, delta, beta all << 1), the vacuum/medium Fresnel
        amplitude reflectivity r = (k_z1 - k_z2)/(k_z1 + k_z2), with
        k_z1 = k0 sin(theta) ~ k0 theta and
        k_z2 = k0 sqrt(n^2 - cos^2 theta) ~ k0 sqrt(theta^2 - 2 delta - 2i beta)
        (using n^2 - 1 ~ -2 delta - 2i beta and cos^2 theta ~ 1 - theta^2 for
        theta << 1 rad), reduces to the standard small-angle form:

            r(theta) = (theta - sqrt(theta^2 - 2 delta - 2i beta))
                       / (theta + sqrt(theta^2 - 2 delta - 2i beta))
            R(theta) = |r(theta)|^2

        Source: Als-Nielsen & McMorrow, "Elements of Modern X-ray Physics" 2nd
        ed., Ch. 3 (refraction and reflection at an interface); equivalently
        Attwood & Sakdinawat, "X-Rays and Extreme Ultraviolet Radiation" 2nd
        ed., Ch. 3. ASSUMES grazing incidence, where the s- and p-polarization
        reflectivities coincide to good approximation (the polarization factor
        -> cos(2 theta) -> 1 as theta -> 0), so R is treated here as
        polarization-independent -- standard practice for grazing-incidence
        SXR optics, stated explicitly since it is an approximation.

        Limiting cases (beta -> 0 idealization, theta_c = sqrt(2 delta)):
          - theta << theta_c: the sqrt argument is negative, so r is a pure
            phase, |r| = 1 -> total external reflection, R -> 1.
          - theta >> theta_c: R -> (theta_c / (2 theta))^4 (Als-Nielsen &
            McMorrow's steep power-law falloff above the critical angle).

        Validation: grazing-reflectivity
        """
        n_per_ang3 = coating_number_density_per_ang3(self.coating)
        delta, beta = optical_constants(self.coating, E_eV, n_per_ang3)
        theta = self.grazing_angle_rad
        inside = theta**2 - 2.0 * delta - 2.0j * beta
        root = np.sqrt(inside)
        r = (theta - root) / (theta + root)
        return np.abs(r) ** 2

    def throughput(self, E_eV):
        """Combined grating throughput: Fresnel `reflectivity(E)` times the
        placeholder `groove_efficiency` scalar (NOT a real groove-profile
        efficiency model -- see that field's docstring)."""
        return self.reflectivity(E_eV) * self.groove_efficiency


def detector_position_mm(beta_rad, beta_ref_rad, distance_mm):
    """Where a ray diffracted at beta lands on a flat detector a distance
    ``distance_mm`` from the grating, whose face is perpendicular to the
    reference direction ``beta_ref_rad``: x = L tan(beta - beta_ref) [mm]."""
    return distance_mm * np.tan(np.asarray(beta_rad, float) - beta_ref_rad)


def disperse_spectrum(
    E_grid_eV, spec, grating, distance_mm, *, beta_ref_rad=None, weight_by_throughput=False
):
    """Map an energy spectrum onto detector positions through ``grating``.

    Returns ``(position_mm, intensity_per_mm)`` for the input energies whose order
    propagates (NaN-beta energies dropped). Flux-conserving: the per-eV spectrum
    is reweighted by the Jacobian ``|dE/dx|`` so that integral(I dx) == integral(spec dE).
    ``beta_ref_rad`` (the detector-normal direction) defaults to the mean
    diffraction angle, centring the dispersed band on the detector.

    ``weight_by_throughput`` (default False -- preserves the existing purely-
    geometric behavior): when True, ``spec`` is first multiplied by
    ``grating.throughput(E)`` (Fresnel reflectivity x groove_efficiency, see
    ``Grating.throughput``) before the Jacobian remap below, so the returned
    intensity is a throughput-weighted (still flux-conserving, now w.r.t. the
    *weighted* input) profile rather than a purely geometric one. This only
    rescales the input spectrum; the Jacobian math itself is untouched.
    """
    E = np.asarray(E_grid_eV, float)
    spec = np.asarray(spec, float)
    if weight_by_throughput:
        spec = spec * grating.throughput(E)
    beta = grating.diffraction_angle_rad(E)
    if beta_ref_rad is None:
        beta_ref_rad = float(np.nanmean(beta))
    x = detector_position_mm(beta, beta_ref_rad, distance_mm)
    ok = np.isfinite(x)
    E, spec, x = E[ok], spec[ok], x[ok]
    if E.size < 2:
        return x, np.zeros_like(x)
    dxdE = np.gradient(x, E)
    inten = np.where(np.abs(dxdE) > 0, spec / np.abs(dxdE), 0.0)
    inten[~np.isfinite(inten)] = 0.0
    return x, inten


# ---- simple CCD: pixel binning of the dispersed profile (phased-plan step 3) --
# greateyes ALEX-s, two interchangeable formats (docs/grazing-grating.md
# "Hardware targets"). n_pix / pixel_um are along the DISPERSION axis (the
# sensor's long axis); active_mm is (dispersion, cross-dispersion) extent,
# included for reference only (this simple model is 1-D, along dispersion).
class AlexsSensor(TypedDict):
    """One greateyes ALEX-s sensor format. Typing the registry with this (rather
    than ``dict[str, Any]``) lets pyright keep ``n_pix`` an ``int`` and
    ``pixel_um`` a ``float`` through the ``SimpleCCD.from_alexs`` spread, the
    same way :class:`config.MaterialGrid` types ``_MATERIAL_GRIDS``."""

    n_pix: int
    pixel_um: float
    active_mm: tuple[float, float]


ALEXS_SENSORS: dict[str, AlexsSensor] = {
    "1k256": AlexsSensor(n_pix=1024, pixel_um=26.0, active_mm=(26.6, 6.7)),
    "2k512": AlexsSensor(n_pix=2048, pixel_um=13.5, active_mm=(27.6, 6.9)),
}


@dataclass(frozen=True)
class SimpleCCD:
    """A geometry-only CCD pixel grid along the grating's dispersion axis.

    n_pix : number of pixels along the dispersion direction.
    pixel_mm : pixel pitch along the dispersion direction [mm].

    Deliberately crude (phased-plan step 3, docs/grazing-grating.md): a fixed
    array of equal-width bins with no QE, charge-sharing, or energy-resolution
    structure -- that comes later (step 5). Use `bin_to_pixels` to rebin a
    `disperse_spectrum` profile onto this grid.
    """

    n_pix: int
    pixel_mm: float

    @classmethod
    def from_alexs(cls, variant):
        """Build from a named greateyes ALEX-s format ("1k256" or "2k512")."""
        if variant not in ALEXS_SENSORS:
            raise KeyError(f"unknown ALEX-s variant {variant!r} (have {list(ALEXS_SENSORS)})")
        s = ALEXS_SENSORS[variant]
        return cls(n_pix=s["n_pix"], pixel_mm=s["pixel_um"] * 1.0e-3)

    @property
    def width_mm(self) -> float:
        return self.n_pix * self.pixel_mm

    def pixel_edges_mm(self, center_mm=0.0):
        """The `n_pix + 1` pixel boundary positions [mm], centered on ``center_mm``."""
        half = 0.5 * self.width_mm
        return float(center_mm) + np.linspace(-half, half, self.n_pix + 1)

    def pixel_centers_mm(self, center_mm=0.0):
        edges = self.pixel_edges_mm(center_mm)
        return 0.5 * (edges[:-1] + edges[1:])


def bin_to_pixels(position_mm, intensity_per_mm, ccd, center_mm=None):
    """Rebin a dispersed ``(position_mm, intensity_per_mm)`` profile (e.g. from
    `disperse_spectrum`) onto ``ccd``'s fixed pixel grid.

    Each pixel's value is the integral of ``intensity_per_mm`` across the
    physical span (equivalently the polar-angle span, via the grating's flat-
    detector map) that pixel subtends -- no QE or charge-sharing yet (that is
    step 5; see docs/grazing-grating.md "What is NOT modelled yet"). The
    integral is done on the cumulative-trapezoid of the (sorted) input curve,
    interpolated at the pixel edges, so it is exact for a piecewise-linear
    input and needs no assumption that ``position_mm`` already lies on a
    uniform grid. Light landing outside the sensor's physical extent is
    dropped, like any real finite detector -- so
    ``sum(flux_per_pixel) <= trapz(intensity_per_mm, position_mm)``, with
    equality when the whole dispersed profile fits within the sensor.

    ``center_mm`` (default: the flux-weighted centroid of the input profile)
    positions the fixed pixel grid relative to the dispersed light.

    Returns ``(pixel_centers_mm, flux_per_pixel)``, both length ``ccd.n_pix``.
    """
    x = np.asarray(position_mm, dtype=float)
    inten = np.asarray(intensity_per_mm, dtype=float)
    order = np.argsort(x)
    x, inten = x[order], inten[order]
    if center_mm is None:
        total = np.trapezoid(inten, x)
        center_mm = float(np.trapezoid(inten * x, x) / total) if total > 0 else 0.0
    edges = ccd.pixel_edges_mm(center_mm)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inten[1:] + inten[:-1]) * np.diff(x))])
    cum_at_edges = np.interp(edges, x, cum, left=cum[0], right=cum[-1])
    flux_per_pixel = np.diff(cum_at_edges)
    return ccd.pixel_centers_mm(center_mm), flux_per_pixel


def detected_image(
    E_grid_eV,
    spec,
    grating,
    ccd,
    distance_mm,
    *,
    weight_by_throughput=True,
    beta_ref_rad=None,
    center_mm=None,
):
    """Combined forward-model entry (phased-plan step 4): the model's `mc_spectrum`
    (or `mc_spectrum_solid_angle`) output on ``E_grid_eV`` + a ``grating`` + a
    ``ccd`` (`SimpleCCD`) -> the dispersed, DETECTED image (counts vs pixel), by
    chaining `disperse_spectrum` then `bin_to_pixels`. So it slots in beside the
    Timepix / Eagle XO detector forward models.

    Pure composition of the two already-validated pieces above -- no new physics
    equation and so no new ledger row/validator pass, same precedent as
    `SimpleCCD`/`bin_to_pixels` themselves being pure geometry (see
    docs/grazing-grating.md phased-plan step 3's note on this).

    ``weight_by_throughput`` defaults to True HERE (unlike `disperse_spectrum`'s
    own default of False, kept for backward compatibility there): this function
    IS the detected-image entry point, so a real CCD frame should reflect the
    grating's Fresnel throughput (`Grating.throughput`), not the purely
    geometric dispersion. Pass False to get the geometry-only image instead.

    ``beta_ref_rad``/``center_mm`` forward to `disperse_spectrum`/`bin_to_pixels`
    respectively (both default to auto-centering on the dispersed light).

    Returns ``(pixel_centers_mm, counts_per_pixel)``, both length ``ccd.n_pix`` --
    same contract as `bin_to_pixels`.
    """
    x, inten = disperse_spectrum(
        E_grid_eV,
        spec,
        grating,
        distance_mm,
        beta_ref_rad=beta_ref_rad,
        weight_by_throughput=weight_by_throughput,
    )
    return bin_to_pixels(x, inten, ccd, center_mm=center_mm)


def resolving_power(E_eV, grating, distance_mm, pixel_mm, beta_ref_rad=None):
    """Pixel-limited resolving power lambda/dlambda at E_eV: one detector pixel
    subtends dlambda = pixel_mm / (dx/dlambda), with the linear dispersion
    dx/dlambda = distance * d(beta)/d(lambda) / cos^2(beta - beta_ref)."""
    beta = grating.diffraction_angle_rad(E_eV)
    if beta_ref_rad is None:
        beta_ref_rad = beta
    dbeta_dlam = grating.angular_dispersion_rad_per_angstrom(E_eV)
    dx_dlam = distance_mm * dbeta_dlam / np.cos(beta - beta_ref_rad) ** 2  # mm/Angstrom
    dlam = pixel_mm / dx_dlam  # Angstrom per pixel
    lam = wavelength_angstrom(E_eV)
    return lam / dlam


# ---- physical CCD: QE, charge sharing, energy resolution (phased-plan step 5) --
# Upgrades SimpleCCD's geometry-only pixel binning with real detector physics,
# the same "geometry -> QE(E) -> charge diffusion" ladder eaglexo_response.py
# climbed for the Eagle XO. No digitized greateyes ALEX-s datasheet QE/noise
# curve is publicly available (unlike the Eagle XO's eaglexo_qe.csv), so QE and
# charge diffusion below are both closed-form physics models (Beer-Lambert
# absorption efficiency; Einstein-relation drift-diffusion) with the
# DEVICE-SPECIFIC numeric constants (active thickness, depletion voltage,
# temperature, read noise) flagged '### FILL IN' pending a real greateyes
# datasheet/measurement -- same pattern as eaglexo_response.py's
# DEFAULT_DISTANCE_M / ACTIVE_SI_UM.
ACTIVE_SI_UM = 30.0  ### FILL IN -- active (depleted) Si thickness [um] of the
#   greateyes ALEX-s sensor; no public datasheet value found. Deep-depletion
#   back-illuminated CCDs built for SXR/EUV typically run ~30-50 um; this is a
#   mid-range placeholder, NOT a measured/confirmed ALEX-s number.
DEPLETION_VOLTAGE_V = 40.0  ### FILL IN -- full-depletion bias [V]; a typical
#   deep-depletion scientific-CCD range (~20-50 V), not ALEX-s-confirmed.
OPERATING_TEMP_C = -60.0  ### FILL IN -- greateyes markets ALEX-s as
#   "deep-cooled"; -60 C is a representative TE-cooled operating point for
#   that product family, not a confirmed ALEX-s spec.
READ_NOISE_E = 3.0  ### FILL IN -- read noise [e- RMS] at slow readout; no
#   public ALEX-s figure found, set close to the Eagle XO's 2.3 e-
#   (eaglexo_response.READ_NOISE_E) as a representative deep-cooled-CCD value.
ENTRANCE_QE_PEAK = 0.9  ### FILL IN -- entrance-surface / dead-layer loss
#   factor capping the Beer-Lambert QE below 1; representative back-
#   illuminated-CCD value (cf. eaglexo_response.qe_absorption_model's
#   peak=0.93 default), not measured for ALEX-s.

_K_B_EV_PER_K = 8.617333262e-5  # Boltzmann constant [eV/K] (== [V/K], 1 eV/e- = 1 V)


def qe_absorption(E_eV, active_um=None, peak=None):
    """CCD quantum efficiency from pure Beer-Lambert absorption in the active
    silicon:

        QE(E) = peak * (1 - exp(-t / L_abs(E)))

    t = ``active_um`` (default `ACTIVE_SI_UM`), L_abs from Henke f2
    (`crystallography.absorption_length_ang`, the same absorption length used
    throughout this repo -- see the `absorption-length` ledger entry).

    Derivation: a photon normally incident on a slab of thickness t is
    absorbed within it with probability ``1 - exp(-t/L_abs)`` (Beer-Lambert).
    ``peak`` < 1 folds in the fixed entrance-surface / dead-layer loss a real
    back-illuminated CCD has on top of bulk absorption (not modelled here from
    first principles -- see `ENTRANCE_QE_PEAK`). This is the SAME functional
    form as `eaglexo_response.qe_absorption_model` (a cross-check there, since
    a measured datasheet curve is primary for the Eagle XO); here it IS
    primary, since no greateyes ALEX-s QE datasheet is publicly available --
    see the module docstring and `ACTIVE_SI_UM`.

    Limiting cases:
      - t << L_abs(E) (hard photon / thin sensor): QE -> peak * t/L_abs(E),
        the optically-thin linear regime.
      - t >> L_abs(E) (soft photon / thick sensor): QE -> peak, full capture.

    Validation: alexs-qe-absorption
    """
    t_um = ACTIVE_SI_UM if active_um is None else active_um
    p = ENTRANCE_QE_PEAK if peak is None else peak
    E = np.asarray(E_eV, dtype=float)
    L_ang = absorption_length_ang("Si", np.clip(E, 1e-3, None), SI_N_PER_ANG3)
    out = p * (1.0 - np.exp(-(t_um * 1.0e4) / L_ang))  # um -> Angstrom
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def charge_cloud_sigma_um(E_eV, active_um=None, v_dep=None, temp_c=None):
    """Lateral charge-cloud RMS spread [um] from diffusion during drift to the
    front-side electrodes, for a photon absorbed at depth z(E) in a back-
    illuminated, fully-depleted sensor of active thickness t under bias
    ``v_dep`` (uniform field ``E_field = v_dep/t``) at temperature ``temp_c``.

    Derivation: the drifting charge cloud diffuses while it drifts a distance
    ``d = t - z`` to the front (collecting) surface. With the Einstein
    relation ``D = mu*kT/q`` and drift velocity ``v_d = mu*E_field``, the
    drift time is ``t_drift = d/v_d``, so:

        sigma^2 = 2 D t_drift = 2 (kT/q) d / E_field = 2 (kT/q) t d / v_dep

    -- mobility mu cancels, leaving only the thermal voltage kT/q, the
    geometry (t, d), and the bias v_dep. ``z(E)`` is approximated by the
    characteristic absorption depth ``min(L_abs(E), t)`` (exact only in the
    ``L_abs << t`` limit for the true exponential absorption profile; a
    reasonable single-number stand-in otherwise -- a full depth-resolved
    treatment is future work). Source: Einstein relation + drift-diffusion,
    the standard semiconductor-detector treatment of charge-cloud spreading in
    back-illuminated CCDs (e.g. Janesick, "Scientific Charge-Coupled Devices",
    SPIE 2001, Ch. 4).

    Limiting cases:
      - Soft photon (L_abs -> 0): absorbed right at the back (entrance)
        surface, drifts the FULL thickness t -> sigma -> t*sqrt(2 kT/(q*v_dep)),
        the maximum blur -- matches the well-known result that back-
        illuminated CCDs blur soft X-rays MORE than hard ones.
      - Hard photon (L_abs >= t): absorbed at the front (z=t) -> zero drift
        distance -> sigma -> 0 (though QE is also falling here, see
        `qe_absorption`).

    Validation: alexs-charge-diffusion
    """
    t_um = ACTIVE_SI_UM if active_um is None else active_um
    v = DEPLETION_VOLTAGE_V if v_dep is None else v_dep
    T_c = OPERATING_TEMP_C if temp_c is None else temp_c
    kT_over_q = _K_B_EV_PER_K * (T_c + 273.15)  # thermal voltage [V] (1 eV/e- == 1 V)
    E = np.asarray(E_eV, dtype=float)
    L_ang = absorption_length_ang("Si", np.clip(E, 1e-3, None), SI_N_PER_ANG3)
    L_um = L_ang * 1.0e-4
    z_um = np.minimum(L_um, t_um)
    drift_um = t_um - z_um
    sigma_um = np.sqrt(2.0 * kT_over_q * t_um * drift_um / v)
    return np.nan_to_num(sigma_um, nan=0.0, posinf=0.0, neginf=0.0)


def energy_fwhm_eV(E_eV, n_pix=4, read_noise_e=None):
    """Single-photon energy-measurement FWHM [eV] in photon-counting mode:
    Fano statistics on the e-h pair count plus read noise over ``n_pix``
    pixels the charge cloud covers, in quadrature on the variance -- the
    SAME formula (and role) as `eaglexo_response.energy_fwhm_eV`, restated
    here for the ALEX-s's own W_Si/Fano/read-noise constants:

        sigma_N^2 = F (E / W_Si) + n_pix sigma_read^2   [electrons^2]
        FWHM = 2.3548 * W_Si * sigma_N

    Not applied by default anywhere in this module (the ALEX-s is used here as
    an imaging, not photon-counting, detector) -- exposed for parity with
    `eaglexo_response` and future photon-counting-mode use. No new physics
    equation vs. that existing formula, so no separate ledger row.
    """
    rn = READ_NOISE_E if read_noise_e is None else read_noise_e
    E = np.asarray(E_eV, dtype=float)
    var_e = FANO_SI * (E / W_EHP_EV) + n_pix * rn**2  # [e-^2]
    return 2.3548 * W_EHP_EV * np.sqrt(var_e)  # [eV]


def detected_image_physical(
    E_grid_eV,
    spec,
    grating,
    ccd,
    distance_mm,
    *,
    weight_by_throughput=True,
    beta_ref_rad=None,
    center_mm=None,
    active_um=None,
    v_dep=None,
    temp_c=None,
    peak_qe=None,
):
    """Physical-CCD combined forward-model entry (phased-plan step 5): like
    `detected_image`, but folds in the ALEX-s QE(E) (`qe_absorption`) before
    binning and the charge-cloud diffusion blur (`charge_cloud_sigma_um`)
    after binning, instead of `SimpleCCD`'s unit-efficiency, no-blur geometry.

    Pipeline: ``spec * grating.throughput(E)`` (optional, via
    ``weight_by_throughput``) ``* qe_absorption(E)`` -> `disperse_spectrum`'s
    flux-conserving Jacobian remap -> `bin_to_pixels` -> a Gaussian blur in
    pixel space with sigma set by `charge_cloud_sigma_um` evaluated at the
    FLUX-WEIGHTED MEAN energy of the (QE- and throughput-weighted) input
    spectrum. That is a single characteristic width for the whole frame, not a
    per-photon energy-dependent blur (the latter would need per-photon
    tracking this module doesn't do) -- a reasonable simplification when one
    line/band dominates the frame, less so for a broad multi-line spectrum
    spanning a wide charge-cloud-sigma range. The blur is applied with
    zero-padded (not periodic) edges, so it is flux-conserving except for
    charge diffusing past the sensor's physical edge pixels -- the same
    "real finite detector" behavior `bin_to_pixels` already has for light
    landing outside the sensor outright.

    Returns ``(pixel_centers_mm, counts_per_pixel)``, same contract as
    `detected_image`.
    """
    E = np.asarray(E_grid_eV, dtype=float)
    spec = np.asarray(spec, dtype=float) * qe_absorption(E, active_um=active_um, peak=peak_qe)
    x, inten = disperse_spectrum(
        E,
        spec,
        grating,
        distance_mm,
        beta_ref_rad=beta_ref_rad,
        weight_by_throughput=weight_by_throughput,
    )
    centers, counts = bin_to_pixels(x, inten, ccd, center_mm=center_mm)

    weighted_spec = spec * grating.throughput(E) if weight_by_throughput else spec
    total = np.trapezoid(weighted_spec, E)
    E_mean = float(np.trapezoid(weighted_spec * E, E) / total) if total > 0 else float(np.mean(E))
    sigma_um = float(charge_cloud_sigma_um(E_mean, active_um=active_um, v_dep=v_dep, temp_c=temp_c))
    sigma_px = (sigma_um * 1.0e-3) / ccd.pixel_mm  # um -> mm -> pixels
    if sigma_px > 0:
        from scipy.ndimage import gaussian_filter1d

        counts = gaussian_filter1d(counts, sigma_px, mode="constant", cval=0.0)
    return centers, counts
