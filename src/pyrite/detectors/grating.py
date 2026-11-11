"""
detectors/grating.py

EXPLORATORY forward model for a grazing-incidence soft X-ray diffraction grating
spectrometer (TODO P3 #8): disperse the model's PXR+CBS CXR spectrum across a
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
see docs/research/instrumentation/grazing-grating.md for the modality and the phased plan. Grazing-
incidence Fresnel reflectivity of the coating is modelled (`Grating.reflectivity`);
groove-profile diffraction efficiency is NOT -- `Grating.groove_efficiency` is a
placeholder scalar (a rigorous scalar/RCWA treatment is future work, see
docs/research/instrumentation/grazing-grating.md).

`SimpleCCD` + `bin_to_pixels` (phased-plan step 3) rebin a dispersed profile
onto a fixed pixel grid sized to the greateyes ALEX-s 1k256/2k512 formats --
geometry only (no QE, charge-sharing, or energy resolution yet).

`detected_image` (phased-plan step 4) is the single combined forward-model
entry: `mc_cxr_spectrum` output + a `Grating` + a `SimpleCCD` -> the dispersed,
detected image (counts vs pixel), chaining `disperse_spectrum` and
`bin_to_pixels` so it slots in beside the Timepix / Eagle XO detector models.

`qe_absorption`, `charge_cloud_sigma_um`, `energy_fwhm_eV`, and
`detected_image_physical` (phased-plan step 5) upgrade the geometry-only
`SimpleCCD`/`bin_to_pixels` with closed-form detector physics -- Beer-Lambert
absorption QE, Einstein-relation drift-diffusion charge sharing, Fano +
read-noise energy resolution -- the same ladder `eaglexo_response.py` climbed
for the Eagle XO. Their ALEX-s device constants come from the public greateyes
DS-ALEXs Rev05 datasheet (15 um epitaxial Si, -100 C, 3.5 e- read noise at
50 kHz); the depletion bias and entrance-window QE peak are not in that
datasheet and stay flagged '### FILL IN'. The charge-integrating ALEX-s
response itself lives in `alexs_response.py`.

Flat-field spectrometer (issue #1081): `Grating.efficiency_curve` replaces the
Fresnel x placeholder throughput by a tabulated absolute efficiency,
`FlatFieldMount` maps diffraction angle onto a flat focal plane at distance
r_H' with normal beta_H, and `Grating.from_catalog` builds both from a catalog
`[gratings.NAME]` record. The slit-image kernel and the 2D ALEX-s image live in
`grating_image.py`. No ray tracing.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..materials.crystal import (
    HC_EV_ANG,
    absorption_length_ang,
    optical_constants,
)  # h*c [eV*Angstrom]
from ._si_sensor import FANO_SI, SI_N_PER_ANG3, W_EHP_EV
from ._tabulated import TabulatedEfficiency
from .alexs_response import (
    ALEXS_SENSORS,
    DARK_CURRENT_TEMP_C,
    EPI_THICKNESS_UM,
    READ_NOISE_E_BY_RATE_HZ,
)
from .alexs_response import (
    AlexsSensor as AlexsSensor,
)

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
class FlatFieldMount:
    """Flat-field spectrometer mount: entrance arm and flat focal plane.

    Angles follow this module's convention (from the grating normal, with
    ``sin beta = m lambda/d - sin alpha``). The focal plane's normal leaves the
    grating centre at ``focal_normal_rad`` (the vendor's ``beta_H``) and meets
    the plane at ``focal_distance_mm`` (``r_H'``).

    Parameters
    ----------
    entrance_arm_mm
        Entrance-slit-to-grating distance ``r`` [mm].
    focal_distance_mm
        Perpendicular distance ``r_H'`` from the grating centre to the focal
        plane [mm].
    focal_normal_rad
        Direction ``beta_H`` of that perpendicular from the grating normal [rad].
    focal_plane_length_mm
        Optional tabulated distance between the band-end foci [mm].
    band_nm
        Optional ``(short, long)`` design band [nm].
    band_beta_rad
        Optional tabulated diffraction angles at the band ends [rad], in this
        module's convention.
    """

    entrance_arm_mm: float
    focal_distance_mm: float
    focal_normal_rad: float
    focal_plane_length_mm: float | None = None
    band_nm: tuple[float, float] | None = None
    band_beta_rad: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for name in ("entrance_arm_mm", "focal_distance_mm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"FlatFieldMount.{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not abs(float(self.focal_normal_rad)) < 0.5 * np.pi:
            raise ValueError("FlatFieldMount.focal_normal_rad must lie inside (-pi/2, pi/2)")

    def position_mm(self, beta_rad):
        """Dispersion-axis position on the flat focal plane [mm].

        A ray leaving the grating centre at ``beta`` meets the plane
        ``{p : p . n_H = r_H'}`` (unit normal ``n_H`` at ``beta_H``) after the
        path ``r'(beta) = r_H' / cos(beta - beta_H)``; its offset along the
        plane from the foot of the perpendicular is

            s(beta) = r_H' tan(beta - beta_H).

        For a plane perpendicular to the grating surface (``beta_H -> +-90``
        deg) this is ``s = -+r_H' cot(beta)`` measured from the grating plane,
        the usual flat-field statement. It is the existing
        :func:`detector_position_mm` map with ``L = r_H'`` and
        ``beta_ref = beta_H``. Limiting case: ``beta = beta_H`` lands at the
        foot, ``s = 0``. Source: plane-line intersection; flat-field mount
        geometry of Kita et al., Appl. Opt. 22, 512 (1983).

        Validation: flat-field-focal-plane
        """
        return detector_position_mm(beta_rad, self.focal_normal_rad, self.focal_distance_mm)

    def exit_arm_mm(self, beta_rad):
        """Grating-to-focal-plane path ``r'(beta) = r_H'/cos(beta - beta_H)`` [mm].

        Validation: flat-field-focal-plane
        """
        return self.focal_distance_mm / np.cos(np.asarray(beta_rad, float) - self.focal_normal_rad)


@dataclass(frozen=True)
class Grating:
    """A reflection grating in a fixed mount.

    Parameters
    ----------
    groove_density_per_mm
        Ruling density in lines per mm.
    alpha_rad
        Incidence angle from the grating normal in radians; grazing incidence is
        near ``pi/2``.
    order
        Integer diffraction order; one is typical and zero is specular.
    coating
        Reflective coating element symbol supported by the internal density
        table. The default is gold.
    groove_efficiency
        Placeholder scalar diffraction efficiency in ``[0, 1]``. This is not a
        groove-profile or rigorous coupled-wave model.
    efficiency_curve
        Optional tabulated absolute efficiency ``eta(E)`` (vendor curve). When
        set, :meth:`throughput` is ``eta(E)`` alone and is zero outside the
        curve's band; ``groove_efficiency`` must stay at one.
    mount
        Optional :class:`FlatFieldMount` used for the focal-plane map when no
        explicit detector distance is given.

    Notes
    -----
    The McPherson 251MX gratings are documented as gold-coated, but their exact
    groove profiles are not encoded. See the grazing-grating research note.
    """

    groove_density_per_mm: float
    alpha_rad: float
    order: int = 1
    coating: str = "Au"
    groove_efficiency: float = 1.0
    efficiency_curve: TabulatedEfficiency | None = None
    mount: FlatFieldMount | None = None

    def __post_init__(self) -> None:
        if self.efficiency_curve is not None:
            if not isinstance(self.efficiency_curve, TabulatedEfficiency):
                raise TypeError("Grating.efficiency_curve must be a TabulatedEfficiency")
            if self.groove_efficiency != 1.0:
                raise ValueError(
                    "an absolute efficiency_curve already includes the groove efficiency; "
                    "leave groove_efficiency at 1"
                )
        if self.mount is not None and not isinstance(self.mount, FlatFieldMount):
            raise TypeError("Grating.mount must be a FlatFieldMount or None")

    @classmethod
    def from_catalog(cls, name: str, catalog=None) -> "Grating":
        """Build a grating and its flat-field mount from a ``[gratings.NAME]`` record.

        Parameters
        ----------
        name
            Grating record name.
        catalog
            Loaded :class:`~pyrite.materials.MaterialCatalog`; ``None`` loads the
            selected catalog with its user layer (where private gratings live).

        Notes
        -----
        A ``"difference"`` record (``m lambda = d (sin alpha - sin beta)``)
        keeps its order and maps its diffraction angles to this module's
        ``"sum"`` convention by ``beta -> -beta``, since
        ``sin(-beta) = m lambda/d - sin alpha``. The flat-field mount needs
        ``r_H_prime_mm`` and ``beta_H_deg``; without them ``mount`` is
        ``None``. ``efficiency_curve`` resolves against the catalog's curves.

        Validation: flat-field-focal-plane
        """
        if catalog is None:
            from ..materials import load_material_catalog

            catalog = load_material_catalog()
        try:
            record = catalog.gratings[name]
        except KeyError:
            raise KeyError(
                f"no catalog grating {name!r}; have {sorted(catalog.gratings)}"
            ) from None
        curve_name = cast(str | None, record.get("efficiency_curve"))
        curve = (
            None if curve_name is None else TabulatedEfficiency.from_catalog(curve_name, catalog)
        )
        mount = cast(Mapping[str, Any], record["mount"])
        return cls(
            groove_density_per_mm=cast(float, record["groove_density_per_mm"]),
            alpha_rad=float(np.deg2rad(mount["alpha_deg"])),
            order=cast(int, record["order"]),
            coating=cast(str, record.get("coating", "Au")),
            efficiency_curve=curve,
            mount=_catalog_mount(mount, 1.0 if record["order_convention"] == "sum" else -1.0),
        )

    def mount_residuals(self) -> dict[str, object]:
        """Compare the grating equation with the mount's tabulated band ends.

        Returns ``beta_error_deg`` (computed minus tabulated diffraction angle
        at each band end, degrees) and ``focal_plane_length_rel_error``
        (``|s(beta_long) - s(beta_short)|`` from the computed angles relative
        to the tabulated focal-plane length); entries are ``None`` when the
        mount lacks the tabulated values.

        Validation: flat-field-focal-plane
        """
        mount = self.mount
        if mount is None or mount.band_nm is None:
            raise ValueError("mount_residuals needs a mount with band_nm")
        energy = HC_EV_ANG / (10.0 * np.asarray(mount.band_nm, float))
        beta = self.diffraction_angle_rad(energy)
        beta_error = None
        if mount.band_beta_rad is not None:
            beta_error = tuple(np.rad2deg(beta - np.asarray(mount.band_beta_rad)).tolist())
        length_error = None
        if mount.focal_plane_length_mm is not None:
            span = abs(float(np.diff(mount.position_mm(beta))[0]))
            length_error = span / mount.focal_plane_length_mm - 1.0
        return {"beta_error_deg": beta_error, "focal_plane_length_rel_error": length_error}

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
        ``abs(sin(beta)) > 1`` (no propagating order) return NaN."""
        lam = wavelength_angstrom(E_eV)
        s = self.order * lam / self.d_angstrom - np.sin(self.alpha_rad)
        s = np.where(np.abs(s) <= 1.0, s, np.nan)
        return np.arcsin(s)

    def angular_dispersion_rad_per_angstrom(self, E_eV):
        """d(beta)/d(lambda) = m / (d cos beta) [rad/Angstrom]; the grating's
        grating-only dispersion before any detector geometry."""
        beta = self.diffraction_angle_rad(E_eV)
        return self.order / (self.d_angstrom * np.cos(beta))

    def reflectivity(self, E_eV):
        """Return bare-coating grazing-incidence Fresnel reflectivity.

        Parameters
        ----------
        E_eV
            Scalar or array photon energy in eV.

        Returns
        -------
        numpy.ndarray
            Dimensionless intensity reflectivity shaped like ``E_eV``.

        Notes
        -----
        The coating uses ``n = 1 - delta - 1j*beta`` and the small-angle
        vacuum/medium Fresnel amplitude
        ``r = (theta - sqrt(theta**2 - 2*delta - 2j*beta)) /``
        ``(theta + sqrt(theta**2 - 2*delta - 2j*beta))``. The returned value is
        ``abs(r)**2``. Groove-profile efficiency is excluded; see
        :meth:`throughput`.

        The approximation treats s and p polarization equally at grazing
        incidence. With vanishing absorption, reflectivity tends to one below
        the critical angle and falls with the fourth power above it.

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
        """Grating throughput into the working order.

        With ``efficiency_curve`` set, this is the tabulated absolute
        efficiency ``eta(E)`` only: a vendor absolute efficiency already
        includes the coating reflectivity, so ``reflectivity(E)`` is not
        applied again. Energies outside the curve's band return zero (never
        extrapolated). Without a curve, the unchanged product of Fresnel
        `reflectivity(E)` and the placeholder `groove_efficiency` scalar.
        """
        if self.efficiency_curve is not None:
            return self.efficiency_curve(E_eV)
        return self.reflectivity(E_eV) * self.groove_efficiency


def _catalog_mount(mount, sign: float) -> FlatFieldMount | None:
    """Lower a validated catalog mount table; ``sign`` maps beta to "sum"."""
    if "r_H_prime_mm" not in mount or "beta_H_deg" not in mount:
        return None
    beta = mount.get("beta_deg")
    return FlatFieldMount(
        entrance_arm_mm=float(mount["r_mm"]),
        focal_distance_mm=float(mount["r_H_prime_mm"]),
        focal_normal_rad=sign * float(np.deg2rad(mount["beta_H_deg"])),
        focal_plane_length_mm=mount.get("focal_plane_length_mm"),
        band_nm=mount.get("band_nm"),
        band_beta_rad=None if beta is None else tuple(sign * np.deg2rad(beta)),
    )


def detector_position_mm(beta_rad, beta_ref_rad, distance_mm):
    """Where a ray diffracted at beta lands on a flat detector a distance
    ``distance_mm`` from the grating, whose face is perpendicular to the
    reference direction ``beta_ref_rad``: x = L tan(beta - beta_ref) [mm]."""
    return distance_mm * np.tan(np.asarray(beta_rad, float) - beta_ref_rad)


def _focal_plane(grating, beta, distance_mm, beta_ref_rad):
    """Resolve the focal-plane map: ``(position_mm, dposition_dbeta_mm)``.

    An explicit ``distance_mm`` keeps the existing flat-detector map
    ``L tan(beta - beta_ref)`` (``beta_ref`` defaulting to the mean angle);
    ``distance_mm=None`` uses the grating's :class:`FlatFieldMount`.
    """
    beta = np.asarray(beta, float)
    if distance_mm is None:
        if grating.mount is None:
            raise ValueError("distance_mm is required for a grating without a FlatFieldMount")
        mount = grating.mount
        slope = mount.focal_distance_mm / np.cos(beta - mount.focal_normal_rad) ** 2
        return mount.position_mm(beta), slope
    distance = float(distance_mm)
    reference = float(np.nanmean(beta)) if beta_ref_rad is None else beta_ref_rad
    position = detector_position_mm(beta, reference, distance)
    return position, distance / np.cos(beta - reference) ** 2


def disperse_spectrum(
    E_grid_eV, spec, grating, distance_mm=None, *, beta_ref_rad=None, weight_by_throughput=False
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

    Parameters
    ----------
    E_grid_eV
        One-dimensional photon-energy coordinate in eV.
    spec
        Spectral density per eV on ``E_grid_eV``.
    grating
        Reflection-grating geometry and throughput model.
    distance_mm
        Grating-to-detector distance in mm. ``None`` uses the grating's
        flat-field mount (:meth:`FlatFieldMount.position_mm`).
    beta_ref_rad
        Detector-normal diffraction direction in radians. ``None`` centres the
        propagating band on its mean angle. Ignored for the flat-field mount.
    weight_by_throughput
        Multiply by grating reflectivity and groove efficiency before remapping.

    Returns
    -------
    position_mm, intensity_per_mm
        Propagating detector positions and flux density per mm. Their integral
        equals the input integral after any requested throughput weighting.
    """
    E = np.asarray(E_grid_eV, float)
    spec = np.asarray(spec, float)
    if weight_by_throughput:
        spec = spec * grating.throughput(E)
    beta = grating.diffraction_angle_rad(E)
    x, _ = _focal_plane(grating, beta, distance_mm, beta_ref_rad)
    ok = np.isfinite(x)
    E, spec, x = E[ok], spec[ok], x[ok]
    if E.size < 2:
        return x, np.zeros_like(x)
    dxdE = np.gradient(x, E)
    inten = np.where(np.abs(dxdE) > 0, spec / np.abs(dxdE), 0.0)
    inten[~np.isfinite(inten)] = 0.0
    return x, inten


# ---- simple CCD: pixel binning of the dispersed profile (phased-plan step 3) --
# greateyes ALEX-s formats now live with the ALEX-s response
# (`alexs_response.ALEXS_SENSORS`, with the row dimension); re-exported here for
# existing imports. n_pix / pixel_um run along the DISPERSION axis.
@dataclass(frozen=True)
class SimpleCCD:
    """A geometry-only CCD pixel grid along the grating's dispersion axis.

    Parameters
    ----------
    n_pix
        Number of pixels along the dispersion direction.
    pixel_mm
        Pixel pitch along the dispersion direction in mm.

    Deliberately crude (phased-plan step 3, docs/research/instrumentation/grazing-grating.md): a fixed
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
    step 5; see docs/research/instrumentation/grazing-grating.md "What is NOT modelled yet"). The
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
    distance_mm=None,
    *,
    weight_by_throughput=True,
    beta_ref_rad=None,
    center_mm=None,
):
    """Combined forward-model entry (phased-plan step 4): the model's `mc_cxr_spectrum`
    (or `mc_cxr_spectrum_solid_angle`) output on ``E_grid_eV`` + a ``grating`` + a
    ``ccd`` (`SimpleCCD`) -> the dispersed, DETECTED image (counts vs pixel), by
    chaining `disperse_spectrum` then `bin_to_pixels`. So it slots in beside the
    Timepix / Eagle XO detector forward models.

    Pure composition of the two already-validated pieces above -- no new physics
    equation and so no new ledger row/validator pass, same precedent as
    `SimpleCCD`/`bin_to_pixels` themselves being pure geometry (see
    docs/research/instrumentation/grazing-grating.md phased-plan step 3's note on this).

    ``weight_by_throughput`` defaults to True HERE (unlike `disperse_spectrum`'s
    own default of False, kept for backward compatibility there): this function
    IS the detected-image entry point, so a real CCD frame should reflect the
    grating's Fresnel throughput (`Grating.throughput`), not the purely
    geometric dispersion. Pass False to get the geometry-only image instead.

    ``beta_ref_rad``/``center_mm`` forward to `disperse_spectrum`/`bin_to_pixels`
    respectively (both default to auto-centering on the dispersed light).

    Returns ``(pixel_centers_mm, counts_per_pixel)``, both length ``ccd.n_pix`` --
    same contract as `bin_to_pixels`.

    Parameters
    ----------
    E_grid_eV
        Photon-energy coordinate in eV.
    spec
        Input spectral density per eV.
    grating
        Reflection-grating geometry and throughput model.
    ccd
        One-dimensional detector pixel grid.
    distance_mm
        Grating-to-detector distance in mm; ``None`` uses the flat-field mount.
    weight_by_throughput
        Apply grating throughput before spatial remapping.
    beta_ref_rad
        Optional detector-normal diffraction direction in radians.
    center_mm
        Optional detector-grid centre in mm.

    Returns
    -------
    pixel_centers_mm, counts_per_pixel
        Arrays of length ``ccd.n_pix``. Counts use the integral units of ``spec``.
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
# climbed for the Eagle XO. QE and charge diffusion below are closed-form
# physics models (Beer-Lambert absorption efficiency; Einstein-relation
# drift-diffusion). Device constants come from the public greateyes DS-ALEXs
# Rev05 (May 2025) datasheet where it states them; the rest stay '### FILL IN'.
ACTIVE_SI_UM = EPI_THICKNESS_UM  # greateyes DS-ALEXs Rev05: 15 um epitaxial Si
#   (standard sensor; 40 um is the deep-depletion variant only).
DEPLETION_VOLTAGE_V = 40.0  ### FILL IN -- full-depletion bias [V]; not in the
#   ALEX-s datasheet; a typical deep-depletion scientific-CCD value.
OPERATING_TEMP_C = DARK_CURRENT_TEMP_C  # -100 C, the datasheet dark-current point
READ_NOISE_E = READ_NOISE_E_BY_RATE_HZ[50.0e3]  # 3.5 e- RMS at 50 kHz (datasheet)
ENTRANCE_QE_PEAK = 0.9  ### FILL IN -- entrance-surface / dead-layer loss factor
#   capping the Beer-Lambert QE below 1; not a datasheet number. Replaced by the
#   digitized catalog QE curve (`AlexS.qe`) once the user supplies it.

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
    line/band dominates the frame, less so for a broad multi-CXR spectrum
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
