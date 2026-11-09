"""Coherent slit diffraction radiation against Castellano et al., PRE 63, 056501 (2001), Fig. 8.

The paper measures total coherent power behind a horizontal slit of width ``a`` in a
45-degree aluminized Si screen (two 44 x 20 mm^2 halves), at 225 MeV, normalized to
the closed slit (TR, ``a = 0``). Collection: a 200 mm focal-length parabolic mirror
with the screen at its front focus, a Martin-Puplett interferometer that keeps the
vertical (y) polarization, and pyroelectric detectors at an unstated distance ``b``
behind the mirror with an unstated aperture. Their model (Eq. (3)) is the Fresnel
image of the screen source with phase ``exp(i k rho^2 (1 - b/f) / (2 f))``.

This check builds the PEC screen source with
:func:`~pyrite.montecarlo.spectrum.optical_finite_screen.circular_pec_transition_field`.
It uses projected normal-incidence geometry: the 44 mm side is foreshortened to
44/sqrt(2) mm and the slit edges are kept. It weights frequencies by the digitized
universal coherent curve of Fig. 7 (``a = 0``). It then integrates the y-polarized
Fresnel image over a detector disc of angular radius ``theta_d = r_d / f``.

The result scans the two unpublished parameters ``b/f`` and ``theta_d``:

* focal-plane detection (``b = f``, the far field of the finite screen) cannot match
  the measured slit dependence; and
* plausible near-field geometries reach a Fig. 8 rms of about 0.05.

This is a screening record, not an anchor. The paper's ref. [13] (Castellano et al.,
NIM A 435, 297 (1999)) is generic Kirchhoff theory and gives no TTF detector geometry.
Its kappa <= k truncation of the source (the ``J0(k rho)/rho`` term of Eq. (3)) changes
these ratios by < 1 % for b/f <= 7. That model difference therefore does not explain
the focal-plane miss. A point electron is used and transverse beam size is ignored,
as in the paper's model.

Leissner et al. (PAC 1999, p. 2172; PRE ref. [15]) give the interferometer topology
but no distances. A parabolic mirror collimates the radiator light, the
Martin-Puplett splits and recombines it, and a second parabolic mirror focuses it
onto horn-antenna pyroelectric detectors. A radiator at the front focus of the first
mirror and detectors at the focus of the second form an *imaging* relay (NIM A 435,
scheme (c)). The check therefore also scans an image-plane model, with one free
parameter once the pupil is taken from PRE's 200 mrad acceptance:

* the y-polarized image is low-pass filtered by the pupil half-angle ``theta_a``
  (0.1 or 0.2 rad, since "200 mrad" is ambiguous between full and half angle); and
* it is integrated over a detector disc of radius ``rho_d``, referred to the screen
  (horn radius / magnification).

With ``rho_d -> infinity`` this equals focal-plane detection over ``theta_a`` (Parseval),
which the check asserts. The best ``rho_d`` (10-15 mm) is a plausible horn footprint
but is not published.

Leissner's diploma thesis (PITHA 99/4, 1998, Ch. 4) gives the relay for the same
interferometer:

* both parabolas have f = 200 mm and a 90 mm aperture, so the relay images at unit
  magnification for any spacing (953 mm here);
* the radiator sat 220 mm from the first mirror, which leaves about 20 mm of residual
  defocus (changes the ratios by about 0.001, not modelled);
* the 90 mm mirror at 200-220 mm sets the pupil half-angle at 0.20-0.22 rad,
  matching PRE's "200 mrad"; and
* the horns are 55.9 mm (mouth) to 6.2 mm (throat), on 5 mm pyro crystals.

Brightness conservation fixes the horn's throughput but not its phase-space shape.
``thesis_models`` therefore evaluates three equal-etendue readings with no fitted
parameter:

* ``disc`` -- full pupil, screen-referred disc of radius r_t / sin(theta_a), with
  r_t the throat or crystal radius;
* ``mouth`` -- the whole 55.9 mm mouth at the image, angles limited to
  asin(r_t / r_mouth); and
* ``throat`` -- the throat itself at the image plane.

Only ``disc`` reproduces Fig. 8, so the agreement is conditional on that horn model.
The thesis describes the 1998 TR station; the 2000 DR run used the same
interferometer (PRE ref. [15]), but its installation distances are assumed.

Run: ``uv run python checks/castellano_slit_dr.py`` (about 1 min).
"""

import sys

import numpy as np

from pyrite.montecarlo.spectrum.optical_finite_screen import (
    PECCircularScreen,
    circular_pec_transition_field,
)

H_EV_S = 4.135667696e-15
C_M_S = 299792458.0
GAMMA = 1.0 + 225.0 / 0.51099895
BETA = float(np.sqrt(1.0 - 1.0 / GAMMA**2))
FOCAL_M = 0.2
HALF_WIDTH_M = 0.044 / np.sqrt(2.0) / 2.0  # 44 mm side seen at 45 degrees
HALF_HEIGHT_M = 0.020
PITCH_M = 0.5e-3
SLITS_M = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]) * 1e-3
# Fig. 8 measured points (digitized; +-0.01 reading).
MEASURED = np.array([1.0, 0.93, 0.755, 0.64, 0.535, 0.44])
# Fig. 7 universal coherent / theoretical incoherent curve, a = 0 (digitized).
FIG7_THZ = np.array(
    [0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    + [0.17, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40]
)
FIG7_WEIGHT = np.array(
    [0.0, 0.3, 0.4, 0.6, 0.8, 0.95, 1.0, 0.93, 0.8, 0.65, 0.5, 0.35, 0.25, 0.15]
    + [0.1, 0.06, 0.04, 0.02, 0.0, 0.05, 0.12, 0.10]
)


def screen_source(frequency_thz: float, slit_m: float):
    """Pixel coordinates and the y component of the PEC source on the slit screen."""
    energy = H_EV_S * frequency_thz * 1e12
    corner = np.hypot(HALF_WIDTH_M, HALF_HEIGHT_M + SLITS_M.max() / 2)
    n = int(np.ceil(2.2 * corner / PITCH_M))
    n += n % 2
    # A disc covering the rectangle supplies the incident field; the slit mask
    # then defines the metal (same physical-optics source, other outline).
    disc = PECCircularScreen(outer_radius_m=1.05 * corner, beta_z=-BETA)
    field = circular_pec_transition_field(energy, disc, nx=n, ny=n, dx_m=PITCH_M, dy_m=PITCH_M)
    axis = (np.arange(n) - (n - 1) / 2) * PITCH_M
    cols = np.abs(axis) <= HALF_WIDTH_M
    x, y = axis[cols], axis
    ay = np.abs(y)[:, None]
    metal = (ay >= slit_m / 2) & (ay <= slit_m / 2 + HALF_HEIGHT_M)
    return energy, x, y, np.where(metal, field.electric_xy_v_s_m[1][:, cols], 0.0)


def detector_power(source, b_over_f: float, theta_d: float, n_grid: int = 81) -> float:
    """Relative y-polarized energy on a detector disc of angular radius theta_d.

    Fresnel image of the screen through the lens (paper Eq. (3)) on a uniform
    angular grid, evaluated as a separable discrete Fourier sum.
    """
    energy, x, y, ey = source
    k = 2 * np.pi * energy / (H_EV_S * C_M_S)
    r2 = x[None, :] ** 2 + y[:, None] ** 2
    phased = ey * np.exp(1j * k * r2 * (1 - b_over_f) / (2 * FOCAL_M))
    t = (np.arange(n_grid) - (n_grid - 1) / 2) * (2 * theta_d / n_grid)
    amplitude = np.exp(-1j * k * np.outer(t, y)) @ phased @ np.exp(-1j * k * np.outer(x, t))
    inside = np.hypot(t[:, None], t[None, :]) <= theta_d
    return float(k**2 * np.sum(np.abs(amplitude[inside]) ** 2))


def slit_curve(sources, weights, b_over_f: float, theta_d: float) -> np.ndarray:
    power = np.array(
        [
            sum(w * detector_power(s, b_over_f, theta_d) for w, s in zip(weights, row, strict=True))
            for row in sources
        ]
    )
    return power / power[0]


def image_curves(sources, weights, theta_a: float, radii_m, pad: int = 512) -> np.ndarray:
    """Image-plane relay: pupil-filtered y image integrated over discs of ``radii_m``."""
    q = 2 * np.pi * np.fft.fftfreq(pad, PITCH_M)
    # Centred coordinates of the padded (wrapped) image, origin at the beam.
    power = np.zeros((len(sources), len(radii_m)))
    for i, row in enumerate(sources):
        for w, (energy, x, y, ey) in zip(weights, row, strict=True):
            k = 2 * np.pi * energy / (H_EV_S * C_M_S)
            grid = np.zeros((pad, pad), complex)
            grid[: ey.shape[0], : ey.shape[1]] = ey
            spectrum = np.fft.fft2(grid) * (np.hypot(q[None], q[:, None]) <= k * np.sin(theta_a))
            image = np.abs(np.fft.ifft2(spectrum)) ** 2
            span = pad * PITCH_M
            gx = (x[0] + np.arange(pad) * PITCH_M + span / 2) % span - span / 2
            gy = (y[0] + np.arange(pad) * PITCH_M + span / 2) % span - span / 2
            r = np.hypot(gx[None], gy[:, None])
            power[i] += w * np.array([image[r <= rd].sum() for rd in radii_m])
    return power / power[0]


MOUTH_RADIUS_M = 55.9e-3 / 2
THROAT_RADIUS_M = 6.2e-3 / 2
CRYSTAL_RADIUS_M = 5.0e-3 / 2


def thesis_models(sources, weights) -> dict[str, float]:
    """Fig. 8 rms of the equal-etendue horn readings with Leissner (1998) geometry."""
    rms = {}
    for pupil in (0.20, 0.22):
        for label, r_t in (("throat", THROAT_RADIUS_M), ("crystal", CRYSTAL_RADIUS_M)):
            curve = image_curves(sources, weights, pupil, [r_t / np.sin(pupil)])[:, 0]
            rms[f"disc pupil={pupil} {label}"] = float(np.sqrt(np.mean((curve - MEASURED) ** 2)))
            print(f"thesis disc pupil={pupil} r={label}", np.round(curve, 3))
        curve = image_curves(sources, weights, pupil, [THROAT_RADIUS_M])[:, 0]
        rms[f"throat pupil={pupil}"] = float(np.sqrt(np.mean((curve - MEASURED) ** 2)))
    horn = float(np.arcsin(THROAT_RADIUS_M / MOUTH_RADIUS_M))
    curve = image_curves(sources, weights, horn, [MOUTH_RADIUS_M])[:, 0]
    rms["mouth"] = float(np.sqrt(np.mean((curve - MEASURED) ** 2)))
    for name, value in rms.items():
        print(f"thesis {name:<28} rms={value:.3f}")
    return rms


def main() -> int:
    frequencies = np.linspace(0.045, 0.40, 30)
    weights = np.interp(frequencies, FIG7_THZ, FIG7_WEIGHT)
    frequencies, weights = frequencies[weights > 0], weights[weights > 0]
    sources = [[screen_source(f, a) for f in frequencies] for a in SLITS_M]
    print("measured Fig. 8:", MEASURED)
    best, far = np.inf, np.inf
    for theta_d in (0.02, 0.05, 0.1):
        for b_over_f in (1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
            curve = slit_curve(sources, weights, b_over_f, theta_d)
            rms = float(np.sqrt(np.mean((curve - MEASURED) ** 2)))
            print(f"theta_d={theta_d:<5} b/f={b_over_f:<5}", np.round(curve, 3), f"rms={rms:.3f}")
            if b_over_f == 1.0:
                far = min(far, rms)
            else:
                best = min(best, rms)
    print(f"best focal-plane (b=f) rms {far:.3f}; best near-field rms {best:.3f}")
    radii = np.array([5.0, 7.0, 10.0, 12.5, 15.0, 20.0, 1e3]) * 1e-3
    image_best, parseval = np.inf, 0.0
    for theta_a in (0.1, 0.2):
        curves = image_curves(sources, weights, theta_a, radii)
        focal = slit_curve(sources, weights, 1.0, theta_a)
        parseval = max(parseval, float(np.max(np.abs(curves[:, -1] - focal))))
        for rd, curve in zip(radii, curves.T, strict=True):
            rms = float(np.sqrt(np.mean((curve - MEASURED) ** 2)))
            print(
                f"image theta_a={theta_a:<4} rho_d={rd * 1e3:<6g}mm",
                np.round(curve, 3),
                f"rms={rms:.3f}",
            )
            if rd < 1.0:
                image_best = min(image_best, rms)
    print(f"best image-plane rms {image_best:.3f}; |image(rho_d=inf) - focal| {parseval:.4f}")
    thesis = thesis_models(sources, weights)
    disc = max(v for k, v in thesis.items() if k.startswith("disc"))
    other = min(v for k, v in thesis.items() if not k.startswith("disc"))
    # Screening criteria: far-field detection excluded, near-field and one-parameter
    # image fits exist, Parseval limit consistent, and with the thesis geometry the
    # equal-etendue disc horn model fits while the other horn readings do not.
    ok = far > 0.15 and best < 0.07 and image_best < 0.05 and parseval < 0.01
    return 0 if ok and disc < 0.10 and other > 0.15 else 1


if __name__ == "__main__":
    sys.exit(main())
