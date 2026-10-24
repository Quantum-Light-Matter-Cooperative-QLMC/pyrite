"""Bounded cached adapters for optical diagnostic notebooks; no new physics."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


def synthetic_detector(energy):
    from ..detectors.optical_energy import CalibratedPulseEnergyDetector

    return CalibratedPulseEnergyDetector(
        np.array([energy[0], energy[-1]]),
        np.array([0.25, 0.25]),
        pulse_gain_v_per_j=1e12,
        calibration_id="SYNTHETIC flat 25% absorptance; not measured",
    )


@dataclass(frozen=True)
class ScreenConfig:
    """Finite PEC screen collection settings (SI-friendly display units).

    The defaults reproduce the original bounded demo. Every field is open
    to physical values; only the FFT workspace is bounded, by memory.
    """

    hole_um: float = 0.0
    outer_um: float = 20.0
    beta: float = 0.95
    duration_fs: float = 1.0
    charge_pc: float = 0.001
    distance_um: float = 100.0
    aperture_um: float = 10.0
    low_eV: float = 0.8
    high_eV: float = 1.2
    energy_points: int = 9
    grid: int = 128
    pitch_um: float = 0.5
    device: str = "cpu"

    def __post_init__(self):
        values = (self.outer_um, self.distance_um, self.aperture_um, self.pitch_um, self.charge_pc)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("screen, distance, aperture, pitch and charge must be positive")
        if not 0 <= self.hole_um < self.outer_um:
            raise ValueError("hole radius must lie in [0, outer radius)")
        if not 0 < self.beta < 1:
            raise ValueError("beta must lie in (0, 1)")
        if not (np.isfinite(self.duration_fs) and self.duration_fs >= 0):
            raise ValueError("duration must be finite and nonnegative")
        if not 0 < self.low_eV < self.high_eV or self.energy_points < 2:
            raise ValueError("photon band needs 0 < low < high and at least two points")
        if not 16 <= self.grid <= MAX_SCREEN_GRID:
            raise ValueError(f"grid must lie in [16, {MAX_SCREEN_GRID}] samples per side")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")

    @property
    def window_um(self) -> float:
        return self.grid * self.pitch_um

    def xp(self):
        """Array module for the field solves; CUDA needs the CUDA launch backend."""
        if self.device == "cpu":
            return np
        from .._backend import BACKEND

        if BACKEND.name != "cuda":
            raise ValueError("CUDA was requested but the app was not launched with --backend cuda")
        return BACKEND.xp

    def grid_kwargs(self):
        pitch = self.pitch_um * 1e-6
        return dict(nx=self.grid, ny=self.grid, dx_m=pitch, dy_m=pitch)

    def screen(self, **kwargs):
        from ..montecarlo.spectrum.optical_finite_screen import PECCircularScreen

        return PECCircularScreen(
            self.outer_um * 1e-6, inner_radius_m=self.hole_um * 1e-6, beta_z=-self.beta, **kwargs
        )

    def aperture(self):
        from ..montecarlo.spectrum.optical_finite_collection import PlaneAcceptance

        return PlaneAcceptance("circle", radius_m=self.aperture_um * 1e-6)


#: Largest FFT side; 4096² complex vector fields need a few GB of workspace.
MAX_SCREEN_GRID = 4096


@lru_cache(maxsize=4)
def finite_source_demo(config: ScreenConfig = ScreenConfig()):
    """Finite PEC Gaussian bunch: collected band and a single-electron field map.

    Two identical offset-free histories evaluate the analytic temporal source
    law. These are numerical template histories, not a physical realization.
    Spatial offsets and full transported trajectories are not synthesized.
    """
    from ..montecarlo.spectrum.optical_finite_collection import finite_screen_bunch_plane_band
    from ..montecarlo.spectrum.optical_finite_screen import (
        circular_pec_transition_field,
        propagate_screen_field,
    )
    from ..montecarlo.spectrum.optical_transition import OpticalBunchSource

    screen = config.screen()
    energy = np.linspace(config.low_eV, config.high_eV, config.energy_points)
    source = OpticalBunchSource.gaussian(
        config.charge_pc,
        "analytic temporal Gaussian; identical PEC paths",
        rms_t_fs=config.duration_fs,
    )
    xp = config.xp()
    band = finite_screen_bunch_plane_band(
        energy,
        [screen, screen],
        source=source,
        **config.grid_kwargs(),
        plane_z_m=config.distance_um * 1e-6,
        aperture=config.aperture(),
        xp=xp,
    )
    field = circular_pec_transition_field(
        float(energy[len(energy) // 2]), screen, **config.grid_kwargs(), xp=xp
    )
    propagated = propagate_screen_field(field, distance_m=config.distance_um * 1e-6)
    return band, propagated, band.detect(synthetic_detector(energy), repetition_hz=1000)


@lru_cache(maxsize=4)
def collection_convergence(config: ScreenConfig = ScreenConfig()):
    from ..montecarlo.spectrum.optical_finite_convergence import (
        finite_screen_collection_convergence,
    )

    report = finite_screen_collection_convergence(
        float(np.sqrt(config.low_eV * config.high_eV)),
        config.screen(),
        **config.grid_kwargs(),
        plane_z_m=config.distance_um * 1e-6,
        aperture=config.aperture(),
        green_pitch_m=2 * config.pitch_um * 1e-6,
    )
    return {
        **asdict(report),
        "converged": report.converged,
        "green_relative_difference": report.green_relative_difference,
    }


def report_rows(report: dict) -> list[dict]:
    """Flatten a convergence/fit report into (quantity, value) table rows."""
    rows = []
    for key, value in report.items():
        if isinstance(value, dict):
            rows.extend(
                {"quantity": f"{key}.{inner}", "value": str(v)} for inner, v in value.items()
            )
        else:
            rows.append({"quantity": key, "value": str(value)})
    return rows


@lru_cache(maxsize=4)
def inference_demo(
    duration_fs=200.0,
    low_energy_eV=0.0005,
    high_energy_eV=0.025,
    mirror_mm=1.2,
    nodes=61,
    noise_uV=15.0,
):
    """Synthetic two-port Gaussian exercise, explicitly independent of PEC demo.

    ``mirror_mm`` is the one-sided mirror travel (OPD is twice it), sampled
    at ``nodes`` points; ``noise_uV`` is the declared RMS voltage error of a
    noise-free expectation scan.
    """
    if not (0 < low_energy_eV < high_energy_eV and mirror_mm > 0 and nodes >= 5 and noise_uV > 0):
        raise ValueError("need 0 < low < high band, positive travel/noise and at least 5 nodes")
    from .._optical_coherency import OpticalSpectralCoherency
    from ..detectors.optical_ctr_inference import (
        GaussianCTRTemplates,
        OpticalVoltageScan,
        _profile_at_sigma,
        _response_kernel,
        fit_gaussian_ctr,
        gaussian_ctr_forward_voltage,
    )
    from ..detectors.optical_interferometer import IdealMartinPuplett, polarized_interferogram

    energy = np.linspace(low_energy_eV, high_energy_eV, 100)
    templates = GaussianCTRTemplates(
        energy,
        np.full(100, 1e8),
        np.full(100, 2e10),
        np.pi / 4,
        "SYNTHETIC per-bunch templates; not finite-source or transport-derived",
    )
    detector = synthetic_detector(energy)
    instrument = IdealMartinPuplett("SYNTHETIC ideal instrument; no measured calibration")
    mirror = np.linspace(-mirror_mm * 1e-3, mirror_mm * 1e-3, int(nodes))
    voltage = gaussian_ctr_forward_voltage(
        templates,
        mirror,
        rms_duration_fs=duration_fs,
        instrument=instrument,
        detectors=(detector, detector),
    )
    scan = OpticalVoltageScan(
        mirror,
        voltage,
        np.full(voltage.shape, noise_uV * 1e-6),
        provenance="SYNTHETIC noise-free scan with declared RMS error",
    )
    fit = fit_gaussian_ctr(
        scan,
        templates,
        instrument=instrument,
        detectors=(detector, detector),
        sigma_bounds_fs=(20, 3000),
        profile_points=51,
    )
    # Display the fitter's existing profiled objective, with the same nuisance settings.
    kernel = _response_kernel(templates, mirror, instrument, (detector, detector), 4_000_000)
    sigma_grid = np.unique(np.r_[np.geomspace(*fit.profile_bounds_fs, 81), fit.rms_duration_fs])
    chi2 = [
        _profile_at_sigma(
            sigma,
            templates,
            kernel,
            scan,
            fit_scale=True,
            fit_offsets=False,
            scale_prior_rms_fraction=None,
        )[0]
        for sigma in sigma_grid
    ]
    fit.metadata["diagnostic_profile"] = {
        "sigma_fs": sigma_grid.tolist(),
        "delta_chi2": (np.asarray(chi2) - fit.chi_square).tolist(),
    }
    density = templates.photon_density(duration_fs)
    coherency = OpticalSpectralCoherency.linearly_polarized(
        energy, density, angle_rad=np.pi / 4, provenance=templates.provenance
    )
    interferogram = polarized_interferogram(
        coherency, mirror, instrument=instrument, detectors=(detector, detector)
    )
    return scan, fit, interferogram


def load_saved_optical(path, kind):
    """Use existing non-pickled artifact loaders; never run source/transport."""
    from ..detectors.optical_ctr_inference import load_gaussian_ctr_fit
    from ..detectors.optical_interferometer import load_optical_interferogram
    from ..montecarlo.spectrum.optical_transition import load_optical_transition

    loaders = {
        "Fit": load_gaussian_ctr_fit,
        "Interferogram": load_optical_interferogram,
        "Planar source": load_optical_transition,
    }
    return loaders[kind](Path(path).expanduser())


@lru_cache(maxsize=8)
def cherenkov_cone_map(
    n=1.46, photon_energy_eV=2.5, kinetic_keV=(100.0, 1e5), points=200, angle_bins=180
):
    """Bulk Frank--Tamm cone versus electron kinetic energy at one photon energy.

    Internal source in an unbounded lossless medium of constant index ``n``:
    no interfaces, escape or total internal reflection. Each kinetic energy
    deposits its whole yield in the polar bin (about the velocity) holding
    ``theta_C``; values are bin-averaged photons/m/eV/sr. Returns the map
    panel, the kinetic threshold [keV] (inf when ``n <= 1``) and the
    ultrarelativistic cone angle [deg].
    """
    from ..materials.crystal import M_E_EV
    from ..montecarlo.spectrum.cherenkov import ChargedPassage, ConstantIndex, frank_tamm_yield
    from ..montecarlo.spectrum.transition_radiation import beta_from_kinetic_eV
    from .angular_maps import AngularPanel

    medium = ConstantIndex(n)
    low, high = kinetic_keV
    if not (0 < low < high) or points < 2 or angle_bins < 2:
        raise ValueError("need 0 < kinetic min < max and at least two energies and angle bins")
    kinetic = np.geomspace(low, high, int(points))
    edges = np.linspace(0.0, np.pi / 2, int(angle_bins) + 1)
    solid_angle = 2 * np.pi * (np.cos(edges[:-1]) - np.cos(edges[1:]))
    per_bin = np.stack(
        [
            frank_tamm_yield(
                ChargedPassage(beta_from_kinetic_eV(t * 1e3)), [photon_energy_eV], medium
            ).angular_histogram(edges)[0]
            for t in kinetic
        ]
    )
    panel = AngularPanel(
        f"Cherenkov cone · n = {n:g} · {photon_energy_eV:g} eV",
        np.rad2deg(0.5 * (edges[:-1] + edges[1:])),
        kinetic,
        (per_bin / solid_angle).T,
    )
    threshold = M_E_EV / 1e3 * (1 / np.sqrt(1 - 1 / n**2) - 1) if n > 1 else np.inf
    cone = np.rad2deg(np.arccos(1 / n)) if n > 1 else np.nan
    return panel, float(threshold), float(cone)
