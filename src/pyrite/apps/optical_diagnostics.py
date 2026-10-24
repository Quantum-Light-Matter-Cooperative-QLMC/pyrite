"""Bounded cached adapters for optical diagnostic notebooks; no new physics."""

from dataclasses import asdict
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


@lru_cache(maxsize=4)
def finite_source_demo(hole_um=0.0, duration_fs=1.0, charge_pc=0.001, distance_um=100.0):
    """Finite PEC Gaussian bunch: collected band and a single-electron field map.

    Two identical offset-free histories evaluate the analytic temporal source
    law. These are numerical template histories, not a physical realization.
    Spatial offsets and full transported trajectories are not synthesized.
    """
    from ..montecarlo.spectrum.optical_finite_collection import (
        PlaneAcceptance,
        finite_screen_bunch_plane_band,
    )
    from ..montecarlo.spectrum.optical_finite_screen import (
        PECCircularScreen,
        circular_pec_transition_field,
        propagate_screen_field,
    )
    from ..montecarlo.spectrum.optical_transition import OpticalBunchSource

    screen = PECCircularScreen(outer_radius_m=20e-6, inner_radius_m=hole_um * 1e-6)
    energy = np.linspace(0.8, 1.2, 9)
    source = OpticalBunchSource.gaussian(
        charge_pc, "analytic temporal Gaussian; identical PEC paths", rms_t_fs=duration_fs
    )
    grid = dict(nx=128, ny=128, dx_m=0.5e-6, dy_m=0.5e-6)
    band = finite_screen_bunch_plane_band(
        energy,
        [screen, screen],
        source=source,
        **grid,
        plane_z_m=distance_um * 1e-6,
        aperture=PlaneAcceptance("circle", radius_m=10e-6),
    )
    field = circular_pec_transition_field(1.0, screen, **grid)
    propagated = propagate_screen_field(field, distance_m=distance_um * 1e-6)
    return band, propagated, band.detect(synthetic_detector(energy), repetition_hz=1000)


@lru_cache(maxsize=4)
def collection_convergence(hole_um=0.0, distance_um=100.0):
    from ..montecarlo.spectrum.optical_finite_collection import PlaneAcceptance
    from ..montecarlo.spectrum.optical_finite_convergence import (
        finite_screen_collection_convergence,
    )
    from ..montecarlo.spectrum.optical_finite_screen import PECCircularScreen

    report = finite_screen_collection_convergence(
        1.0,
        PECCircularScreen(20e-6, inner_radius_m=hole_um * 1e-6),
        nx=128,
        ny=128,
        dx_m=0.5e-6,
        dy_m=0.5e-6,
        plane_z_m=distance_um * 1e-6,
        aperture=PlaneAcceptance("circle", radius_m=10e-6),
        green_pitch_m=1e-6,
    )
    return {
        **asdict(report),
        "converged": report.converged,
        "green_relative_difference": report.green_relative_difference,
    }


@lru_cache(maxsize=4)
def inference_demo(duration_fs=200.0, low_energy_eV=0.0005):
    """Synthetic two-port Gaussian exercise, explicitly independent of PEC demo."""
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

    energy = np.linspace(low_energy_eV, 0.025, 100)
    templates = GaussianCTRTemplates(
        energy,
        np.full(100, 1e8),
        np.full(100, 2e10),
        np.pi / 4,
        "SYNTHETIC per-bunch templates; not finite-source or transport-derived",
    )
    detector = synthetic_detector(energy)
    instrument = IdealMartinPuplett("SYNTHETIC ideal instrument; no measured calibration")
    mirror = np.linspace(-0.0012, 0.0012, 61)
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
        np.full(voltage.shape, 1.5e-5),
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
