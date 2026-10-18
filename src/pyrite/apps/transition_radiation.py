"""Bounded TR/CTR demo adapters around the existing planar and run solvers.

The straight-track Gaussian comparison restates ``tr-coherent-bunch`` in
docs/physics/radiation-physics/transition-radiation.md; it is not a transport
approximation used by production. No sampled offsets enter that comparison.
"""

import warnings
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.constants import electron_volt, elementary_charge, hbar

from ..campaign.longitudinal import LongitudinalDistribution
from ..materials.crystal import HBARC_EV_ANG
from ..montecarlo.geometry import tilted_geometry
from ..montecarlo.spectrum.coherent_transverse import transverse_form_factor, transverse_spot
from ..montecarlo.spectrum.transition_radiation import (
    beta_from_kinetic_eV,
    medium_dielectric,
    planar_transition_radiation,
)


@dataclass(frozen=True)
class DemoConfig:
    """CPU demo settings; durations in as, spot FWHM in nm, charge in pC."""

    energy_keV: float = 30.0
    thickness_nm: float = 20.0
    angle_deg: float = 150.0
    duration_as: float = 1.0
    spot_nm: float = 1.0
    charge_pc: float = 0.0001

    def __post_init__(self):
        for name, lower, upper in (
            ("energy_keV", 20, 100),
            ("thickness_nm", 1, 100),
            ("angle_deg", 100, 175),
            ("duration_as", 0.1, 100),
            ("spot_nm", 0.1, 100),
            ("charge_pc", 1e-6, 0.001),
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name}: demo limit is {lower} to {upper}")

    @property
    def population(self):
        return self.charge_pc * 1e-12 / elementary_charge


def photon_grid():
    """Demo source samples in eV, shared with the reference catalog profiles."""
    return np.arange(200.0, 1010.0, 10.0)


@lru_cache(maxsize=4)
def straight_track_demo(config: DemoConfig):
    """Film/interface densities and Gaussian ensemble CTR; no transport or I/O."""
    energy = photon_grid()
    angles = np.linspace(100.0, 175.0, 76)
    tilt = np.deg2rad(5.0)
    beam, ray_sample = tilted_geometry(np.deg2rad(config.angle_deg), tilt)
    sample_directions = np.array([tilted_geometry(np.deg2rad(a), tilt)[1] for a in angles])
    # Rotate the entrance face to solver +z, matching the crossing solver.
    flip = np.array([1.0, -1.0, -1.0])
    directions = sample_directions * flip
    beta = beta_from_kinetic_eV(config.energy_keV * 1e3)
    velocity = beam * beta * flip
    eps = medium_dielectric("silicon", energy)
    film = planar_transition_radiation(
        energy[:, None],
        directions,
        velocity,
        [(eps[:, None], config.thickness_nm * 10)],
    )
    ray = ray_sample * flip
    spectrum = planar_transition_radiation(
        energy,
        ray,
        velocity,
        [(eps, config.thickness_nm * 10)],
    )
    interface = planar_transition_radiation(energy, ray, velocity, [], substrate_eps=eps)
    # Existing Gaussian longitudinal + transverse law, without sampled offsets.
    # Validation: tr-coherent-bunch
    sigma_ang = config.spot_nm * 10 / np.sqrt(8 * np.log(2))
    omega_t = energy * config.duration_as * 1e-18 / (hbar / electron_volt)
    spot = transverse_spot(
        dict(
            spot_covariance_ang2=np.eye(2) * sigma_ang**2,
            face_arrival_delay=True,
            tilt_polar_rad=tilt,
            tilt_azim_rad=0.0,
            E0_keV=config.energy_keV,
        )
    )
    factor_map = np.exp(-(omega_t[:, None] ** 2)) * np.stack(
        [
            transverse_form_factor(spot, energy / HBARC_EV_ANG, n, np.zeros(3))
            for n in sample_directions
        ],
        axis=-1,
    )
    factor = np.exp(-(omega_t**2)) * transverse_form_factor(
        spot,
        energy / HBARC_EV_ANG,
        ray_sample,
        np.zeros(3),
    )
    return dict(
        energy=energy,
        angles=angles,
        film=film,
        spectrum=spectrum,
        interface=interface,
        ctr=spectrum * (1 + (config.population - 1) * factor),
        ctr_map=film * (1 + (config.population - 1) * factor_map),
        form_factor=factor,
    )


def reference_scene(config: DemoConfig, mode: str):
    """Build the production counterpart of the demo's planar reference scene."""
    import pyrite as pr

    if mode not in {"history", "bunch"}:
        raise ValueError("mode must be history or bunch")
    return pr.Scene(
        pr.Beam(
            energy_keV=config.energy_keV,
            transverse=pr.Spot(fwhm_x_mm=config.spot_nm * 1e-6),
            longitudinal=LongitudinalDistribution(
                kind="gaussian",
                envelope_rms_fs=config.duration_as * 0.001,
            ),
            bunch_charge_pc=config.charge_pc,
            rep_rate_hz=5000.0,
        ),
        pr.Slab(
            "silicon",
            thickness_ang=config.thickness_nm * 10,
            tilt_deg=5.0,
            footprint=pr.Footprint(5.0, 5.0),
        ),
        pr.Detector(
            config.angle_deg, acceptance=pr.AngularAcceptance(polar_deg=1.0, solid_angle_sr=0.0001)
        ),
        physics=pr.Physics(
            transition_radiation=mode,
            emission="incoherent",
            radiative_model="uncoupled",
            bremsstrahlung_model="eedl",
            inelastic_model="continuous",
        ),
    )


@lru_cache(maxsize=2)
def transported_demo(config: DemoConfig):
    """Run two fixed-count CPU scenes, retaining signed spectra and warnings."""
    import pyrite as pr

    numerics = pr.Numerics(
        cxr_trials=32,
        brem_trials=64,
        energy_model="midpoint",
        transport_core="per-electron",
        energy_bins=pr.EnergyBins(cxr=photon_grid(), brem=np.arange(200.0, 30000.0, 100.0)),
    )
    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = {
            mode: pr.simulate(
                scene.beam, scene.target, scene.detector, physics=scene.physics, numerics=numerics
            )
            for mode in ("history", "bunch")
            for scene in (reference_scene(config, mode),)
        }
    return (
        results,
        tuple(dict.fromkeys(str(item.message) for item in caught)),
        perf_counter() - start,
    )


def comparison_figure(data, config: DemoConfig, *, per_bunch=False, energy_eV=500.0):
    """Redraw cached straight-track spectra and an angular cut without solving."""
    import matplotlib.pyplot as plt

    scale = config.population if per_bunch else 1.0
    unit = "bunch" if per_bunch else "electron"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    energy = data["energy"]
    for values, label in (
        (data["spectrum"], "Film TR"),
        (data["ctr"], "Film CTR: Gaussian expectation"),
        (data["interface"], "Semi-infinite interface TR"),
    ):
        axes[0].semilogy(energy, values * scale, label=label)
    axes[0].set(
        xlabel="Photon energy (eV)",
        ylabel=f"photons / {unit} / eV / sr",
        title=f"Straight tracks · {config.angle_deg:g}° from beam",
    )
    axes[0].legend(fontsize=8)
    index = int(np.argmin(abs(energy - energy_eV)))
    for values, label in ((data["film"], "TR"), (data["ctr_map"], "CTR expectation")):
        axes[1].semilogy(data["angles"], values[index] * scale, label=label)
    axes[1].set(
        xlabel="Angle from beam (deg)",
        ylabel=f"photons / {unit} / eV / sr",
        title=f"Angular cut · {energy[index]:g} eV",
    )
    axes[1].legend(fontsize=8)
    axes[2].plot(energy, data["form_factor"])
    axes[2].set(
        xlabel="Photon energy (eV)",
        ylabel="Intensity form factor |F|²",
        ylim=(0, 1),
        title="Gaussian time and transverse envelope",
    )
    return fig


def transport_figure(results, config: DemoConfig, *, per_bunch=False):
    """Show the signed production component and the same total with TR removed."""
    import matplotlib.pyplot as plt

    scale = config.population if per_bunch else 1.0
    unit = "bunch" if per_bunch else "electron"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, mode in zip(axes, ("history", "bunch"), strict=True):
        result = results[mode]
        for spectrum, label in (
            (result.transition, "TR only" if mode == "history" else "CTR only (signed)"),
            (result.combined_cxr(include_transition=False), "Other CXR components"),
            (result.combined_cxr(), "Combined CXR (includes TR once)"),
        ):
            ax.plot(spectrum.photon_energy_eV, spectrum.density * scale, label=label)
        ax.set_yscale("symlog", linthresh=1e-10 * scale)
        ax.set(
            xlabel="Photon energy (eV)",
            ylabel=f"photons / {unit} / eV / sr",
            title=f"Transport · {mode} · {config.angle_deg:g}°",
        )
        ax.legend(fontsize=8)
    return fig
