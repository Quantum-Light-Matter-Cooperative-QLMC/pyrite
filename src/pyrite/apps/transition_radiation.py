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
    """Straight-track TR/CTR settings.

    Durations in as, spot FWHM in nm, charge in pC, angles in degrees from the
    incoming beam in the lab x–z plane (forward below 90°, backward above).
    The photon band and map angles are point grids; validation is physical
    (finite, positive, below the beam energy) and a memory bound only.
    """

    energy_keV: float = 30.0
    thickness_nm: float = 20.0
    angle_deg: float = 150.0
    duration_as: float = 1.0
    spot_nm: float = 1.0
    charge_pc: float = 0.0001
    material: str = "silicon"
    tilt_deg: float = 5.0
    start_eV: float = 200.0
    stop_eV: float = 1000.0
    energy_points: int = 81
    log_energy: bool = False
    angle_min_deg: float = 100.0
    angle_max_deg: float = 175.0
    angle_points: int = 76

    def __post_init__(self):
        for name, lower, upper in (
            ("energy_keV", 0.1, 1e8),
            ("thickness_nm", 0.01, 1e7),
            ("angle_deg", -180, 180),
            ("duration_as", 1e-3, 1e15),
            ("spot_nm", 1e-3, 1e10),
            ("charge_pc", 1e-9, 1e5),
            ("tilt_deg", -89.5, 89.5),
            ("start_eV", 1.0, 1e9),
            ("angle_min_deg", -180, 180),
            ("angle_max_deg", -180, 180),
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name}: demo limit is {lower} to {upper}")
        if not self.start_eV < self.stop_eV <= self.energy_keV * 1e3:
            raise ValueError("photon band must satisfy start < stop <= beam kinetic energy")
        if not self.angle_min_deg < self.angle_max_deg:
            raise ValueError("map angles must satisfy min < max")
        if self.energy_points < 2 or self.angle_points < 2:
            raise ValueError("use at least two energy and two angle points")
        if self.energy_points * self.angle_points > MAX_MAP_ELEMENTS:
            raise ValueError(
                f"map needs {self.energy_points * self.angle_points:,} samples; "
                f"the bound is {MAX_MAP_ELEMENTS:,}"
            )
        if abs(_sample_direction(self.angle_deg, self.tilt_deg)[2]) < GRAZING_COS:
            raise ValueError("observation direction is parallel to the film faces")

    @property
    def population(self):
        return self.charge_pc * 1e-12 / elementary_charge


#: Largest (energy x angle) straight-track map; about 0.5 GB of complex scratch.
MAX_MAP_ELEMENTS = 4_000_000
#: Directions within this |cos| of the film plane are not radiated into.
GRAZING_COS = 1e-6


def photon_grid(config: DemoConfig | None = None):
    """Photon sample energies in eV; the default matches the catalog reference profiles."""
    config = DemoConfig() if config is None else config
    if config == DemoConfig():
        return np.arange(200.0, 1010.0, 10.0)
    spacing = np.geomspace if config.log_energy else np.linspace
    return spacing(config.start_eV, config.stop_eV, config.energy_points)


def map_angles(config: DemoConfig):
    return np.linspace(config.angle_min_deg, config.angle_max_deg, config.angle_points)


def _sample_direction(angle_deg, tilt_deg):
    return tilted_geometry(np.deg2rad(angle_deg), np.deg2rad(tilt_deg))[1]


def _solver_frame(beam, directions):
    """Map sample-frame rays to the planar solver's emitting face at +z.

    Backward rays (n_z < 0) leave the entrance face: flip the frame so the
    electron enters from the vacuum (v_z < 0). Forward rays leave the exit
    face unflipped (v_z > 0). A single free-standing film is symmetric, so
    the layer order is unchanged. Grazing rays return NaN masks.
    """
    flip = np.array([1.0, -1.0, -1.0])
    backward = directions[..., 2] < 0
    solver = np.where(backward[..., None], directions * flip, directions)
    velocity = np.where(backward[..., None], beam * flip, beam)
    grazing = np.abs(directions[..., 2]) < GRAZING_COS
    return solver, velocity, grazing


@lru_cache(maxsize=4)
def straight_track_demo(config: DemoConfig):
    """Film/interface densities and Gaussian ensemble CTR; no transport or I/O.

    ``film``/``ctr_map`` have shape (energy, angle) over :func:`map_angles`;
    grazing map directions are NaN (unsupported), never zero.
    """
    energy = photon_grid(config)
    angles = map_angles(config)
    tilt = np.deg2rad(config.tilt_deg)
    beam, ray_sample = tilted_geometry(np.deg2rad(config.angle_deg), tilt)
    sample_directions = np.array([_sample_direction(a, config.tilt_deg) for a in angles])
    beta = beta_from_kinetic_eV(config.energy_keV * 1e3)
    eps = medium_dielectric(config.material, energy)
    if not np.all(np.isfinite(eps)):
        raise ValueError(f"no dielectric data for {config.material} over the photon band")
    directions, velocities, grazing = _solver_frame(beam, sample_directions)
    safe = np.where(grazing[:, None], np.array([0.0, 0.0, 1.0]), directions)
    film = planar_transition_radiation(
        energy[:, None],
        safe,
        velocities * beta,
        [(eps[:, None], config.thickness_nm * 10)],
    )
    film = np.where(grazing[None, :], np.nan, film)
    ray, velocity, _ = _solver_frame(beam, ray_sample)
    spectrum = planar_transition_radiation(
        energy,
        ray,
        velocity * beta,
        [(eps, config.thickness_nm * 10)],
    )
    interface = planar_transition_radiation(energy, ray, velocity * beta, [], substrate_eps=eps)
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


def map_panels(data, config: DemoConfig, *, per_bunch=False):
    """TR and CTR expectation maps as shared angular-map panels."""
    from .angular_maps import AngularPanel

    scale = config.population if per_bunch else 1.0
    return [
        AngularPanel(title, data["angles"], data["energy"], (data[key] * scale).T)
        for key, title in (("film", "Film TR"), ("ctr_map", "Film CTR expectation"))
    ]


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
            config.material,
            thickness_ang=config.thickness_nm * 10,
            tilt_deg=config.tilt_deg,
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
def transported_demo(config: DemoConfig, *, trials=32, transport_core="per-electron"):
    """Run two fixed-count scenes, retaining signed spectra and warnings.

    ``trials`` line histories (twice as many continuum histories). The CUDA
    core needs the CUDA array backend fixed at launch; TR crossings then use
    the fused CUDA kernel.
    """
    import pyrite as pr

    stop = config.energy_keV * 1e3
    numerics = pr.Numerics(
        cxr_trials=trials,
        brem_trials=2 * trials,
        energy_model="midpoint",
        transport_core=transport_core,
        energy_bins=pr.EnergyBins(
            cxr=photon_grid(config),
            brem=np.arange(200.0, stop, 100.0)
            if config == DemoConfig()
            else np.geomspace(max(config.start_eV, 10.0), 0.999 * stop, 300),
        ),
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
    """Redraw cached straight-track spectra and an angular cut without solving.

    Uses a standalone Figure (no pyplot state), so a long-running app server
    does not accumulate figures.
    """
    from matplotlib.figure import Figure

    scale = config.population if per_bunch else 1.0
    unit = "bunch" if per_bunch else "electron"
    fig = Figure(figsize=(15, 4), layout="constrained")
    axes = fig.subplots(1, 3)
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
        title=f"Straight tracks · {config.material} · {config.angle_deg:g}° from beam",
    )
    axes[0].legend(fontsize=8)
    index = int(np.argmin(abs(energy - energy_eV)))
    for values, label in ((data["film"], "TR"), (data["ctr_map"], "CTR expectation")):
        axes[1].semilogy(data["angles"], values[index] * scale, label=label)
    axes[1].set(
        xlabel="Signed angle from beam (deg)",
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
    from matplotlib.figure import Figure

    scale = config.population if per_bunch else 1.0
    unit = "bunch" if per_bunch else "electron"
    fig = Figure(figsize=(12, 4), layout="constrained")
    axes = fig.subplots(1, 2)
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


@lru_cache(maxsize=2)
def cached_dtr_reference(two_beam=False):
    """Reuse the maintained unverified field reference; no transport."""
    from ..validation.dtr_reference import reference_spectrum

    return reference_spectrum(two_beam=two_beam)
