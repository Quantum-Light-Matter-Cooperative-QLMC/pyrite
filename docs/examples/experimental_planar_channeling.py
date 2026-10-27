"""Shared, bounded experimental planar diagnostic for the guide and notebook."""

import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

from pyrite.experimental.quantum import propagate_planar_source, score_planar_source
from pyrite.quantum import (
    QuantumDynamics,
    QuantumNumerics,
    QuantumRegion,
    QuantumSetup,
    TransverseStateSpec,
)


def calculate(depth_ang=2_000.0, incidence_rad=0.0):
    """Project one Gaussian, evolve its density, and score forward radiation."""
    source = TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
    setup = QuantumSetup(
        region=QuantumRegion(kind="planar", plane_hkl=(1, 1, 0)),
        dynamics=QuantumDynamics(kind="channeling"),
        numerics=QuantumNumerics(basis_size=8, bloch_cells=4, step_ang=20.0),
    )
    entry = setup.project_state(
        source,
        crystal="silicon",
        kinetic_energy_eV=17e6,
        incidence_rad=incidence_rad,
        impact="fixed",
    )
    history = propagate_planar_source(entry, depth_ang=depth_ang)
    radiation = score_planar_source(
        entry,
        depth_ang=depth_ang,
        photon_energy_eV=np.linspace(1_000.0, 50_000.0, 1_025),
        theta_rad=0.0,
        sample_ang=100.0,
        absorption=False,
    )
    return source, entry, history, radiation


def compute(depth_ang=2_000.0, incidence_rad=0.0):
    """Explicit regeneration only; no stochastic histories or transport tables."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source, entry, history, radiation = calculate(depth_ang, incidence_rad)
    return {
        "source": source.to_dict(),
        "source_digest": source.digest,
        "setup": asdict(entry.setup),
        "settings": {
            "crystal": entry.crystal,
            "plane_hkl": entry.plane_hkl,
            "kinetic_energy_eV": entry.kinetic_energy_eV,
            "temperature_K": entry.temperature_K,
            "incidence_rad": entry.incidence_rad,
            "impact": entry.impact,
            "margin_eV": entry.margin_eV,
            "depth_ang": depth_ang,
            "sample_ang": 100.0,
            "theta_rad": 0.0,
            "absorption": False,
            "include_capture": True,
            "backend": "cpu",
        },
        "entry": {
            "retained": entry.retained_population,
            "free": float(entry.free_population.sum()),
            "outside": entry.outside_population,
        },
        "depth_ang": history.z_ang,
        "bound_population": history.bound_population,
        "retained_population": history.trace,
        "band_populations": history.band_populations,
        "free_flux": history.free_flux,
        "overflow_flux": history.overflow_flux,
        "reprojection_flux": history.reprojection_flux,
        "entry_flux": history.entry_flux,
        "energy_eV": radiation.photon_energy_eV,
        "spectrum": radiation.spectrum,
        "capture_spectrum": radiation.capture_spectrum,
        "units": {"population": "per incident electron", "spectrum": "ph/incident-electron/eV/sr"},
        "warnings": list(dict.fromkeys(str(item.message) for item in caught)),
    }


def plot(data):
    """Render the diagnostic's unconditioned populations and forward spectrum."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), layout="constrained")
    depth_um = np.asarray(data["depth_ang"]) / 10_000
    axes[0].plot(depth_um, data["retained_population"], label="Retained subspace")
    axes[0].plot(depth_um, data["bound_population"], label="Below barrier")
    bands = np.asarray(data["band_populations"])
    for band in range(min(3, bands.shape[1])):
        axes[0].plot(depth_um, bands[:, band], "--", lw=1, label=f"Band {band}")
    axes[0].set(
        xlabel="Depth (µm)",
        ylabel="Population per incident electron",
        title="Experimental Si(110) · 17 MeV · fixed-impact 1D Gaussian",
    )
    axes[0].legend(ncol=2, fontsize=9)
    energy_keV = np.asarray(data["energy_eV"]) / 1_000
    axes[1].plot(energy_keV, data["spectrum"], label="Total (lines + capture)")
    axes[1].plot(energy_keV, data["capture_spectrum"], "--", label="Capture component")
    axes[1].set(
        xlabel="Photon energy (keV)",
        ylabel="Photons / (incident electron · eV · sr)",
        title="Forward source density · absorption disabled",
        xlim=(energy_keV[0], energy_keV[-1]),
        ylim=(0, None),
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].legend(fontsize=9)
    return fig


if __name__ == "__main__":
    plot(compute())
    plt.show()
