"""A small CPU HOPG simulation."""

import matplotlib.pyplot as plt
import numpy as np

import pyrite as pr
from pyrite.beams import PointBunch, TransverseSpot
from pyrite.config import CxrReflections, FixedSampling
from pyrite.detectors import Detector
from pyrite.geometry import Footprint, Slab


def compute():
    beam = pr.Beam(
        energy_keV=30.0,
        transverse=TransverseSpot(fwhm_x_mm=1.0),
        longitudinal=PointBunch(),
    )
    target = Slab(
        "hopg",
        thickness_ang=1_000.0,
        tilt_deg=30.0,
        footprint=Footprint(5.0, 5.0),
    )
    result = pr.simulate(
        beam,
        target,
        Detector(observation_angle_deg=135.0),
        numerics=pr.Numerics(
            sampling=FixedSampling(cxr_electrons=2**15, brem_electrons=128),
            cxr_reflections=CxrReflections(n_families=2),
        ),
    )

    # Extract spectra
    cxr_spectrum = result.cxr.incoherent
    brem_spectrum = result.brem

    # Interpolate the coarse Brem grid onto the fine CXR grid
    total_spectral_density = np.interp(cxr_spectrum.photon_energy_eV, brem_spectrum.photon_energy_eV, brem_spectrum.density) + cxr_spectrum.density

    # Construct the output dictionary
    data = {
        "seed": result.case["seed"],
        "provenance": dict(result.provenance),
        "spectra": {
            "PXR + CBS": {
                "energy_eV": cxr_spectrum.photon_energy_eV,
                "density": cxr_spectrum.density,
            },
            "Bremsstrahlung": {
                "energy_eV": brem_spectrum.photon_energy_eV,
                "density": brem_spectrum.density,
            },
            "Total Spectrum": {
                "energy_eV": cxr_spectrum.photon_energy_eV,
                "density": total_spectral_density
            }
        },
    }

    return data


def plot(data):
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for name, spectrum in data["spectra"].items():
        ax.plot(spectrum["energy_eV"], spectrum["density"], label=name)
    ax.set(
        xlabel="Photon energy (eV)",
        ylabel="Photons / (incident electron · eV · sr)",
        title="Simulated 30 keV electrons → 100 nm HOPG",
        xlim=(500, 2_500),
        yscale="log",
        ylim=(1e-14, 5e-8),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


if __name__ == "__main__":
    plot(compute())
    plt.show()
