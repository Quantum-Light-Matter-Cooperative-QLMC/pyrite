"""A small CPU HOPG simulation; not a converged scientific prediction."""

import matplotlib.pyplot as plt

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
            sampling=FixedSampling(cxr_electrons=64, brem_electrons=32),
            cxr_reflections=CxrReflections(n_families=2),
        ),
    )
    # Single-scene simulation resolves a fixed seed; retain it with the output.
    return {
        "seed": result.case["seed"],
        "provenance": dict(result.provenance),
        "spectra": {
            name: {
                "energy_eV": spectrum.photon_energy_eV.tolist(),
                "density": spectrum.density.tolist(),
            }
            for name, spectrum in (
                ("PXR + CBS", result.cxr.incoherent),
                ("Bremsstrahlung", result.brem),
            )
        },
    }


def plot(data):
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for name, spectrum in data["spectra"].items():
        ax.plot(spectrum["energy_eV"], spectrum["density"], label=name)
    ax.set(
        xlabel="Photon energy (eV)",
        ylabel="Photons / (incident electron · eV · sr)",
        title="Simulated 30 keV electrons → 100 nm HOPG",
        xlim=(500, 4_000),
        yscale="log",
        ylim=(1e-14, 1e-7),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


if __name__ == "__main__":
    plot(compute())
    plt.show()
