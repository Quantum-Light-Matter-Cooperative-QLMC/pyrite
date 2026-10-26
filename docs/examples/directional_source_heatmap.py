"""Seeded CPU source map; illustrative sampling, not a converged prediction."""

# tutorial-start
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from pyrite import Beam, Numerics
from pyrite.beams import PointBunch, PointSpot
from pyrite.config import CxrReflections, FixedSampling
from pyrite.geometry import Footprint, Slab
from pyrite.radiation import directional_source


def calculate():
    beam = Beam(energy_keV=30.0, transverse=PointSpot(), longitudinal=PointBunch())
    target = Slab("silicon", thickness_ang=1_000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0))
    return directional_source(
        beam,
        target,
        angle_deg=np.linspace(-180, 180, 48, endpoint=False),
        energy_eV=np.linspace(100, 2_000, 513),
        numerics=Numerics(
            sampling=FixedSampling(cxr_electrons=64, brem_electrons=64),
            cxr_reflections=CxrReflections(n_families=1),
        ),
        seed=1,
    )


def plot(data):
    angles, energy, density = (np.asarray(data[k]) for k in ("angle_deg", "energy_eV", "density"))
    fig, (heatmap, spectra) = plt.subplots(2, 1, figsize=(7, 6.5), layout="constrained")
    floor, ceiling = density.max() * 1e-5, density.max()
    # Close the periodic slice for drawing only; zero stays at the color floor.
    closed = np.maximum(np.vstack((density, density[0])).T, floor)
    image = heatmap.pcolormesh(
        np.append(angles, 180),
        energy,
        closed,
        norm=LogNorm(floor, ceiling),
        shading="nearest",
        rasterized=True,
    )
    fig.colorbar(image, ax=heatmap, label="Photons / (incident electron · eV · sr)")
    heatmap.set(xlabel="Lab observation angle (°)", ylabel="Photon energy (eV)", xlim=(-180, 180))
    heatmap.set_title("Simulated PXR + CBS: 30 keV electrons → 100 nm silicon")
    for angle in (0, 90, 135):
        row = np.argmin(abs(angles - angle))
        spectra.plot(energy, density[row], label=f"{angles[row]:g}°")
    spectra.set(
        xlabel="Photon energy (eV)",
        ylabel="Photons / (incident electron · eV · sr)",
        yscale="log",
        ylim=(floor, ceiling),
        xlim=(energy[0], energy[-1]),
    )
    spectra.legend(title="Lab observation angle", ncols=3)
    spectra.grid(alpha=0.25)
    return fig


if __name__ == "__main__":
    source = calculate()
    plot({k: getattr(source, k) for k in ("angle_deg", "energy_eV", "density")})
    plt.show()
# tutorial-end


def compute():
    """Retain the actual runtime warnings alongside the cached densities."""
    import warnings

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        source = calculate()
    return {
        "angle_deg": source.angle_deg.tolist(),
        "energy_eV": source.energy_eV.tolist(),
        "density": source.density.tolist(),
        "supported": source.supported.tolist(),
        "units": source.units,
        "component": source.component,
        "seed": source.case["seed"],
        "case": dict(source.case),
        "provenance": dict(source.provenance),
        "warnings": sorted({f"{w.category.__name__}: {w.message}" for w in recorded}),
    }
