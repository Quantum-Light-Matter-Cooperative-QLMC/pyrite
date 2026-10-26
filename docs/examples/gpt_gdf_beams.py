"""SYNTHETIC DEMONSTRATION: GPT-format electrons, not facility or GPT output."""

# tutorial-start
from pathlib import Path

import easygdf
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, electron_mass, elementary_charge

from pyrite import Beam, Numerics, simulate
from pyrite.config import CxrReflections, FixedSampling
from pyrite.detectors import Detector
from pyrite.geometry import Footprint, Slab


def calculate(path):
    # This fixture has one time snapshot, at 1 ns and lab z = 0.1 m.
    blocks = easygdf.load(str(path))["blocks"]
    fields = {b["name"]: b["value"] for b in blocks[0]["children"]}
    beam = Beam.from_gdf(
        str(path),
        time_s=1e-9,
        z_origin_m=0.1,
        energy_keV=30.0,  # Scalar scene placeholder; the file supplies each energy.
        normalization="gdf_charge",
        rep_rate_hz=1_000_000.0,
    )
    result = simulate(
        beam,
        Slab("hopg", thickness_ang=1_000.0, tilt_deg=30.0, footprint=Footprint(1.0, 1.0)),
        Detector(observation_angle_deg=135.0),
        numerics=Numerics(
            sampling=FixedSampling(cxr_electrons=64, brem_electrons=64),
            cxr_reflections=CxrReflections(n_families=2),
        ),
    )
    return fields, result


# tutorial-end


def plot(data):
    x, slope, energy, probability = (
        np.asarray(data[k]) for k in ("x_um", "slope_mrad", "energy_keV", "probabilities")
    )
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5), layout="constrained")
    points = axes[0].scatter(x, slope, c=energy, s=2000 * probability, cmap="viridis", alpha=0.85)
    axes[0].set(xlabel="Lab snapshot x (µm)", ylabel="Bx / Bz (mrad)")
    axes[0].set_title("Synthetic input: marker area ∝ sampling probability")
    fig.colorbar(points, ax=axes[0], label="Electron energy (keV)")
    axes[1].hist(energy, bins=np.linspace(24, 36, 13), weights=probability, edgecolor="white")
    axes[1].set(xlabel="Electron energy (keV)", ylabel="Weighted probability / bin")
    for name, spectrum in data["spectra"].items():
        axes[2].plot(spectrum["energy_eV"], spectrum["density"], label=name)
    axes[2].set(
        xlabel="Photon energy (eV)",
        ylabel="Photons / (incident electron · eV · sr)",
        xlim=(500, 4_000),
        yscale="log",
        ylim=(1e-14, 1e-7),
    )
    axes[2].set_title("Simulated source: 100 nm HOPG, lab observation 135°")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    return fig


def fixture_path():
    """Support both repository layout and downloads saved beside the script."""
    beside = Path(__file__).with_name("synthetic_beam.gdf")
    return beside if beside.exists() else beside.parent / "data" / beside.name


def make_fixture(path):
    """Write an original deterministic synthetic input; see synthetic_beam.md."""
    rng = np.random.default_rng(42)
    count = 96
    x, y = rng.normal(0, 20e-6, (2, count))
    slope_x = 100 * x + rng.normal(0, 0.001, count)
    slope_y = rng.normal(0, 0.001, count)
    energy_keV = 30 + 0.05 * x * 1e6 + rng.normal(0, 0.4, count)
    gamma = 1 + energy_keV * 1000 * elementary_charge / (electron_mass * c**2)
    speed = np.sqrt(1 - gamma**-2)
    bz = speed / np.sqrt(1 + slope_x**2 + slope_y**2)
    # Exactly 1 pC represented charge, with nonuniform record probabilities.
    weight = np.linspace(1, 3, count)
    weight *= 1e-12 / (elementary_charge * weight.sum())
    fields = {
        "x": x,
        "y": y,
        "z": np.full(count, 0.1),
        "Bx": bz * slope_x,
        "By": bz * slope_y,
        "Bz": bz,
        "m": np.full(count, electron_mass),
        "q": np.full(count, -elementary_charge),
        "G": gamma,
        "nmacro": weight,
    }
    easygdf.save(
        str(path),
        blocks=[
            {
                "name": "time",
                "value": 1e-9,
                "children": [{"name": k, "value": v} for k, v in fields.items()],
            }
        ],
        creation_time=0,
        creator="GPT",  # Format admission marker; this is NOT a GPT-generated beam.
        creator_version=(0, 0),
    )


def compute(input_path=None):
    """Execute explicitly; preserve input identity, resolved seed and model provenance."""
    import hashlib
    import os
    import warnings

    path = fixture_path() if input_path is None else Path(input_path)
    # Relative input paths keep portable tutorial provenance free of home directories.
    path = Path(os.path.relpath(path))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fields, result = calculate(path)
    beta2 = sum(fields[k] ** 2 for k in ("Bx", "By", "Bz"))
    energy = (1 / np.sqrt(1 - beta2) - 1) * electron_mass * c**2 / elementary_charge / 1000
    probability = fields["nmacro"] / fields["nmacro"].sum()
    return {
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
        "input_kind": "SYNTHETIC DEMONSTRATION; not experimental or GPT-generated",
        "x_um": (fields["x"] * 1e6).tolist(),
        "slope_mrad": (fields["Bx"] / fields["Bz"] * 1000).tolist(),
        "energy_keV": energy.tolist(),
        "probabilities": probability.tolist(),
        "seed": result.case["seed"],
        "case": dict(result.case),
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
        "warnings": sorted({f"{w.category.__name__}: {w.message}" for w in recorded}),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=fixture_path())
    parser.add_argument(
        "--make-fixture", action="store_true", help="write synthetic input and exit"
    )
    args = parser.parse_args()
    if args.make_fixture:
        make_fixture(args.input)
    else:
        plot(compute(args.input))
        plt.show()
