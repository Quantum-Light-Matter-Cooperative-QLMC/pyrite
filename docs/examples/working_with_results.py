"""Compare source components in one saved quickstart checkpoint; no transport."""

# tutorial-start
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from pyrite.results import records
from pyrite.results.radiation import radiation_band_metrics
from pyrite.storage import load_checkpoint


def analyze(checkpoint):
    checkpoint = Path(checkpoint)
    saved = records(load_checkpoint(checkpoint.name, checkpoint.parent))
    if len(saved) != 1:
        raise ValueError("choose the quickstart checkpoint containing exactly one case")
    record = saved[0]
    band = radiation_band_metrics(record, band_eV=(100.0, 1_000.0))
    components = {
        "PXR + CBS": (record["E_grid"], record["spec"]),
        "Characteristic": (record["E_grid"], record["spec_characteristic"]),
        "Bremsstrahlung": (record["E_grid_brem"], record["brem_wide"]),
    }
    return {
        "case": record["case"],
        "adaptive_sampling": record["adaptive_sampling"],
        "spectra": {
            name: {"energy_eV": energy.tolist(), "density": density.tolist()}
            for name, (energy, density) in components.items()
        },
        "band_eV": band["band_eV"],
        # This API integrates line-grid components; it does not include brem.
        "band_yields": {name: band["yields"][name] for name in ("line", "characteristic")},
    }


def plot(data):
    fig, (ax, band_ax) = plt.subplots(
        1, 2, figsize=(9, 3.8), width_ratios=(2, 1), layout="constrained"
    )
    for name, spectrum in data["spectra"].items():
        ax.plot(spectrum["energy_eV"], spectrum["density"], label=name)
    ax.axvspan(*data["band_eV"], color="grey", alpha=0.12, label="Integrated band")
    ax.set(
        xlabel="Photon energy (eV)",
        ylabel="Photons / (incident electron · eV · sr)",
        xlim=(100, 3_500),
        ylim=(1e-13, 1e-3),
        yscale="log",
        title=f"Saved {data['case']['E0_keV']:g} keV {data['case']['crystal'].upper()} source components",
    )
    ax.legend(fontsize=8)
    band_ax.bar(
        ["PXR + CBS", "Characteristic"], list(data["band_yields"].values()), color=["C0", "C1"]
    )
    band_ax.set(
        ylabel="Photons / (incident electron · sr)",
        yscale="log",
        title="100–1,000 eV band (no brem)",
        ylim=(1e-7, 1e-3),
    )
    for axis in (ax, band_ax):
        axis.grid(axis="y", alpha=0.25)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="quickstart checkpoint directory")
    plot(analyze(parser.parse_args().checkpoint))
    plt.show()
# tutorial-end


def compute(checkpoint):
    """Retain input provenance when regenerating the published output."""
    import hashlib

    from pyrite.storage import checkpoint_manifest

    checkpoint = Path(checkpoint)
    data = analyze(checkpoint)
    case = data.pop("case")
    sampling = data.pop("adaptive_sampling")
    return {
        **data,
        "case": case,
        "checkpoint_stem": checkpoint.name,
        "seed": case["seed"],
        "provenance": {
            "dataset_identity": checkpoint_manifest(checkpoint.name, checkpoint.parent)[
                "dataset_identity"
            ],
            "case": case,
            "adaptive_sampling": sampling,
            "components_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(checkpoint.glob("*.h5"))
            },
        },
    }
