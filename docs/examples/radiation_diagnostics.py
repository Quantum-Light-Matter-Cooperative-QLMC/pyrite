"""Render the optical app's synthetic far-IR duration exercise.

Documentation support only: the maintained app adapter owns the source,
forward model and fit. Its imports are not a stable scientific API example.
"""

import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np


def compute():
    """Explicitly regenerate two inexpensive, deterministic app calculations."""
    from pyrite.apps.optical_diagnostics import inference_demo

    bands = {}
    settings = dict(duration_fs=200.0, high_energy_eV=0.025, mirror_mm=1.2, nodes=61, noise_uV=15.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name, low in (("turnover_in_band", 0.0005), ("turnover_below_band", 0.02)):
            scan, fit, interferogram = inference_demo(low_energy_eV=low, **settings)
            bands[name] = {
                "low_energy_eV": low,
                "scan": asdict(scan),
                "fit": asdict(fit),
                "instrument_provenance": interferogram.metadata,
            }
    return {
        "settings": settings,
        "units": {"signal": "isolated-pulse V", "mirror_displacement": "m", "duration": "fs RMS"},
        "bands": bands,
        "warnings": list(dict.fromkeys(str(item.message) for item in caught)),
    }


def plot(data, *, view="signal"):
    """Plot native samples and the app's profiled objective, without fitting."""
    bands = data["bands"]
    if view == "signal":
        record = bands["turnover_in_band"]
        scan, fit = record["scan"], record["fit"]
        mirror = np.asarray(scan["mirror_displacement_m"]) * 1e3
        voltage = np.asarray(scan["voltage_v_ports"])
        predicted = np.asarray(fit["predicted_voltage_v_ports"])
        residual = np.asarray(fit["residual_v_ports"])
        fig, axes = plt.subplots(2, 1, figsize=(7, 5.8), sharex=True, layout="constrained")
        for port in range(2):
            axes[0].plot(mirror, voltage[:, port] * 1e3, "o", ms=3, label=f"Port {port + 1}")
            axes[0].plot(mirror, predicted[:, port] * 1e3, color=f"C{port}")
            axes[1].plot(
                mirror, residual[:, port] * 1e6, color=f"C{port}", label=f"Port {port + 1}"
            )
        axes[0].set(
            ylabel="Isolated-pulse voltage (mV)",
            title="Synthetic 200 fs RMS Gaussian · 0.0005–0.025 eV",
        )
        axes[0].legend()
        axes[1].axhline(0, color="0.6", lw=0.7)
        axes[1].set(
            xlabel="Mechanical mirror displacement (mm); OPD = 2 × displacement",
            ylabel="Expectation − fit (µV)",
            title="Noise-free expectation; declared RMS error 15 µV per port",
        )
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    elif view == "profile":
        fig, ax = plt.subplots(figsize=(7, 4.1), layout="constrained")
        for record in bands.values():
            profile = record["fit"]["metadata"]["diagnostic_profile"]
            label = f"{record['low_energy_eV']:g}–{data['settings']['high_energy_eV']:g} eV"
            ax.semilogx(profile["sigma_fs"], profile["delta_chi2"], ".-", ms=3, label=label)
        ax.axhline(3.841, color="0.35", ls="--", label="Conditional 95% threshold")
        ax.axvline(data["settings"]["duration_fs"], color="0.6", ls=":", label="Input 200 fs")
        ax.set_yscale("symlog", linthresh=5)
        ax.set(
            xlabel="Gaussian RMS duration (fs)",
            ylabel="Profile Δχ² (symmetric log scale)",
            title="Conditional Gaussian duration profiles",
        )
        minimum = min(
            min(record["fit"]["metadata"]["diagnostic_profile"]["delta_chi2"])
            for record in bands.values()
        )
        ax.set_ylim(bottom=min(0, minimum) - 0.25)
        ax.set_yticks([0, 3.841, 10, 100, 1000, 10000], ["0", "3.84", "10", "100", "1000", "10000"])
        ax.legend(fontsize=9, loc="center right")
    else:
        raise ValueError(f"unknown view: {view}")
    return fig


if __name__ == "__main__":
    data = compute()
    plot(data)
    plot(data, view="profile")
    plt.show()
