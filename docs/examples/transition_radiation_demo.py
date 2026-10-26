"""Render the packaged TR/CTR app's default calculation for the static guide.

This is documentation rendering support, not a separate scientific API example.
The maintained app adapters own the scene, solver and transport settings.
"""

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np


def compute():
    """Explicitly regenerate the app's analytic and fixed-count CPU outputs."""
    from pyrite.apps.transition_radiation import (
        DemoConfig,
        straight_track_demo,
        transported_demo,
    )

    config = DemoConfig()
    analytic = straight_track_demo(config)
    results, warnings, elapsed = transported_demo(config, transport_core="per-electron")
    cut = int(np.argmin(abs(analytic["energy"] - 500.0)))
    transport = {}
    for mode, result in results.items():
        transport[mode] = {
            "energy_eV": result.transition.photon_energy_eV.tolist(),
            "transition": result.transition.density.tolist(),
            "other_cxr": result.combined_cxr(include_transition=False).density.tolist(),
            "combined_cxr": result.combined_cxr().density.tolist(),
            "case": dict(result.case),
            "seed": result.case["seed"],
            "provenance": dict(result.provenance),
        }
    return {
        "config": asdict(config),
        "physical_electrons": config.population,
        "units": "ph/incident-electron/eV/sr",
        "analytic": {
            "energy_eV": analytic["energy"].tolist(),
            "film": analytic["spectrum"].tolist(),
            "ctr": analytic["ctr"].tolist(),
            "interface": analytic["interface"].tolist(),
            "form_factor": analytic["form_factor"].tolist(),
            "angle_deg": analytic["angles"].tolist(),
            "cut_energy_eV": float(analytic["energy"][cut]),
            "film_cut": analytic["film"][cut].tolist(),
            "ctr_cut": analytic["ctr_map"][cut].tolist(),
        },
        "transport": transport,
        "transport_runtime_s": elapsed,
        "warnings": warnings,
    }


def plot(data, *, view="analytic"):
    """Plot checked arrays, preserving native coordinates and signed estimates."""
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), layout="constrained")
    unit = "Photons / incident electron / eV / sr"
    analytic = data["analytic"]
    if view == "analytic":
        for key, label in (
            ("film", "Film TR"),
            ("ctr", "Film CTR expectation"),
            ("interface", "Semi-infinite interface TR"),
        ):
            axes[0].semilogy(analytic["energy_eV"], analytic[key], label=label)
        axes[0].set(xlabel="Photon energy (eV)", title="Straight tracks · 150° from beam")
        for key, label in (("film_cut", "Film TR"), ("ctr_cut", "Film CTR expectation")):
            axes[1].semilogy(analytic["angle_deg"], analytic[key], label=label)
        axes[1].set(
            xlabel="Signed angle from beam (deg)",
            title=f"Angular cut · {analytic['cut_energy_eV']:g} eV",
        )
    elif view == "transport":
        for ax, mode, expected in zip(axes, ("history", "bunch"), ("film", "ctr"), strict=True):
            record = data["transport"][mode]
            for key, label in (
                ("transition", "TR only" if mode == "history" else "CTR only (signed)"),
                ("other_cxr", "Other CXR components"),
                ("combined_cxr", "Combined CXR"),
            ):
                ax.plot(record["energy_eV"], record[key], label=label)
            ax.plot(
                analytic["energy_eV"], analytic[expected], "k:", label="Straight-track reference"
            )
            ax.set_yscale("symlog", linthresh=1e-14)
            ax.set(xlabel="Photon energy (eV)", title=f"Transport · {mode} · 150° from beam")
    else:
        raise ValueError(f"unknown view: {view}")
    for ax in axes:
        ax.set_ylabel(unit)
        ax.legend(fontsize=9)
    return fig


if __name__ == "__main__":
    data = compute()
    plot(data)
    plot(data, view="transport")
    plt.show()
