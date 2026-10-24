"""PyRITE #680: two-source planar quantum channeling demo for MoLab.

Requires the current `issue-680-planar-source` branch after source injection
(commit 220eccda or later), the fetched PyRITE reference tables, and plots extra.

Run (from the PyRITE repository):
  uv run --extra plots python molab_planar_quantum_demo.py
  uv run --extra plots python molab_planar_quantum_demo.py --bands

The production entrance adapter *only* admits centered one-dimensional Gaussian
states with lattice-averaged impact. This experiment is not yet independently
validated as a general quantum transport model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless cluster / MoLab: write PNGs, do not open windows.
import matplotlib.pyplot as plt
import numpy as np

import pyrite as pr


def make_result(waist_ang: float, args: argparse.Namespace) -> pr.Result:
    """Use the *production* conditional source injection + native transport."""
    state = pr.TransverseStateSpec.recipe("gaussian", dimension=1, waist_ang=waist_ang)
    beam = pr.Beam(
        energy_keV=args.energy_keV,
        transverse=pr.Spot(),
        longitudinal=pr.PointBunch(),
        state=state,
    )
    slab = pr.Slab(
        "silicon",
        thickness_ang=args.thickness_ang,
        tilt_deg=3.0,
        tilt_azim_deg=45.0,
        footprint=pr.Footprint(5.0, 5.0),
    )
    detector = pr.Detector(
        0.0,
        acceptance=pr.SensorAcceptance(distance_mm=1000.0, width_mm=3.0, height_mm=3.0),
    )
    physics = pr.Physics(
        channeling_model="planar",
        channeling_plane=(1, -1, 0),
        radiative_model="uncoupled",
    )
    numerics = pr.Numerics(
        cxr_trials=args.electrons,
        brem_trials=args.electrons,
        transport_core=args.core,
        channeling_basis_size=args.basis_size,
        channeling_bloch_cells=args.bloch_cells,
        channeling_dz_ang=args.dz_ang,
        energy_bins=pr.EnergyBins(
            cxr=np.arange(1000.0, 24000.0 + args.line_step_eV / 2, args.line_step_eV),
            brem=np.geomspace(1000.0, args.energy_keV * 1000.0, 80),
        ),
    )
    return pr.simulate(beam, slab, detector, physics=physics, numerics=numerics)


def plot_spectra(runs: dict[float, pr.Result], output: Path) -> None:
    """CR alone on the CXR grid, in photons/electron/eV/sr."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for waist, result in runs.items():
        if result.channeling is None:
            raise RuntimeError("No channeling spectrum: check Physics(channeling_model='planar')")
        spectrum = result.channeling
        ax.plot(
            np.asarray(spectrum.photon_energy_eV) / 1000.0,
            np.asarray(spectrum.density),
            label=f"Gaussian waist = {waist:g} Å",
            linewidth=1.8,
        )
    ax.set(
        xlabel="Photon energy (keV)",
        ylabel="CR photons / electron / eV / sr",
        title="Planar channeling radiation — quantum Gaussian entrance",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "quantum_cr_spectra.png", dpi=180)
    plt.close(fig)


def plot_fates(runs: dict[float, pr.Result], output: Path) -> None:
    """Recorded *sampled* fate counts, not deterministic CR probabilities."""
    widths = list(runs)
    counts = [dict(r.provenance["channeling"]["fates"]) for r in runs.values()]
    labels = list(dict.fromkeys(key for c in counts for key in c))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bar_width = 0.8 / len(widths)
    for i, (waist, c) in enumerate(zip(widths, counts, strict=True)):
        frac = [float(c.get(label, 0)) / sum(c.values()) for label in labels]
        ax.bar(
            x + (i - (len(widths) - 1) / 2) * bar_width,
            frac,
            width=bar_width,
            label=f"Waist {waist:g} Å",
        )
    ax.set(
        xticks=x,
        xticklabels=[name.replace("_", " ") for name in labels],
        xlabel="Handoff / channeling outcome",
        ylabel="Fraction of incident electrons",
        title="Sampled transport outcomes (small-N noise expected)",
        ylim=(0, 1),
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "quantum_handoff_fates.png", dpi=180)
    plt.close(fig)


def plot_bands(runs: dict[float, pr.Result], output: Path, args: argparse.Namespace) -> None:
    """Separate deterministic Bloch-density diagnostic (not MC track replay)."""
    from pyrite.montecarlo.channeling.handoff import case_plane_incidence
    from pyrite.planar_source import propagate_planar_source

    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig2, ax2 = plt.subplots(figsize=(9, 4.8))
    for waist, result in runs.items():
        case = result.case
        record = case["channeling"]
        setup = pr.QuantumSetup(
            region=pr.QuantumRegion(kind="planar", plane_hkl=tuple(record["plane_hkl"])),
            dynamics=pr.QuantumDynamics(kind="channeling"),
            numerics=pr.QuantumNumerics(
                basis_size=int(record["basis_size"]),
                bloch_cells=int(record["bloch_cells"]),
                step_ang=float(record["dz_ang"]),
            ),
        )
        state = result.scene.beam.state
        projected = setup.project_state(
            state,
            crystal=case["crystal"],
            kinetic_energy_eV=float(case["E0_keV"]) * 1000.0,
            incidence_rad=float(case_plane_incidence(case).effective_rad),
            impact="averaged",  # Matches the admitted production source.
            temperature_K=float(record["temperature_K"]),
            margin_eV=float(record["margin_eV"]),
        )
        # Effective depth is the beam path (slightly longer than slab thickness).
        depth = float(result.provenance["channeling"]["path_ang"])
        diag = propagate_planar_source(
            projected,
            depth_ang=depth,
            record_every=1,
            stopping_eV_per_ang=float(result.provenance["channeling"]["stopping_keV_per_ang"])
            * 1000.0,
        )
        z_um = np.asarray(diag.z_ang) * 1e-4
        ax.plot(z_um, diag.bound_population, label=f"Bound: {waist:g} Å")
        ax.plot(z_um, diag.trace, "--", label=f"Retained trace: {waist:g} Å")
        ax2.plot(z_um, diag.total_flux, label=f"Waist {waist:g} Å")
        bands = np.asarray(diag.band_populations)
        fig3, ax3 = plt.subplots(figsize=(9, 4.8))
        for band in range(min(4, bands.shape[1])):
            ax3.plot(z_um, bands[:, band], label=f"Bloch band {band}")
        ax3.plot(z_um, diag.bound_population, "k--", label="Total bound")
        ax3.set(
            xlabel="Depth along beam (μm)",
            ylabel="Population per incident electron",
            title=f"Resolved Bloch band populations — waist {waist:g} Å",
        )
        ax3.legend()
        ax3.grid(alpha=0.25)
        fig3.tight_layout()
        fig3.savefig(output / f"quantum_bands_{waist:g}A.png", dpi=180)
        plt.close(fig3)
        print(
            f"  {waist:g} Å diagnostic entry: retained={projected.retained_population:.4f}, "
            f"unretained={projected.entry_population:.4f}, "
            f"max balance defect={np.max(np.abs(diag.balance_defect)):.2e}"
        )
        np.savez_compressed(
            output / f"bloch_{waist:g}A.npz",
            depth_ang=diag.z_ang,
            band_populations=diag.band_populations,
            bound_population=diag.bound_population,
            trace=diag.trace,
            total_flux=diag.total_flux,
        )
    ax.set(
        xlabel="Depth along beam (μm)",
        ylabel="Probability per incident electron",
        title="Planar Bloch density-matrix populations (diagnostic)",
        ylim=(0, 1.05),
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "quantum_retained_bound_populations.png", dpi=180)
    plt.close(fig)
    ax2.set(
        xlabel="Depth along beam (μm)",
        ylabel="Cumulative nonretained population",
        title="Entry + dechanneling flux (diagnostic)",
        ylim=(0, 1.05),
    )
    ax2.legend()
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(output / "quantum_cumulative_flux.png", dpi=180)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("molab_quantum_output"))
    parser.add_argument("--energy-keV", type=float, default=17000.0)
    parser.add_argument("--thickness-ang", type=float, default=10000.0)
    parser.add_argument(
        "--electrons",
        type=int,
        default=16,
        help="Number of incident electrons for each estimator (quick test: 8)",
    )
    parser.add_argument(
        "--waists",
        nargs=2,
        type=float,
        default=(2.0, 20.0),
        metavar=("NARROW", "WIDE"),
        help="Two Gaussian waist sizes, Å",
    )
    parser.add_argument("--basis-size", type=int, default=8)
    parser.add_argument("--bloch-cells", type=int, default=4)
    parser.add_argument("--dz-ang", type=float, default=500.0)
    parser.add_argument(
        "--line-step-eV",
        type=float,
        default=40.0,
        help="CXR energy-grid spacing (use 5–10 eV for line details)",
    )
    parser.add_argument(
        "--core",
        choices=("auto", "per-electron", "cuda"),
        default="per-electron",
        help="Native *post-handoff* transport core",
    )
    parser.add_argument(
        "--bands",
        action="store_true",
        help="Additionally plot quantum populations vs depth (extra CPU work)",
    )
    args = parser.parse_args()
    if args.electrons < 1 or args.line_step_eV <= 0 or args.energy_keV <= 0:
        parser.error("electrons, line-step-eV and energy-keV must be positive")
    if len(set(args.waists)) != 2 or any(w <= 0 for w in args.waists):
        parser.error("provide two distinct positive Gaussian waists")
    args.output.mkdir(parents=True, exist_ok=True)

    runs: dict[float, pr.Result] = {}
    summary: dict[str, dict] = {}
    for waist in args.waists:
        print(f"\n>>> Quantum production case: Gaussian waist {waist:g} Å", flush=True)
        result = make_result(waist, args)
        runs[waist] = result
        cr = result.channeling
        assert cr is not None
        energy = np.asarray(cr.photon_energy_eV)
        density = np.asarray(cr.density)
        fate = dict(result.provenance["channeling"]["fates"])
        summary[str(waist)] = {
            "waist_ang": waist,
            "case_digest": result.provenance.identity_digest,
            "source_model": result.case["channeling_source"]["model"],
            "peak_cr_eV": float(energy[np.argmax(density)]),
            "peak_cr_density_per_electron_eV_sr": float(np.max(density)),
            "handoff_fates": fate,
            "stopping_keV_per_ang": float(result.provenance["channeling"]["stopping_keV_per_ang"]),
        }
        np.savez_compressed(
            args.output / f"spectra_{waist:g}A.npz",
            energy_eV=energy,
            channeling=density,
            cxr=result.cxr.incoherent.density,
            characteristic=(
                result.characteristic.density
                if result.characteristic is not None
                else np.zeros_like(density)
            ),
        )
        print(
            f"  CR peak: {summary[str(waist)]['peak_cr_eV']:.1f} eV; "
            f"peak density: {density.max():.3e} photons/(electron eV sr)"
        )
        print(f"  Fates: {fate}")
        print(f"  Case digest: {result.provenance.identity_digest}")

    plot_spectra(runs, args.output)
    plot_fates(runs, args.output)
    if args.bands:
        plot_bands(runs, args.output, args)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved plots, arrays and metadata in {args.output.resolve()}")


if __name__ == "__main__":
    main()
