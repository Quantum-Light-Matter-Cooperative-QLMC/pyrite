"""Render #278 benchmark JSON and spectra into a contributor comparison page."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("evidence", type=Path)
    ap.add_argument("page", type=Path)
    args = ap.parse_args()
    records = [json.loads(p.read_text()) for p in sorted(args.evidence.glob("*.json"))]
    assets = args.page.parent / "figures"
    assets.mkdir(exist_ok=True)
    lines = [
        "# Continuum grid benchmark comparison",
        "",
        "Paired CUDA runner measurements for issue #278. Automatic case-local photon nodes are compared with an explicit uniform 25 eV continuum axis. These are numerical and performance comparisons, not independent validation of the radiation model.",
        "",
        "## Protocol",
        "",
        "HOPG, 45° polar / 180° azimuth, 20 electrons, seed 1; thicknesses 0.1 and 1 mm; incident energies 0.3, 3, 30 and 100 MeV. Each arm runs in a separate process after a one-electron transport/scoring warm-up. Two repeats reverse arm order. Both use the same explicit 600,000-node line axis from 0.1 to 100 keV. Timings include transport and spectrum reduction; case/grid construction and detector response are excluded. These bounded probes do not predict 20,000-electron runtime. The upper beam energy follows the documented transport benchmark range; line-model validity remains separate.",
        "",
        "Job 3774, code 6584b1b11, one RTX 5080 GPU, 4 CPU threads, 24 GB requested host memory. Memory below is production-reported peak host RSS; GPU allocator peaks are retained in the linked JSON. Warm-up allocator history may contribute to peaks. Two repeats provide a range, not a confidence interval.",
        "",
        "## Runtime and spectral agreement",
        "",
        "| Beam (MeV) | Thickness (mm) | Uniform total range (s) | Automatic total range (s) | Mean speedup | Relative yield difference |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    data = args.page.parent / "data"
    data.mkdir(exist_ok=True)
    (data / "continuum-grid-2026-11-03.json").write_text(json.dumps(records, indent=2))
    lines[2:2] = [
        "",
        "[Machine-readable measurements](data/continuum-grid-2026-11-03.json). See the [beam-energy validity map](../../physics/capabilities.md#beam-energy-validity) for model scope.",
        "",
    ]
    memory = [
        "",
        "## Memory and recovery",
        "",
        "| Beam (MeV) | Thickness (mm) | Uniform host peak (MiB) | Automatic host peak (MiB) | Uniform GPU pool peak (MiB) | Automatic GPU pool peak (MiB) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    plots = []
    for energy in (300, 3000, 30000, 100000):
        for thickness in (1000000, 10000000):
            group = [
                r for r in records if r["energy_keV"] == energy and r["thickness_ang"] == thickness
            ]
            auto = [r for r in group if r["grid"] == "automatic"]
            uniform = [r for r in group if r["grid"] == "uniform25"]
            if len(auto) != 2 or len(uniform) != 2:
                raise ValueError(f"incomplete case {energy}, {thickness}")
            memory.append(
                f"| {energy / 1000:g} | {thickness / 1e7:g} | {max(r['timing']['_host_rss_peak_mib'] for r in uniform):.0f} | {max(r['timing']['_host_rss_peak_mib'] for r in auto):.0f} | {max(r['timing']['_cupy_pool_peak_mib'] for r in uniform):.0f} | {max(r['timing']['_cupy_pool_peak_mib'] for r in auto):.0f} |"
            )
            at = [r["seconds"]["total"] for r in auto]
            ut = [r["seconds"]["total"] for r in uniform]
            delta = abs(
                auto[0]["brem_integral"]["brem_wide"] / uniform[0]["brem_integral"]["brem_wide"] - 1
            )
            lines.append(
                f"| {energy / 1000:g} | {thickness / 1e7:g} | {min(ut):.3f}–{max(ut):.3f} | {min(at):.3f}–{max(at):.3f} | {np.mean(ut) / np.mean(at):.2f}× | {delta:.2e} |"
            )
            prefix = f"{energy}-{thickness}-0-"
            with (
                np.load(args.evidence / (prefix + "automatic.json.npz")) as a,
                np.load(args.evidence / (prefix + "uniform25.json.npz")) as u,
            ):
                x, y = a["energy_eV"], a["density"]
                ux, uy = u["energy_eV"], u["density"]
                interp = np.interp(x, ux, uy)
                mask = (x >= ux[0]) & (x <= ux[-1]) & (interp > np.max(interp) * 1e-8)
                fig, axes = plt.subplots(
                    2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
                )
                axes[0].loglog(ux / 1000, uy, label="Uniform 25 eV", lw=2)
                axes[0].loglog(x / 1000, y, "--", label="Automatic", lw=1.2)
                axes[0].set_ylabel("Density (photons/eV)")
                axes[0].legend()
                axes[0].set_title(
                    f"HOPG {thickness / 1e7:g} mm · {energy / 1000:g} MeV · 20 electrons"
                )
                axes[1].semilogx(x[mask] / 1000, (y[mask] / interp[mask] - 1) * 100)
                axes[1].axhline(0, color="gray", lw=0.5)
                axes[1].set_ylabel("Difference (%)")
                axes[1].set_xlabel("Photon energy (keV)")
                fig.tight_layout()
                name = f"continuum-{energy}-{thickness}.png"
                fig.savefig(assets / name, dpi=150)
                plt.close(fig)
                plots += [
                    f"### {energy / 1000:g} MeV, {thickness / 1e7:g} mm",
                    "",
                    f"![Continuum spectrum overlay and relative difference](figures/{name})",
                    "",
                ]
    lines += [
        "",
        "Spectral differences interpolate the uniform density onto automatic coordinates only within the shared band; bins below 1e-8 of the maximum reference density are omitted from the ratio. The automatic floor and 25 eV uniform floor differ, so integrated yield differences also include the low-energy band difference. Neither grid is an independent truth reference. The runner emits kinematic PXR/CBS warnings for the high-energy geometry: first-order line spectra may require dynamical diffraction. The plots here compare the bremsstrahlung continuum; the shared line workload is retained only in the runtime comparison.",
        "",
        "## Spectra",
        "",
    ] + plots
    lines += memory
    retries = sum(r["timing"].get("_gpu_oom_retries", 0) for r in records)
    lines += ["", f"Total recorded GPU OOM retries across the matrix: {retries}.", ""]
    args.page.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
