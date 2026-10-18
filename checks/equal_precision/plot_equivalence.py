"""Render fixed-design equivalence intervals relative to their frozen margins."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    if not data["complete_fixed_design"]:
        raise ValueError("complete fixed-design results required for an equivalence plot")
    records = data["comparisons"]
    scenes = list(dict.fromkeys(r["scene"] for r in records))
    names = list(dict.fromkeys(r["observable"] for r in records))
    labels = {
        "transmitted": "Transmission",
        "backscattered": "Backscatter",
        "side_exited": "Side exit",
        "stopped": "Stopped",
        "transmitted_energy_keV_per_primary": "Transmitted energy",
        "photons_ge10kev": "Photons ≥10 keV",
        "photons_10_50kev": "Photons 10–50 keV",
        "photons_50_100kev": "Photons 50–100 keV",
        "photons_ge100kev": "Photons ≥100 keV",
        "photon_energy_keV_per_primary": "Photon energy",
        "continuous_loss_keV_per_primary": "Continuous loss",
        "terminal_energy_keV_per_primary": "Terminal energy",
        **{f"transmitted_mu_bin_{i}": f"Transmitted cosine bin {i}" for i in range(4)},
    }
    colors = {"passed": "#2474a8", "unresolved": "#b87319", "failed": "#b83e45"}
    fig, axes = plt.subplots(
        1, len(scenes), figsize=(12, 8), sharey=True, layout="constrained", squeeze=False
    )
    limit = max(1.15, max(abs(v) / r["margin"] for r in records for v in r["difference_ci"]) * 1.08)
    for ax, scene in zip(axes.ravel(), scenes, strict=True):
        subset = {r["observable"]: r for r in records if r["scene"] == scene}
        ax.axvspan(-1, 1, color="#e9f1e5", zorder=0)
        ax.axvline(0, color="#98a0a8", lw=0.7)
        for i, name in enumerate(names):
            r = subset[name]
            margin = r["margin"]
            center = r["difference"] / margin
            low, high = np.asarray(r["difference_ci"]) / margin
            ax.errorbar(
                center,
                i,
                xerr=[[max(0, center - low)], [max(0, high - center)]],
                fmt="o",
                color=colors[r["status"]],
                markersize=4,
                capsize=3,
                lw=1.5,
            )
        ax.set_yticks(range(len(names)), [labels.get(n, n) for n in names], fontsize=10)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_title("Si 300 keV / 100 µm" if scene.startswith("si-") else "W 300 keV / 10 µm")
        ax.set_xlabel("Mean difference / frozen equivalence margin")
        ax.grid(axis="x", alpha=0.2)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle("Simultaneous 95% paired mean-difference intervals", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
