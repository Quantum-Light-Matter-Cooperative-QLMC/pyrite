"""Plot recorded search selections and independent finalist validation; no transport."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = "docs/validation/check-records/pxr-optimization.json"
WORKFLOW = "checks/pxr_optimization.py"


def compute():
    """Reduce the published job-2511 record without inventing trial histories."""
    evidence = json.loads((ROOT / EVIDENCE).read_text())
    return {
        "repository_input_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (EVIDENCE, WORKFLOW)
        },
        "simulation_source": evidence["source"],
        "simulation_pyrite_version": evidence["pyrite_version"],
        "job_id": evidence["job_id"],
        "units": evidence["units"],
        "configuration": evidence["configuration"],
        "totals": evidence["totals"],
        "comparisons": evidence["comparisons"],
    }


def plot(data, view="finalists"):
    """Display selected noisy search values, independent means/SEM or grid sensitivity."""
    rows = data["comparisons"]
    labels = [row["label"].replace("-", " / ") for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    if view == "finalists":
        ax.plot(
            x,
            [row["selected_search_yield"] * 1e8 for row in rows],
            "x",
            label="Selected search value (32 histories)",
        )
        ax.errorbar(
            x,
            [row["validation_mean"] * 1e8 for row in rows],
            yerr=[row["validation_sem"] * 1e8 for row in rows],
            fmt="o",
            capsize=3,
            label="Independent finalist mean ± SEM (4 × 512)",
        )
        ax.set_ylabel("820–900 eV yield\n(10⁻⁸ photons / sr / incident electron)")
    elif view == "precision":
        ax.plot(x, [row["relative_sem"] * 100 for row in rows], "o", label="Relative SEM")
        ax.plot(
            x,
            [row["max_relative_paired_grid_difference"] * 100 for row in rows],
            "s",
            label="Max paired grid change / fine-grid mean",
        )
        config = data["configuration"]
        ax.axhline(config["relative_sem_limit"] * 100, ls="--", color="C0", label="SEM target")
        ax.axhline(
            config["relative_paired_grid_change_limit"] * 100,
            ls=":",
            color="C1",
            label="Grid target",
        )
        ax.set(yscale="log", ylabel="Relative uncertainty or grid change (%)")
    else:
        plt.close(fig)
        raise ValueError("view must be finalists or precision")
    ax.set_xticks(x, labels, rotation=40, ha="right")
    ax.set_xlabel("Method / search seed")
    ax.legend(fontsize=8)
    return fig


if __name__ == "__main__":
    output = compute()
    plot(output)
    plot(output, view="precision")
    plt.show()
