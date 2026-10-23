"""EEDL and Bote--Salvat shell ionization against measured cross sections.

Evaluates both characteristic-scoring ionization models at every transcribed
thin-film measurement (``src/pyrite/validation/reference_data/
inner_shell_ionization``) and prints the geometric-mean ``model/measured``
ratio and one-sigma agreement per shell and overvoltage band, then the K-shell
near-threshold (U < 1.5) ratios per element. Report only: no gate.

    uv run python checks/inner_shell_experiment_comparison.py

Exit 0 on a report, 2 when the opt-in ``bote-salvat`` dataset is not installed
(``pyrite tables fetch bote-salvat``). Issue #92.

Validation: bote-salvat-ionization-backend
"""

import sys

import numpy as np

from pyrite.datasets import DatasetNotFoundError
from pyrite.validation import inner_shell_experiment as experiment


def main() -> int:
    k_shell = experiment.load_k_shell_measurements()
    try:
        k_model = experiment.model_cross_sections_b(k_shell)
    except DatasetNotFoundError as exc:
        print(f"SKIP: {exc}")
        return 2
    l_shell = experiment.load_l_subshell_measurements()
    l_model = experiment.model_cross_sections_b(l_shell)
    print("shell  U band      n  EEDL/exp  BS/exp  EEDL<=1sig  BS<=1sig")
    for measurements, model in ((k_shell, k_model), (l_shell, l_model)):
        for s in experiment.summarise(measurements, model):
            print(
                f"{s.shell:5s} [{s.band[0]:g}, {s.band[1]:g})".ljust(19)
                + f"{s.n:3d}  {s.eedl_ratio:8.3f}  {s.bote_salvat_ratio:6.3f}"
                + f"  {s.eedl_within:10.2f}  {s.bote_salvat_within:8.2f}"
            )
    print("\nK shell, U < 1.5, per element (geometric-mean model/measured)")
    near = [i for i, m in enumerate(k_shell) if m.overvoltage < 1.5]
    for symbol in dict.fromkeys(k_shell[i].symbol for i in near):
        rows = [i for i in near if k_shell[i].symbol == symbol]
        sigma = np.array([k_shell[i].sigma_b for i in rows])
        ratios = []
        for column in k_model[rows].T:
            positive = column > 0.0
            ratio = column[positive] / sigma[positive]
            ratios.append(np.exp(np.mean(np.log(ratio))) if ratio.size else np.nan)
        print(f"{symbol:3s} n={len(rows):2d}  EEDL {ratios[0]:.3f}  BS {ratios[1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
