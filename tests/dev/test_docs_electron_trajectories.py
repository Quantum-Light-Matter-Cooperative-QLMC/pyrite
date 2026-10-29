"""Public trajectory tutorial smoke test and normalized-transport assertions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK = Path(__file__).parents[2] / "docs/notebooks/00-electron-trajectories.py"


@pytest.mark.slow
def test_electron_trajectories_cpu_smoke(monkeypatch):
    """Capture-facing notebook must execute from the public API on CPU.

    The analytic SR model needs no external ELSEPA tables; avoid a table fetch
    in this offline test while retaining that setup step in the tutorial.
    """
    monkeypatch.setenv("PYRITE_MC_BACKEND", "cpu")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import pyrite

    monkeypatch.setattr(pyrite, "fetch_tables", lambda: [])
    # Smoke-test a bounded copy; the published capture retains the full budget.
    source = NOTEBOOK.read_text()
    assert source.count("histories = 16_384") == 1
    bounded_notebook = tmp_path / NOTEBOOK.name
    bounded_notebook.write_text(source.replace("histories = 16_384", "histories = 64"))
    spec = importlib.util.spec_from_file_location("electron_trajectory_notebook", bounded_notebook)
    notebook = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(notebook)
    _, values = notebook.app.run()

    result = values["transport_result"]
    n = values["histories"]
    rows = result["electron_id"]
    assert n == 16_384
    assert np.unique(rows).size == n
    assert np.asarray(result["r_mid"]).shape[1] == 3
    assert np.all(np.asarray(result["L_ang"]) > 0)
    assert np.all(np.asarray(result["E_end_keV"]) <= result["E_start_keV"])
    assert len(values["last_rows"]) == n
    assert sum(values["fates"].values()) == n

    depths = values["depth_loss_per_electron_nm"]
    assert depths.shape == (40,)
    assert np.all(np.isfinite(depths)) and np.all(depths >= 0)
    assert depths.sum() > 0
    assert values["escaped_top_keV"].size == result["n_backscattered"]
    assert values["escaped_bottom_keV"].size == result["n_transmitted"]

    import matplotlib.pyplot as plt

    plt.close("all")
