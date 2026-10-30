"""Offline smoke test for the public physical-units fundamentals tutorial."""

import importlib.util
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"


def test_units_notebook_cpu_smoke(monkeypatch):
    """Exercise quantity normalization and coupled density conversion without transport."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    path = DOCS / "notebooks" / "00a-units-and-quantities.py"
    spec = importlib.util.spec_from_file_location("units_notebook", path)
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    _, definitions = notebook.app.run()

    assert definitions["beam"].energy_keV == pytest.approx(30)
    assert definitions["target"].thickness_ang == pytest.approx(1_000)
    assert definitions["detector"].observation_angle_deg == pytest.approx(135)
    assert definitions["average_current"].to("nA").magnitude == pytest.approx(50)
    assert definitions["integral_per_eV"] == pytest.approx(definitions["integral_per_keV"])
    assert definitions["in_keV"].count_basis == "photons_per_incident_electron"
    import matplotlib.pyplot as plt

    plt.close("all")
