"""Frozen #510 matrix expansion, range pinning and observable applicability."""

import json
import runpy
from pathlib import Path

import pytest

from pyrite.perf.benchmark_matrix import (
    case_id,
    csda_range_g_cm2,
    expand_matrix,
    gaussian_fwhm_mm,
    observable_applicability,
    parse_range_table,
    waist_normalized_emittance_mm_mrad,
)

ROOT = Path(__file__).resolve().parents[2] / "checks/equal_precision"
TABLE = "kinetic_energy_MeV,csda_range_g_cm2\n0.1,0.01\n1.0,0.5\n10.0,5.0\n"


def test_range_table_is_hashed_and_log_log_interpolated():
    table = parse_range_table(TABLE)
    assert len(table["sha256"]) == 64
    assert csda_range_g_cm2(table, 1000.0) == pytest.approx(0.5)
    # Power-law segment 1-10 MeV: R ∝ E, so 3.1623 MeV -> sqrt(10) * 0.5.
    assert csda_range_g_cm2(table, 1000.0 * 10**0.5) == pytest.approx(0.5 * 10**0.5)
    with pytest.raises(ValueError, match="outside"):
        csda_range_g_cm2(table, 20000.0)


@pytest.mark.parametrize(
    "text",
    [
        "energy,range\n0.1,0.01\n1,0.5\n",
        "kinetic_energy_MeV,csda_range_g_cm2\n0.1,0.01\n",
        "kinetic_energy_MeV,csda_range_g_cm2\n1,0.5\n0.1,0.01\n",
        "kinetic_energy_MeV,csda_range_g_cm2\n0.1,-1\n1,0.5\n",
    ],
)
def test_invalid_range_tables_refused(text):
    with pytest.raises(ValueError):
        parse_range_table(text)


def test_endpoint_bands_and_stopping_regime_applicability():
    low = observable_applicability({"energy_keV": 100, "t_over_r": 3})
    assert low["photons_ge100kev"] == "excluded"
    assert low["photons_50_100kev"] == "headline"
    assert low["transmitted_energy_keV_per_primary"] == "diagnostic"
    thin = observable_applicability({"energy_keV": 800, "t_over_r": 0.03, "width_mm": 1.0})
    assert thin["transmitted_energy_keV_per_primary"] == "headline"
    assert thin["side_exited"] == "headline"


def test_beam_moment_conversions():
    assert gaussian_fwhm_mm(0.1) == pytest.approx(0.235482, rel=1e-5)
    eps_n, beta = waist_normalized_emittance_mm_mrad(0.001, 5.0, 800.0)
    assert beta == pytest.approx(2.0e-4)
    gamma = 1 + 800 / 510.99895
    assert eps_n == pytest.approx(0.005 * (gamma**2 - 1) ** 0.5)


def test_frozen_spec_expands_to_core_product_with_pending_thickness():
    spec = json.loads((ROOT / "matrix-spec.json").read_text())
    cases = expand_matrix(spec)
    core = [c for c in cases if c["variant"] == "core"]
    assert len(core) == 18
    assert {c["status"] for c in core} == {"pending-range-source"}
    assert all(c["thickness_um"] is None for c in core)
    # 5 MeV is capped at t/R = 1: tr3 would be 38.5 mm of silicon in a 10 mm prism.
    assert {c["t_over_r"] for c in core if c["energy_keV"] == 5000} == {0.03, 0.3, 1}
    assert {c["t_over_r"] for c in core if c["energy_keV"] == 800} == {0.03, 0.3, 3}
    anchors = [c for c in cases if c["variant"] == "anchor"]
    assert {c["status"] for c in anchors} == {"frozen-absolute"}
    assert all(not c["physics"]["pxr"] and not c["physics"]["cbs"] for c in cases)
    assert len({c["id"] for c in cases}) == len(cases)


def test_pinned_range_freezes_one_absolute_thickness_for_every_engine():
    spec = json.loads((ROOT / "matrix-spec.json").read_text())
    table = parse_range_table(TABLE)
    cases = {c["id"]: c for c in expand_matrix(spec, {"silicon": table})}
    case = cases["silicon-800kev-tr0.3-0deg"]
    expected = 0.3 * csda_range_g_cm2(table, 800) / case["density_g_cm3"] * 1e4
    assert case["thickness_um"] == pytest.approx(expected)
    assert case["range_source"] == table["sha256"]
    # Oblique cases keep the same surface-normal thickness.
    assert cases["silicon-800kev-tr0.3-75deg-oblique"]["thickness_um"] == case["thickness_um"]
    assert cases["tungsten-800kev-tr0.3-0deg"]["status"] == "pending-range-source"


def test_duplicate_case_ids_refused():
    spec = json.loads((ROOT / "matrix-spec.json").read_text())
    spec["extensions"].append(dict(spec["extensions"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        expand_matrix(spec)


def test_case_id_is_stable():
    assert case_id({"material": "W", "energy_keV": 800, "t_over_r": 3, "angle_deg": 75}) == (
        "w-800kev-tr3-75deg"
    )


def test_estar_text_columns_and_committed_matrix_regenerates(tmp_path, monkeypatch):
    module = runpy.run_path(str(ROOT / "build_matrix.py"))
    estar = "ESTAR: SILICON\nKinetic Collision Radiative Total CSDA\n"
    estar += "1.000E-01 3.0 0.1 3.1 1.000E-02 0.1 0.2\n"
    estar += "1.000E+00 1.5 0.1 1.6 5.000E-01 0.1 0.2\n"
    table = parse_range_table(module["estar_to_csv"](estar))
    assert csda_range_g_cm2(table, 1000.0) == pytest.approx(0.5)
    # A table missing a stopping-power column would shift CSDA out of column 5.
    with pytest.raises(ValueError, match="seven columns"):
        module["estar_to_csv"]("Kinetic CSDA\n1.000E-01 3.0 3.1 1.000E-02 0.1 0.2\n")
    with pytest.raises(ValueError, match="CSDA"):
        module["estar_to_csv"]("1.000E-01 3.0 0.1 3.1 1.000E-02 0.1 0.2\n")
    output = tmp_path / "matrix.json"
    monkeypatch.setattr("sys.argv", ["build_matrix", "--output", str(output)])
    module["main"]()
    assert output.read_text() == (ROOT / "matrix.json").read_text()


def test_coverage_gaps_stay_explicit():
    gaps = runpy.run_path(str(ROOT / "audit_coverage.py"))["case_gaps"]
    domains = {
        "elsepa": [{"element": "W", "min_keV": 0.1, "max_keV": 1e5}],
        "sbethe": {"error": "TableNotFoundError: missing"},
        "bremslib": [{"element": "W", "min_keV": 0.01, "max_keV": 3e4}],
    }
    case = {"energy_keV": 5e4, "material": "tungsten", "physics": {}, "detector": {"a": 1}}
    found = gaps(case, domains, 10.0)
    assert "sbethe unavailable" in found
    assert "bremslib:W domain 0.01-30000 keV" in found
    assert any("per-segment scorer" in g for g in found)
