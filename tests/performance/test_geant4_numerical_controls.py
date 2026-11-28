"""Numerical ladders must audit effective settings, not echoed requests."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

HARNESS = Path(__file__).resolve().parents[2] / "checks/geant4_performance"
AUDIT = runpy.run_path(str(HARNESS / "geant4_bench.py"))["audit_log"]


def log(cutoff=2.5, bins=28, factor=0.02, elastic="gs"):
    processes = ["Transportation", "eIoni", "eBrem", "msc"]
    if elastic == "ss":
        processes[-1] = "CoulombScat"
    elif elastic == "wvi":
        processes.append("CoulombScat")
    return (
        "\n".join(f"G4BENCH_PROCESS e- {p}" for p in processes)
        + f"""
Material : BenchSi
Energy thresholds : gamma 1 keV e- 1 GeV
Lowest e+e- kinetic energy {cutoff} keV
Fluctuations of dE/dx are enabled 0
Step function for e+- (0.1, 0.25 mm)
G4BENCH_NUMERICS electron_cutoff_keV={cutoff} bins_per_decade={bins} msc_range_factor={factor} msc_theta_limit_rad=0 msc_skin=3
G4BENCH_TIMING init_s=1.0 warmup_s=0.2 measured_s=0.5 threads=1
"""
    )


def test_refined_settings_are_audited_with_units():
    bounds, _ = AUDIT(
        log(),
        "Si",
        "gs",
        expected=dict(
            electron_cutoff_keV=2.5,
            bins_per_decade=28,
            msc_range_factor=0.02,
            step_dr=0.1,
            step_final_um=250,
        ),
    )
    assert bounds["numerics"]["electron_cutoff_keV"] == 2.5
    assert bounds["numerics"]["step_final_um"] == 250


@pytest.mark.parametrize(
    "key,value",
    [
        ("electron_cutoff_keV", 10),
        ("bins_per_decade", 7),
        ("msc_range_factor", 0.08),
        ("step_dr", 0.2),
        ("step_final_um", 1000),
    ],
)
def test_silently_ignored_override_is_rejected(key, value):
    with pytest.raises(RuntimeError, match=key):
        AUDIT(log(), "Si", "gs", expected={key: value})


def test_missing_effective_audit_is_rejected():
    text = "\n".join(line for line in log().splitlines() if not line.startswith("G4BENCH_NUMERICS"))
    with pytest.raises(RuntimeError, match="numerical"):
        AUDIT(text, "Si", "gs")


def controls(**overrides):
    args = dict(
        step_dr=0.2,
        step_final_um=1000,
        electron_cutoff_kev=10,
        bins_per_decade=7,
        msc_range_factor=None,
        elastic="ss",
    )
    args.update(overrides)
    return runpy.run_path(str(HARNESS / "geant4_bench.py"))["numerical_controls"](
        SimpleNamespace(**args),
        300,
    )


def test_step_controls_are_independent():
    assert controls(step_final_um=125)["step_dr"] == 0.2
    assert controls(step_dr=0.025)["step_final_um"] == 1000


@pytest.mark.parametrize(
    "override",
    [
        dict(step_dr=0),
        dict(step_dr=1.1),
        dict(step_final_um=-1),
        dict(electron_cutoff_kev=300),
        dict(electron_cutoff_kev=float("nan")),
        dict(bins_per_decade=4),
        dict(bins_per_decade=1000000),
        dict(msc_range_factor=0.02),
        dict(elastic="gs", msc_range_factor=1),
        dict(elastic="wvi", msc_range_factor=float("inf")),
    ],
)
def test_invalid_controls_cannot_launch_a_sample(override):
    with pytest.raises(ValueError):
        controls(**override)


@pytest.mark.parametrize("elastic,factor", [("gs", 0.08), ("wvi", 0.04)])
def test_nominal_msc_settings_remain_explicit(elastic, factor):
    assert controls(elastic=elastic)["msc_range_factor"] == factor


@pytest.mark.parametrize(
    "override",
    [
        dict(step_dr=0.025),
        dict(step_final_um=125),
        dict(electron_cutoff_kev=1.25),
        dict(bins_per_decade=56),
        dict(elastic="gs", msc_range_factor=0.01),
    ],
)
def test_refinements_cannot_enter_nominal_timing_populations(override):
    accuracy_only = runpy.run_path(str(HARNESS / "geant4_bench.py"))["accuracy_only"]
    assert accuracy_only(controls(**override), override.get("elastic", "ss"))


@pytest.mark.parametrize("elastic", ["ss", "gs", "wvi"])
def test_nominal_settings_preserve_explicit_accuracy_only_flag(elastic):
    accuracy_only = runpy.run_path(str(HARNESS / "geant4_bench.py"))["accuracy_only"]
    nominal = controls(elastic=elastic)
    assert not accuracy_only(nominal, elastic)
    assert accuracy_only(nominal, elastic, requested=True)
