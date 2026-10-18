"""Contracts for isolated radiative-precision diagnostic variants."""

import ast
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/pyrite/montecarlo/transport/_jit_radiative_device.py"


def generator():
    return runpy.run_path(str(ROOT / "checks/equal_precision/precision_source.py"))[
        "precision_source"
    ]


def test_fp64_reference_source_is_unchanged():
    source = SOURCE.read_text(encoding="utf-8")
    assert generator()(source, "fp64") == source


@pytest.mark.parametrize("variant,iterations", [("rad-fp32-52", 52), ("rad-fp32-24", 24)])
def test_fp32_helpers_cast_inputs_and_keep_integer_streams_separate(variant, iterations):
    source = generator()(SOURCE.read_text(encoding="utf-8"), variant)
    tree = ast.parse(source)
    attributes = [
        a.attr
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        for a in ast.walk(n)
        if isinstance(a, ast.Attribute)
    ]
    assert "float64" not in attributes
    assert "float32" in attributes
    assert f"I32_BISECTION_STEPS = np.int32({iterations})" in source
    assert "energy_eV_input" in source
    assert "cutoff_eV_input" in source
    assert "uniform_input" in source
    imports = [
        n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "_jit_device"
    ]
    assert {a.name for n in imports for a in n.names} == {"I32_ONE", "I32_TWO", "I32_ZERO"}
    assert SOURCE.read_text(encoding="utf-8") != source


def test_mixed_cell_uses_double_only_inside_integral():
    tree = ast.parse(generator()(SOURCE.read_text(encoding="utf-8"), "rad-cell64-24"))
    double_functions = {
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any(isinstance(a, ast.Attribute) and a.attr == "float64" for a in ast.walk(n))
    }
    assert double_functions == {"_rad_cell_rate"}
    cell = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rad_cell_rate"
    )
    assert isinstance(cell.body[-1], ast.Return)
    assert ast.unparse(cell.body[-1].value).startswith("np.float32(")


def test_unknown_precision_variant_rejected():
    with pytest.raises(ValueError, match="precision variant"):
        generator()(SOURCE.read_text(encoding="utf-8"), "production-default")


@pytest.mark.parametrize("variant", ["rad-fp32-24", "rad-cell64-24", "rad-stable32-24"])
def test_constant_sdcs_cell_integral(variant):
    import numpy as np

    tree = ast.parse(generator()(SOURCE.read_text(encoding="utf-8"), variant))
    cell = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rad_cell_rate"
    )
    cell.decorator_list = []
    namespace = {"np": np, "xp": np, "F64_ZERO": np.float32(0)}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), "<precision-cell>", "exec"), namespace)
    actual = namespace["_rad_cell_rate"](
        np.float32(1), np.float32(2), np.float32(1.25), np.float32(1.25)
    )
    assert actual.dtype == np.dtype("float32")
    assert actual == pytest.approx(1.25 * np.log(2), rel=2 * np.finfo(np.float32).eps)


@pytest.mark.parametrize("variant", ["rad-cell64-24", "rad-stable32-24"])
def test_mixed_cell_resolves_close_photon_nodes(variant):
    import numpy as np

    tree = ast.parse(generator()(SOURCE.read_text(encoding="utf-8"), variant))
    cell = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rad_cell_rate"
    )
    cell.decorator_list = []
    namespace = {"np": np, "xp": np, "F64_ZERO": np.float32(0)}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), "<precision-cell>", "exec"), namespace)
    actual = namespace["_rad_cell_rate"](
        np.float32(1e6), np.float32(1e6 + 0.25), np.float32(1.25), np.float32(1.25)
    )
    assert actual == pytest.approx(1.25 * np.log1p(0.25 / 1e6), rel=2 * np.finfo(np.float32).eps)


def _toy_trajectory():
    import numpy as np

    return dict(
        electron_id=np.array([0, 0, 1]),
        event_kind=np.array([9, 5, 7]),
        E_start_keV=np.array([100.0, 70.0, 100.0]),
        E_end_keV=np.array([80.0, 60.0, 10.0]),
        hard_radiative_k_eV=np.array([10000.0, 0.0, 0.0]),
        v_hat=np.array([[0.0, 0.0, 1.0]] * 3),
        L_ang=np.ones(3),
    )


def test_precision_energy_budget_groups_all_segments_and_terminal_residual():
    import numpy as np

    checks = runpy.run_path(str(ROOT / "checks/equal_precision/precision_checks.py"))[
        "trajectory_checks"
    ]
    scores, report = checks(_toy_trajectory(), 2, 100.0, 1000.0)
    np.testing.assert_allclose(scores["continuous_loss_keV_per_primary"], [30, 90])
    np.testing.assert_allclose(scores["photon_energy_keV_per_primary"], [10, 0])
    np.testing.assert_allclose(scores["terminal_energy_keV_per_primary"], [60, 10])
    assert report["energy_ledger_max_error_keV"] == 0
    assert report["photon_debit_max_error_keV"] == 0


@pytest.mark.parametrize(
    "field,value", [("E_start_keV", 71.0), ("hard_radiative_k_eV", -1.0), ("v_hat", 0.0)]
)
def test_precision_budget_refuses_invalid_trajectories(field, value):
    checks = runpy.run_path(str(ROOT / "checks/equal_precision/precision_checks.py"))[
        "trajectory_checks"
    ]
    result = _toy_trajectory()
    result[field][1 if field == "E_start_keV" else 0] = value
    with pytest.raises(ValueError):
        checks(result, 2, 100.0, 1000.0)


@pytest.mark.parametrize("variant", ["rad-fp32-52", "rad-fp32-24", "rad-cell64-24"])
def test_exported_ev_to_kev_converter_preserves_double_precision(variant):
    import numpy as np

    tree = ast.parse(generator()(SOURCE.read_text(encoding="utf-8"), variant))
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "F64_KEV_PER_EV" for t in n.targets)
    )
    value = eval(compile(ast.Expression(assignment.value), "<unit-conversion>", "eval"), {"np": np})
    assert value.dtype == np.dtype("float64")
    assert value == np.float64(1e-3)
    row = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rad_row")
    assert "F32_KEV_PER_EV" in ast.unparse(row)


@pytest.mark.parametrize("failure", [None, "sample", "probe", "endpoint"])
def test_pilot_requires_successful_preflight_and_operating_cutoff_screen(tmp_path, failure):
    import json

    import numpy as np

    protocol = {
        "variants": ["fp64"],
        "full_screen_variants": ["fp64"] if failure == "endpoint" else [],
        "preflight": {
            "histories": 64,
            "seed": 57901,
            "moment_scaled_error_limit": 0.0002,
            "inverse_error_over_incident_energy_limit": 0.0002,
        },
    }
    sample = tmp_path / "si-64-57901-fp64.json"
    sample.write_text(
        json.dumps(
            {
                "schema": "pyrite.equal-precision.precision-sample.v1",
                "precision_source": {"variant_sha256": "frozen"},
            }
        )
    )
    (tmp_path / "probe-fp64.json").write_text(
        json.dumps(
            {
                "source": {"variant_sha256": "frozen"},
                "reports": [
                    {
                        "element": "Si",
                        "finite": True,
                        "nonnegative": True,
                        "photon_support": True,
                        "screening_passed": failure != "endpoint",
                    }
                ],
            }
        )
    )
    expected = np.array([[1.0, 2.0, 5000.0]])
    actual = expected.copy()
    if failure == "sample":
        sample.with_suffix(".failed.json").write_text("{}")
    if failure == "probe":
        actual[0, 0] += 0.1
    np.savez(
        tmp_path / "probe-fp64-Si.npz",
        cutoff_eV=[1000.0],
        energy_eV=[10000.0],
        expected=expected,
        actual=actual,
    )
    check = runpy.run_path(str(ROOT / "checks/equal_precision/precision_checks.py"))[
        "eligible_variants"
    ]
    eligible, reasons = check(tmp_path, protocol, [{"id": "si", "element": "Si"}])
    assert eligible == (["fp64"] if failure is None else [])
    assert bool(reasons) == (failure is not None)
