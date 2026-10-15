"""Endpoint coordinates must survive before evaluating FP32 cell math."""

import ast
import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def boundary_namespace():
    generate = runpy.run_path(str(ROOT / "checks/equal_precision/precision_source.py"))[
        "precision_source"
    ]
    source = generate(
        (ROOT / "src/pyrite/montecarlo/transport/_jit_radiative_device.py").read_text(
            encoding="utf-8"
        ),
        "rad-boundary64-24",
    )
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            n.decorator_list = []
    ns = {
        "np": np,
        "xp": np,
        "_MB_CM2": 1e-27,
        "F64_HALF": np.float64(0.5),
        "F64_ONE": np.float64(1),
        "F64_ZERO": np.float64(0),
        "F64_INF": np.float64(np.inf),
        "I32_ONE": np.int32(1),
        "I32_TWO": np.int32(2),
        "I32_ZERO": np.int32(0),
    }
    exec(compile(tree, "<boundary-precision>", "exec"), ns)
    return ns


def test_boundary_cell_preserves_width_smaller_than_float32_spacing():
    ns = boundary_namespace()
    left, right = np.float64(23643.516581539167), np.float64(23643.540225079392)
    value = ns["_rad_cell_rate"](left, right, np.float32(1.25), np.float32(1.25))
    assert value.dtype == np.dtype("float32")
    assert value == pytest.approx(1.25 * np.log1p((right - left) / left), rel=3e-7)


def test_boundary_constant_sdcs_moments_and_inverse_near_endpoint():
    ns = boundary_namespace()
    energy, cutoff = np.float64(23643.540225079392), np.float64(23643.516581539167)
    incident = np.array([10.0, 100.0])
    nominal = np.array([0.0, 0.5, 1.0])
    top = np.ones(2)
    chi = np.full(6, 1.25, dtype=np.float32)
    args = (
        incident,
        nominal,
        top,
        chi,
        np.int32(0),
        np.int32(2),
        np.int32(3),
        np.int32(2),
        np.float64(14),
        energy,
        cutoff,
    )
    hard = ns["_rad_moment"](ns["RAD_HARD_RATE"], *args)
    assert hard.dtype == np.dtype("float32")
    assert hard == pytest.approx(1e-27 * 14**2 * 1.25 * np.log(energy / cutoff), rel=4e-7)
    soft = ns["_rad_moment"](ns["RAD_SOFT"], *args)
    assert soft.dtype == np.dtype("float32")
    assert soft == pytest.approx(1e-27 * 14**2 * 1.25 * cutoff, rel=4e-7)
    photon = ns["_rad_sample_photon_eV"](*args, np.float64(0.5))
    assert cutoff <= photon <= energy
    assert photon == pytest.approx(np.sqrt(cutoff * energy), abs=energy * 1e-10)
    assert ns["F64_KEV_PER_EV"].dtype == np.dtype("float64")
